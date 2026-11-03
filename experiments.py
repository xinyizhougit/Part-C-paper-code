from pathlib import Path
import json
import numpy as np
from scipy.stats import ttest_rel


def summarize_runs(roots, output):
                                              
    values = [json.loads(Path(root, 'test_metrics.json').read_text()) for root in roots]
    summary = {}
    for h in ('15', '30', '45', '60'):
        summary[h] = {}
        for metric in ('MAE', 'RMSE', 'WMAPE'):
            a = [v[h][metric] for v in values]
            if any(x is None for x in a):
                summary[h][metric] = dict(mean=None, std=None, runs=len(a))
            else:
                summary[h][metric] = dict(mean=float(np.mean(a)),
                    std=float(np.std(a, ddof=1)) if len(a) > 1 else None, runs=len(a))
    Path(output, 'five_seed_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def paired_days(ours_path, baseline_path, output):
                                         
    a, b = np.load(ours_path), np.load(baseline_path)
    for key in ('y_true', 'mask', 'origins', 'nodes', 'target_times'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'Baseline is not evaluated on identical {key}')
    if a['y_pred'].shape != b['y_pred'].shape:
        raise ValueError('Prediction shapes differ')
    report = {}
    for h in range(4):
        days = np.char.substr(a['target_times'][:, h], 0, 10) if hasattr(np.char, 'substr') else np.array([t[:10] for t in a['target_times'][:, h]])
        pairs = []
        for day in np.unique(days):
            valid = a['mask'][days == day, :, h].astype(bool)
            y = a['y_true'][days == day, :, h]
            ap, bp = a['y_pred'][days == day, :, h], b['y_pred'][days == day, :, h]
            if np.any(valid & (~np.isfinite(ap) | ~np.isfinite(bp) | ~np.isfinite(y))):
                raise ValueError('Non-finite values on the shared paired-test mask')
            if valid.any():
                pairs.append((str(day), float(np.abs(y-ap)[valid].mean()), float(np.abs(y-bp)[valid].mean())))
        if len(pairs) < 2:
            raise ValueError('Need at least two independent test days')
        result = ttest_rel([p[2] for p in pairs], [p[1] for p in pairs])
        report[str((h+1)*15)] = dict(unit='target day', statistic='daily MAE; baseline minus ours',
            n_days=len(pairs), df=len(pairs)-1,
            t=float(result.statistic) if np.isfinite(result.statistic) else None,
            p_two_sided=float(result.pvalue) if np.isfinite(result.pvalue) else None, pairs=pairs)
    Path(output, 'paired_test.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def benchmark(data, builder, c):
    import time
    import torch
    from model import DTSSM
    from training import Windows, make_graphs
    if not c.checkpoint:
        raise ValueError('--checkpoint must identify a trained best.pt')
    checkpoint = torch.load(c.checkpoint, map_location=c.device, weights_only=True)
    from parameters import Config
    from dataclasses import replace
    saved = Config(**checkpoint['config'])
    saved = replace(saved, device=c.device)
    if checkpoint['nodes'] != data['nodes'] or saved.history != c.history:
        raise ValueError('Checkpoint nodes/history mismatch')
    model = DTSSM(saved).to(c.device).eval()
    model.load_state_dict(checkpoint['model'])
    windows = Windows(data, builder, c, 'test')
    if len(windows) < c.batch_size:
        raise ValueError('Not enough test windows for specified inference batch')
    def sync():
        if c.device.startswith('cuda'):
            torch.cuda.synchronize()
    def step(iteration):
                                          
        origins = [windows.origins[(iteration*c.batch_size+j) % len(windows)] for j in range(c.batch_size)]
        for start in range(0, len(origins), c.micro_batch):
            ids = origins[start:start+c.micro_batch]
            graphs = make_graphs(builder, ids, c)
            x = torch.tensor(np.stack([data['x'][t-c.history+1:t+1] for t in ids]), device=c.device)
            output = model(x, graphs)
            mean = torch.tensor(checkpoint['mean'], device=c.device)
            scale = torch.tensor(checkpoint['scale'], device=c.device)
            output = output*scale[None, :, None]+mean[None, :, None]
    if c.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    elapsed = []
    with torch.no_grad():
        for i in range(c.benchmark_warmup+c.benchmark_iterations):
            sync(); begin = time.perf_counter()
            step(i)
            sync()
            if i >= c.benchmark_warmup:
                elapsed.append((time.perf_counter()-begin)*1000)
    report = dict(device=c.device, batch_size=c.batch_size, physical_micro_batch=c.micro_batch,
        warmup=c.benchmark_warmup, iterations=c.benchmark_iterations,
        milliseconds_per_logical_batch=float(np.mean(elapsed)),
        amortized_milliseconds_per_sample=float(np.mean(elapsed)/c.batch_size),
        p95_milliseconds_per_batch=float(np.percentile(elapsed, 95)),
        parameters=sum(p.numel() for p in model.parameters()),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if c.device.startswith('cuda') else None,
        paper_physical_batch_protocol=bool(c.batch_size == c.micro_batch == 64),
        scope='graph construction + device transfer + RPCA + full forward + inverse normalization')
    Path(c.output, 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
