from pathlib import Path
from dataclasses import replace
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import DTSSM
from indicators import calculate_metrics


class Windows(Dataset):
    def __init__(self, data, builder, c, split):
        self.data, self.builder, self.c = data, builder, c
        a, b = data['split']
        lo, hi = {'train': (0, a), 'val': (a, b), 'test': (b, len(data['x']))}[split]
                                         
        self.origins = [t for t in range(max(c.history-1, lo-1), hi-4)
                        if t+1 >= lo and data['observed'][t+1:t+5].any()]
        if not self.origins:
            raise ValueError(f'No windows for {split}')

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        t = self.origins[idx]
        l = self.c.history
        y = self.data['raw'][t+1:t+5].T.astype('float32')
        mask = self.data['observed'][t+1:t+5].T & np.isfinite(y)
        return (self.data['x'][t-l+1:t+1], np.nan_to_num(y), mask, t)


def make_graphs(builder, origins, c):
                                          
    samples = []
    for t in origins:
        graph = [builder.at(s) for s in range(int(t)-c.history+1, int(t)+1)]
        samples.append(tuple(np.stack([g[k] for g in graph]) for k in range(4)))
    return [torch.tensor(np.stack([g[k] for g in samples]), device=c.device) for k in range(4)]


def evaluate(model, loader, c, mean, scale, save=False):
    model.eval()
    ys, ps, masks, origins = [], [], [], []
    with torch.no_grad():
        for x, y, mask, t in loader:
            for start in range(0, len(x), c.micro_batch):
                stop = start+c.micro_batch
                graphs = make_graphs(loader.dataset.builder, t[start:stop], c)
                pred = model(x[start:stop].to(c.device), graphs)*scale[None, :, None]+mean[None, :, None]
                                           
                ys.append(y[start:stop].numpy()); ps.append(pred.cpu().numpy())
                masks.append(mask[start:stop].numpy()); origins.extend(t[start:stop].tolist())
    y, p, mask = np.concatenate(ys), np.concatenate(ps), np.concatenate(masks)
    metrics = {str(15*(h+1)): calculate_metrics(y[:, :, h], p[:, :, h], mask[:, :, h]) for h in range(4)}
    if save:
        times = loader.dataset.data['times']
        target_times = np.array([[str(times[o+h]) for h in range(1, 5)] for o in origins])
        np.savez_compressed(Path(c.output, 'test_predictions.npz'), y_true=y, y_pred=p, mask=mask,
                            origins=origins, nodes=np.array(loader.dataset.data['nodes']), target_times=target_times)
        Path(c.output, 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def train(data, builder, c, test=True):
    random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loaders = {s: DataLoader(Windows(data, builder, c, s), batch_size=c.batch_size, shuffle=s=='train') for s in ('train', 'val', 'test')}
    model = DTSSM(c).to(c.device)
    if c.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    mean = torch.tensor(data['mean'], device=c.device)
    scale = torch.tensor(data['scale'], device=c.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
    best, stale, history = float('inf'), 0, []
    Path(c.output).mkdir(parents=True, exist_ok=True)
    for epoch in range(c.epochs):
        model.train()
        total, count = 0., 0
        epoch_start = time.perf_counter()
        for x, y, mask, origins in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
                                           
            denominator = mask.sum((0, 1)).to(c.device)
            if (denominator == 0).any():
                raise ValueError('A logical batch has a horizon without targets; check evaluation mask')
            batch_loss = 0.
            for start in range(0, len(x), c.micro_batch):
                stop = start+c.micro_batch
                graphs = make_graphs(builder, origins[start:stop], c)
                pred = model(x[start:stop].to(c.device), graphs)*scale[None, :, None]+mean[None, :, None]
                target = y[start:stop].to(c.device)
                valid = mask[start:stop].to(c.device)
                loss = ((torch.abs(pred-target)*valid).sum((0, 1))/denominator).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite loss')
                loss.backward()
                batch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += batch_loss; count += 1
        if c.device.startswith('cuda'):
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter()-epoch_start
        metrics = evaluate(model, loaders['val'], c, mean, scale)
        usable = [v['MAE'] for v in metrics.values() if v['MAE'] is not None]
        if not usable:
            raise ValueError('No observed validation targets')
        val = float(np.mean(usable))
        history.append(dict(epoch=epoch+1, train_uniform_horizon_mae=total/count, val_mae=val, training_seconds=epoch_seconds))
        print(history[-1], flush=True)
        if val < best:
            best, stale = val, 0
            torch.save(dict(model=model.state_dict(), config=c.__dict__, nodes=data['nodes'],
                            mean=data['mean'].tolist(), scale=data['scale'].tolist(), best_val=best), Path(c.output, 'best.pt'))
        else:
            stale += 1
        if stale >= c.patience:
            break
    Path(c.output, 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    model.load_state_dict(torch.load(Path(c.output, 'best.pt'), map_location=c.device, weights_only=True)['model'])
    if test:
        evaluate(model, loaders['test'], c, mean, scale, save=True)
        Path(c.output, 'rpca_diagnostics.json').write_text(json.dumps(model.rpca.last_diagnostics, indent=2), encoding='utf-8')
    Path(c.output, 'resources.json').write_text(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if c.device.startswith('cuda') else None,
        logical_batch=c.batch_size, micro_batch=c.micro_batch), indent=2), encoding='utf-8')
    return best, model, loaders['test'], mean, scale


def grid_search(data, builder, c):
                               
    scores, winner = [], None
    for h in map(int, c.grid_hidden.split(',')):
        for lr in map(float, c.grid_lr.split(',')):
            cc = replace(c, hidden=h, lr=lr, output=str(Path(c.output, f'h{h}_lr{lr}')))
            score = train(data, builder, cc, test=False)[0]
            scores.append(dict(hidden=h, lr=lr, validation_mae=score, output=cc.output))
            if winner is None or score < winner['validation_mae']:
                winner = scores[-1]
    Path(c.output, 'winner.json').write_text(json.dumps(winner, indent=2), encoding='utf-8')
    Path(c.output, 'grid_results.json').write_text(json.dumps(scores, indent=2), encoding='utf-8')
    bestc = replace(c, hidden=winner['hidden'], lr=winner['lr'], output=winner['output'])
                                
    repeat_runs(data, builder, replace(bestc, output=str(Path(c.output, 'selected_repeats'))))


def repeat_runs(data, builder, c):
    from experiments import summarize_runs
    roots = []
    for seed in map(int, c.seeds.split(',')):
        cc = replace(c, seed=seed, output=str(Path(c.output, f'seed_{seed}')))
        train(data, builder, cc)
        roots.append(cc.output)
    summarize_runs(roots, c.output)
