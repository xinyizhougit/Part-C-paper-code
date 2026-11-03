from dataclasses import dataclass
from pathlib import Path
import argparse
import json


@dataclass
class Config:
                                  
    data: str = r'D:\PH.D\data\EXO_APC_AVL_202309_202310.csv'
    region: str = 'CITLA'
    output: str = 'runs/CITLA'
    mode: str = 'strict'
    external: str = ''
    history: int = 12
    hidden: int = 64
    lr: float = 0.001
    epochs: int = 200
    patience: int = 15
    batch_size: int = 64
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    knn: int = 5
    gmm_components: int = 3
    gmm_density: float = 0.0001
    max_hop: int = 3
    top_m: int = 10
    distance_decay_km: float = 2.0
    spatial_radius_km: float = 0.5
    sigma_minutes: float = 15.0
    delta_minutes: float = 60.0
    alpha: float = 1.0
    mu_km: float = 5.0
    rpca_rank: int = 0
    rpca_iterations: int = 50
    rpca_tol: float = 0.0001
    rank_threshold: float = 0.05
    attention_iterations: int = 3
    device: str = 'cpu'
    max_nodes: int = 0
                                     
    micro_batch: int = 1
    dropout: float = 0.1
    rsvd_oversampling: int = 10
    rsvd_power_iterations: int = 2
    seeds: str = '42,123,1024,2026,3407'
    grid_hidden: str = '32,64,128,256'
    grid_lr: str = '0.0001,0.0005,0.001,0.005,0.01'
    benchmark_warmup: int = 100
    benchmark_iterations: int = 1000
    checkpoint: str = ''
    baseline_predictions: str = ''
    target_policy: str = 'observed'
    expected_start: str = '2023-09-01'
    expected_end: str = '2023-10-31'



def parse_config():
    p = argparse.ArgumentParser()
    for key, value in Config().__dict__.items():
        p.add_argument('--' + key.replace('_', '-'), type=type(value), default=value)
    p.add_argument('--action', choices=['prepare', 'train', 'grid', 'repeat', 'benchmark', 'paired'], default='repeat')
    args = vars(p.parse_args())
    action = args.pop('action')
    c = Config(**args)
    if c.mode not in ('strict', 'exploratory'):
        p.error('mode must be strict or exploratory')
    if not 0 < c.train_ratio < c.train_ratio + c.val_ratio < 1:
        p.error('invalid temporal split')
    if c.history < 1 or c.batch_size < 1 or c.max_hop < 1:
        p.error('history, batch_size and max_hop must be positive')
    if c.sigma_minutes <= 0 or c.delta_minutes <= 0:
        p.error('BNP parameters must be positive')
    if c.micro_batch < 1 or c.micro_batch > c.batch_size:
        p.error('micro_batch must be between 1 and batch_size')
    if c.target_policy not in ('observed', 'zero_grid'):
        p.error('target_policy must be observed or zero_grid')
    Path(c.output).mkdir(parents=True, exist_ok=True)
    Path(c.output, 'config.json').write_text(json.dumps(c.__dict__, indent=2), encoding='utf-8')
    return c, action


def calibrate_survey(e, c):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
                                   
    routes = e['route_ids'].astype(str)
    rng = np.random.default_rng(c.seed)
    results, report = [], []
    bins = e['distance_bins_km']
    if bins[0] > 0 or not np.isinf(bins[-1]):
        raise ValueError('distance_bins_km must cover [0, infinity)')
    for r, name in enumerate(routes):
        sample = e['survey_distance_km'][e['survey_route'].astype(str) == name]
        if len(sample) < 10 or np.any(~np.isfinite(sample)) or np.any(sample < 0):
            raise ValueError(f'Insufficient/invalid mapped survey samples for {name}')
        sample = rng.permutation(sample)
        split = int(0.7 * len(sample))
        emp = np.histogram(sample[:split], bins)[0].astype(float)
        held = np.histogram(sample[split:], bins)[0].astype(float)
        emp, held = emp / emp.sum(), held / held.sum()
        mask = e['route_downstream'][r].astype(bool)
        distance = e['route_distance_km'][r]
        attraction = e['route_attractiveness'][r]
        origin = e['route_origin_weight'][r]
        if np.any(attraction < 0) or np.any(origin < 0) or origin.sum() <= 0:
            raise ValueError('invalid attraction/origin weight')
        valid = mask & np.isfinite(distance) & (distance > 0)
        if np.any((origin > 0) & ~valid.any(axis=1)):
            raise ValueError('positive origin weight without downstream destinations')
        def distribution(logparams):
            alpha, mu = np.exp(logparams)
            w = expit(alpha*(np.nan_to_num(distance, posinf=1e10)-mu)) * np.log1p(attraction)[None, :] * valid
            p = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
            mass = p * (origin / origin.sum())[:, None]
            q = np.histogram(distance[valid], bins, weights=mass[valid])[0] + 1e-12
            return q / q.sum()
        def kl(p, q):
            nz = p > 0
            return float(np.sum(p[nz]*np.log(p[nz]/q[nz])))
        fits = [minimize(lambda z: kl(emp, distribution(z)), np.log([a, m]),
                         method='L-BFGS-B', bounds=[(-8, 8), (-8, 8)])
                for a in (0.1, 1., 5.) for m in (1., 5., 15.)]
        good = [f for f in fits if f.success and np.isfinite(f.fun)]
        if not good:
            raise RuntimeError(f'Calibration failed for route {name}')
        best = min(good, key=lambda f: f.fun)
        alpha, mu = np.exp(best.x)
        results.append((valid, distance, alpha, mu))
                                          
        report.append(dict(route=name, alpha=alpha, mu=mu, training_kl=best.fun,
                           holdout_kl=kl(held, distribution(best.x)),
                           starts=[dict(loss=float(f.fun), success=bool(f.success), parameters=np.exp(f.x).tolist()) for f in fits]))
    Path(c.output, 'survey_calibration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return results
