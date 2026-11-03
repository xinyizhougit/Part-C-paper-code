import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist


ALIASES = {'OMITSJ': 'OMITSJU', 'CTLSV': 'CITSV'}


def load_region(c):
                                   
    chunks = []
    region = ALIASES.get(c.region, c.region)
    for x in pd.read_csv(c.data, dtype=str, chunksize=100000):
        chunks.append(x.loc[x.organisation.eq(region)])
    d = pd.concat(chunks, ignore_index=True)
    if d.empty:
        raise ValueError(f'No rows for {region}')
                                    
    d['time'] = pd.to_datetime(d.date_heure_depart_prevue, errors='coerce')
    d['flow'] = pd.to_numeric(d.montant, errors='coerce')
    d['seq'] = pd.to_numeric(d.sequence_arret, errors='coerce')
    xy = d.arret_geom.str.extract(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)')
    d['lon'], d['lat'] = pd.to_numeric(xy[0], errors='coerce'), pd.to_numeric(xy[1], errors='coerce')
    d = d.dropna(subset=['time', 'arret_code', 'seq', 'lon', 'lat'])
    d.loc[d.flow.lt(0), 'flow'] = np.nan
                                    
    d['bin'] = d.time.dt.floor('15min')
    return d.sort_values('time').reset_index(drop=True)


class Cleaner:
    def fit(self, x, c):
                                             
        self.q1, self.q3 = np.nanquantile(x, [0.25, 0.75], axis=0)
        self.med = np.nanmedian(x, axis=0)
        self.med = np.nan_to_num(self.med)
        self.k = c.knn
        hist = np.where(np.isfinite(x), x, self.med)
        self.dist = cdist(hist.T, hist.T) / np.sqrt(len(x))
        np.fill_diagonal(self.dist, np.inf)
        self.neighbors = np.argsort(self.dist, axis=1)
        z = self.impute(self.mask_iqr(x))
        self.gmm = GaussianMixture(n_components=min(c.gmm_components, len(z.ravel())), random_state=c.seed)
        self.gmm.fit(z.reshape(-1, 1))
        self.threshold = c.gmm_density
        return self

    def mask_iqr(self, x):
        width = self.q3 - self.q1
                                             
        bad = (width > 0) & ((x < self.q1 - 1.5 * width) | (x > self.q3 + 1.5 * width))
        return np.where(bad, np.nan, x)

    def impute(self, x):
        out = x.copy()
        for t, v in zip(*np.where(~np.isfinite(x))):
            ids = self.neighbors[v]
            ids = ids[np.isfinite(x[t, ids]) & np.isfinite(self.dist[v, ids])][:self.k]
            if len(ids):
                w = 1 / (self.dist[v, ids] + 1e-8)
                out[t, v] = np.dot(w, x[t, ids]) / w.sum()
            else:
                out[t, v] = self.med[v]
        return out

    def transform(self, x):
        z = self.impute(self.mask_iqr(x))
        bad = self.gmm.score_samples(z.reshape(-1, 1)).reshape(z.shape) < np.log(max(self.threshold, 1e-30))
        return np.maximum(self.impute(np.where(bad, np.nan, z)), 0).astype('float32')


def prepare(c):
    d = load_region(c)
    timeline = pd.date_range(d.bin.min(), d.bin.max(), freq='15min')
    a = int(len(timeline) * c.train_ratio)
    b = int(len(timeline) * (c.train_ratio + c.val_ratio))
    if a <= c.history + 4 or b - a < 4 or len(timeline) - b < 4:
        raise ValueError('Not enough time bins for history and temporal splits')
    train = d.loc[d.bin < timeline[a]]
                                    
    nodes = sorted(train.arret_code.unique())
    if c.max_nodes:
        if c.mode == 'strict':
            raise ValueError('max_nodes is only allowed for exploratory smoke runs')
        nodes = nodes[:c.max_nodes]
    unknown = int((~d.arret_code.isin(nodes)).sum())
    d = d.loc[d.arret_code.isin(nodes)].copy()
    counts = d.groupby(['bin', 'arret_code']).flow.sum(min_count=1).unstack().reindex(index=timeline, columns=nodes)
    observed = d.groupby(['bin', 'arret_code']).size().unstack(fill_value=0).reindex(index=timeline, columns=nodes, fill_value=0).fillna(0).to_numpy() > 0
                                        
    service_observed = observed.copy()
    raw = counts.to_numpy(dtype=float)
                                              
    if c.target_policy == 'zero_grid':
        raw = np.where(observed, raw, 0.)
        observed = np.isfinite(raw)
    coverage = dict(first=str(d.time.min()), last=str(d.time.max()),
                    expected_start=c.expected_start, expected_end=c.expected_end,
                    covers_requested_span=bool(d.time.min().normalize() <= pd.Timestamp(c.expected_start)
                                               and d.time.max().normalize() >= pd.Timestamp(c.expected_end)))
    cleaner = Cleaner().fit(raw[:a], c)
    clean = cleaner.transform(raw)
    scale = clean[:a].std(axis=0)
    scale[scale < 1e-6] = 1
    mean = clean[:a].mean(axis=0)
    coords = train.groupby('arret_code')[['lon', 'lat']].median().reindex(nodes).to_numpy()
    report = {'rows_retained': len(d), 'nodes': len(nodes), 'bins': len(timeline), 'train_end': str(timeline[a]),
              'validation_end': str(timeline[b]), 'unknown_or_excluded_rows': unknown,
              'time_source': 'scheduled departure; actual strings lack hour/date',
              'target': 'montant (boarding), raw observed targets; descendant unused',
              'missing_target_policy': c.target_policy, 'coverage': coverage, 'mode': c.mode}
    Path(c.output, 'data_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    import joblib
                                     
    joblib.dump(cleaner, Path(c.output, 'cleaner.joblib'))
    np.savez_compressed(Path(c.output, 'prepared.npz'), x=(clean-mean)/scale, clean=clean,
                        raw=raw, observed=observed, mean=mean, scale=scale, nodes=np.array(nodes),
                        times=timeline.astype(str).to_numpy(dtype=str), split=[a, b], coords=coords)
    export_contract(raw, observed, timeline, nodes, (a, b), c)
    return dict(x=((clean-mean)/scale).astype('float32'), clean=clean, raw=raw, observed=observed,
                mean=mean.astype('float32'), scale=scale.astype('float32'), nodes=nodes, service_observed=service_observed,
                times=timeline, split=(a, b), coords=coords, records=d)


def export_contract(raw, observed, times, nodes, split, c):
                                    
    import hashlib
    for name, lo, hi in [('train', 0, split[0]), ('val', split[0], split[1]), ('test', split[1], len(times))]:
        origins = [t for t in range(max(c.history-1, lo-1), hi-4)
                   if t+1 >= lo and observed[t+1:t+5].any()]
        if not origins:
            raise ValueError(f'No canonical windows for {name}')
        y = np.stack([raw[t+1:t+5].T for t in origins]).astype('float32')
        mask = np.stack([observed[t+1:t+5].T for t in origins]) & np.isfinite(y)
        y = np.nan_to_num(y)
        stamps = np.array([[str(times[t+h]) for h in range(1, 5)] for t in origins])
        tod = np.array([times[t].hour*4+times[t].minute//15 for t in origins])
        dow = np.array([times[t].dayofweek for t in origins])
        digest = hashlib.sha256(y.tobytes()+mask.tobytes()+np.array(origins).tobytes()).hexdigest()
        np.savez_compressed(Path(c.output, f'{name}_evaluation_contract.npz'), y_true=y, mask=mask,
                            origins=origins, nodes=np.array(nodes), target_times=stamps,
                            time_of_day=tod, day_of_week=dow, sha256=np.array(digest))
