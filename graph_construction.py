from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.special import expit


def geographic_km(coords):
                             
    lon, lat = np.deg2rad(coords).T
    dl = lat[:, None] - lat[None, :]
    dn = lon[:, None] - lon[None, :]
    h = np.sin(dl/2)**2 + np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin(dn/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class GraphBuilder:
    def __init__(self, data, c):
        self.c, self.data = c, data
        self.n = len(data['nodes'])
        self.index = {v: i for i, v in enumerate(data['nodes'])}
        self.geo = geographic_km(data['coords'])
        self.external = None
        if c.external:
            self.external = np.load(c.external, allow_pickle=False)
            if list(self.external['nodes'].astype(str)) != data['nodes']:
                raise ValueError('External node order must exactly match prepared.npz nodes')
            if list(self.external['times'].astype(str)) != list(data['times'].astype(str)):
                raise ValueError('External timestamps must exactly match prepared.npz times')
            total = len(data['times'])
            for key, shape in [('activity', (total, self.n)), ('cancellation', (total, self.n, self.n)),
                               ('downstream', (self.n, self.n))]:
                if self.external[key].shape != shape:
                    raise ValueError(f'{key} must have shape {shape}')
            tau = self.external['tau_minutes']
            if tau.shape not in [(self.n, self.n), (total, self.n, self.n)]:
                raise ValueError('invalid tau_minutes shape')
            if c.mode == 'strict' and tau.ndim != 3:
                raise ValueError('strict mode requires time-varying directed travel times')
        if c.mode == 'strict' and self.external is None:
            raise ValueError('Strict mode needs external.npz: road_km, directed tau_minutes, downstream, '
                             'activity, cancellation, route-specific survey calibration inputs. See README.')
        train = data['records'].loc[data['records'].bin < data['times'][data['split'][0]]].copy()
        self.st = np.zeros((self.n, self.n), dtype='float32')
        travel = np.full((self.n, self.n), np.inf)
        group = ['date_operationnelle', 'ligne', 'voyage', 'vehicule']
        train = train.sort_values(group + ['seq'])
        nxt = train.groupby(group, sort=False).shift(-1)
        for idx, row in train.iterrows():
            other = nxt.loc[idx]
            if other.get('arret_code') not in self.index or other.get('seq') != row.seq + 1:
                continue
            i, j = self.index[row.arret_code], self.index[other.arret_code]
            minutes = (other.time - row.time).total_seconds() / 60
            if i != j and minutes > 0:
                self.st[i, j] = 1
                travel[i, j] = min(travel[i, j], minutes)
        self.tau = shortest_path(travel, directed=True)
        self.road = shortest_path(np.where(self.st > 0, self.geo, np.inf), directed=True)
        self.downstream = np.isfinite(self.tau) & (self.tau > 0)
        self.alpha, self.mu = c.alpha, c.mu_km
        if self.external is not None:
            self.road = self.external['road_km']
            self.downstream = self.external['downstream'].astype(bool)
            if self.road.shape != (self.n, self.n):
                raise ValueError('road_km shape mismatch')
        self.history = data['clean'][:data['split'][0]]
        self.rho = np.nan_to_num(np.corrcoef(self.history.T))
        self.attract = self.history.mean(axis=0)
        self.calibration = None

    def calibrate(self):
                                          
        if self.external is None:
            return
        from parameters import calibrate_survey
        self.calibration = calibrate_survey(self.external, self.c)

    def at(self, t):
        c, n = self.c, self.n
        if self.external is not None:
            active = self.external['activity'][t].astype(bool)
            cancel = self.external['cancellation'][t].astype(bool)
            tau_data = self.external['tau_minutes']
            tau = tau_data[t] if tau_data.ndim == 3 else tau_data
        else:
                                                
            active = self.data.get('service_observed', self.data['observed'])[t]
            cancel = np.zeros((n, n), bool)
            tau = self.tau
        availability = active[:, None] & active[None, :]
        st = self.st * (~cancel)
        bnp = np.where((tau > 0) & (tau <= c.delta_minutes), np.exp(-np.minimum(tau, 1e10)**2/c.sigma_minutes**2), 0)
        spatial = np.exp(-self.geo / max(c.spatial_radius_km, 1e-8)) * (self.geo <= c.spatial_radius_km)
        np.fill_diagonal(spatial, 0)
        if self.calibration is None:
            weights = expit(self.alpha*(np.nan_to_num(self.road, posinf=1e10)-self.mu)) * np.log1p(self.attract)[None, :]
            weights *= self.downstream
            fe = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        else:
            fe = np.zeros((n, n))
            for mask, distance, alpha, mu in self.calibration:
                weights = expit(alpha*(np.nan_to_num(distance, posinf=1e10)-mu)) * np.log1p(self.attract)[None, :] * mask
                fe += weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        rel = np.stack([st, fe, spatial]) * availability[None]
        support = ((st + fe + spatial) > 0) & availability
                                                  
        edge = support & (bnp > 0)
        adj = edge.astype(float)
        overlap = adj @ adj.T / np.maximum(np.sqrt(adj.sum(1)[:, None]*adj.sum(1)[None, :]), 1)
        score = np.exp(-self.road/c.distance_decay_km) + self.rho + overlap
        reach = np.eye(n, dtype=bool)
        selections = []
        for _ in range(c.max_hop):
            reach = (reach.astype(float) @ adj > 0) | reach
            candidates = reach.copy()
            np.fill_diagonal(candidates, False)
            s = np.where(candidates, score, -np.inf)
            ids = np.argsort(s, axis=1)[:, -min(c.top_m, n):]
            mask = np.zeros((n, n), dtype='float32')
            rows = np.arange(n)[:, None]
            mask[rows, ids] = np.isfinite(s[rows, ids])
            selections.append(mask)
        return rel.astype('float32'), bnp.astype('float32'), support.astype('float32'), np.stack(selections)
