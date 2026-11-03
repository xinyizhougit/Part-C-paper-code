import torch
from torch import nn
from torch.nn import functional as F


def shrink(x, threshold):
    return x.sign() * (x.abs()-threshold).clamp_min(0)


class RPCA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.last_diagnostics = []

    def forward(self, a, support, bnp, state=None):
                                              
        n = a.shape[-1]
        norm = a.norm().clamp_min(1e-8)
        penalty = 1.25 / a.detach().norm().clamp_min(1e-8)
        lam = n ** -0.5
        if state is None:
            l, s, y = (torch.zeros_like(a) for _ in range(3))
        else:
            l, s, y = (v.detach() for v in state)
        residual = float('inf')
        for k in range(self.c.rpca_iterations):
            temp = a-s+y/penalty
            active_n = int((support.any(-1) | support.any(-2)).sum().item())
            target = self.c.rpca_rank or max(30, int(0.05*active_n))
            target = min(target, n)
            rank = min(target+self.c.rsvd_oversampling, n)
                                             
            if rank == n:
                u, sig, vh = torch.linalg.svd(temp, full_matrices=False)
            else:
                u, sig, v = torch.svd_lowrank(temp, q=rank, niter=self.c.rsvd_power_iterations)
                vh = v.transpose(-1, -2)
            keep = min(target, len(sig))
            l = (u[:, :keep] * (sig[:keep]-1/penalty).clamp_min(0)) @ vh[:keep]
            s = shrink(a-l+y/penalty, lam/penalty)
            err = a-l-s
            y = y+penalty*err
            residual = float((err.norm()/norm).detach())
            if residual < self.c.rpca_tol:
                break
        self.last_diagnostics.append(dict(iterations=k+1, residual=residual, converged=residual < self.c.rpca_tol))
        valid = (l >= self.c.rank_threshold) & (support > 0) & (bnp > 0)
        return l*valid, (l, s, y)


def projection(input_dim, hidden, dropout, output_dim=None):
                                         
    layers = [nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
              nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
    if output_dim is not None:
        layers.append(nn.Linear(hidden, output_dim))
    return nn.Sequential(*layers)


class DTSSM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        h = c.hidden
                                             
        self.gamma = nn.Parameter(torch.zeros(3))
        self.rpca = RPCA(c)
        self.temporal = nn.Linear(c.history, h)
        self.scales = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(c.max_hop)])
        self.spatial_fuse = projection(c.max_hop*h, h, c.dropout)
        self.flow = nn.Linear(h, h, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.attention = nn.Linear(h, h, bias=False)
        self.attention_out = nn.Linear(h, h, bias=False)
        self.graph = nn.Linear(h, h, bias=False)
        self.z, self.r, self.candidate = nn.Linear(2*h, h), nn.Linear(2*h, h), nn.Linear(2*h, h)
        self.convs = nn.ModuleList([nn.Conv1d(h, h, k) for k in (3, 5, 7)])
        self.conv_fuse = projection(3*h, h, c.dropout)
        self.heads = nn.ModuleList([projection(2*h, h, c.dropout, 1) for _ in range(4)])

    def forward(self, x, graphs):
                                               
        batch, length, n = x.shape
        if length != self.c.history:
            raise ValueError('input must contain exactly L historical steps')
        h = x.new_zeros(batch, n, self.c.hidden)
        hist, states = [], [None]*batch
        self.rpca.last_diagnostics = []
        for t in range(self.c.history):
            rel, bnp, support, selection = [v[:, t] for v in graphs]
            a = (rel * self.gamma.softmax(0)[None, :, None, None]).sum(1) * bnp
            cleaned = []
            for b in range(batch):
                clean, states[b] = self.rpca(a[b], support[b], bnp[b], states[b])
                cleaned.append(clean)
            clean = torch.stack(cleaned)
            normalized = clean / clean.sum(-1, keepdim=True).clamp_min(1e-8)
                                           
            prefix = F.pad(x[:, :t+1].transpose(1, 2), (self.c.history-t-1, 0))
            emb = self.temporal(prefix)
            scales = [F.gelu(selection[:, k] @ layer(emb)) for k, layer in enumerate(self.scales)]
            spatial = F.gelu(self.spatial_fuse(torch.cat(scales, -1)))
            current = x[:, t]
            diff = (current[:, :, None]-current[:, None, :]).abs()
            weight = normalized * torch.tanh(diff/(F.softplus(self.temperature)+1e-6))
            z = F.gelu(weight @ self.flow(spatial))
            q = self.attention(z)
            score = q @ q.transpose(-1, -2)
            mask = clean > 0
                                                
            logalpha = torch.zeros_like(score)
            safe = mask.any(-1, keepdim=True)
            for _ in range(self.c.attention_iterations):
                logits = (logalpha+score).masked_fill(~mask, -torch.inf)
                logits = torch.where(safe, logits, torch.zeros_like(logits))
                logalpha = torch.log_softmax(logits, dim=-1)
            att = logalpha.exp()*mask
            final = att @ self.attention_out(z)
            g = self.graph(clean @ final)
            gate_z = torch.sigmoid(self.z(torch.cat([g, h], -1)))
            gate_r = torch.sigmoid(self.r(torch.cat([g, h], -1)))
            candidate = torch.tanh(self.candidate(torch.cat([g, gate_r*h], -1)))
            h = (1-gate_z)*h+gate_z*candidate
            hist.append(h)
                                                      
        sequence = torch.stack(hist, dim=2).permute(0, 1, 3, 2).reshape(batch*n, self.c.hidden, self.c.history)
        features = [F.gelu(conv(F.pad(sequence, (conv.kernel_size[0]-1, 0)))[:, :, -1]) for conv in self.convs]
        conv = F.gelu(self.conv_fuse(torch.cat(features, -1))).reshape(batch, n, self.c.hidden)
        fused = torch.cat([h, conv], -1)
        return torch.cat([head(fused) for head in self.heads], -1)
