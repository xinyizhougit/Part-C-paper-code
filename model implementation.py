
# 3. Dynamic Spatiotemporal Transit Forecasting Model (DTSSM)

import torch
import torch.nn as nn
import numpy as np

# [INPUT]
INPUT_X_SEQ = ""
INPUT_A_StE = ""
INPUT_A_FE = ""
INPUT_A_SpE = ""
INPUT_BNP = ""
INPUT_A_MASK = ""

class DTSSM(nn.Module):
    def __init__(self, num_nodes, in_dim, h_dim, out_dim, horizons=4, max_hop=3):
        super(DTSSM, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.horizons = horizons  # e.g., 4 horizons for 15, 30, 45, 60 mins
        self.max_hop = max_hop

        # Eq (10)
        self.gamma_StE = nn.Parameter(torch.tensor(0.33))
        self.gamma_FE = nn.Parameter(torch.tensor(0.33))
        self.gamma_SpE = nn.Parameter(torch.tensor(0.33))

        # Eq (26-29)
        self.W_T = nn.Linear(in_dim, h_dim)
        self.W_K = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(max_hop)])
        self.phi = nn.Linear(h_dim * max_hop, h_dim)

        # Eq (30)
        self.W_f = nn.Linear(h_dim, h_dim)
        self.w_tau = nn.Parameter(torch.tensor(1.0))

        # Eq (31-33)
        self.W_attn = nn.Linear(h_dim, h_dim, bias=False)
        self.W_a = nn.Linear(h_dim, out_dim)

        # Eq (34-35)
        self.W_g = nn.Linear(out_dim, h_dim)
        self.W_z = nn.Linear(h_dim, h_dim)
        self.W_z_p = nn.Linear(h_dim, h_dim, bias=False)
        self.W_r = nn.Linear(h_dim, h_dim)
        self.W_r_p = nn.Linear(h_dim, h_dim, bias=False)
        self.W_h = nn.Linear(h_dim, h_dim)
        self.W_h_p = nn.Linear(h_dim, h_dim, bias=False)

        # Eq (36)
        self.conv3 = nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(h_dim, h_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(h_dim, h_dim, kernel_size=7, padding=3)
        self.phi_c = nn.Linear(h_dim * 3, h_dim)

        # Eq (38-39)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim * 2, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 1)
            ) for _ in range(horizons)
        ])

    def rpca_rsvd_denoising(self, A, bnp, a_mask, rank=10, max_iter=5):
        # Eq (19-24)
        device = A.device
        mu = 1.25 / torch.norm(A, p=2)
        lam = 1.0 / np.sqrt(A.size(0))

        Y, L, S = torch.zeros_like(A), torch.zeros_like(A), torch.zeros_like(A)

        for _ in range(max_iter):
            #
            temp_L = A - S + (1.0 / mu) * Y
            n = temp_L.size(1)
            Omega = torch.randn(n, rank + 5).to(device)
            Q, _ = torch.qr(temp_L @ Omega)
            U, Sigma, V = torch.svd(Q.t() @ temp_L)
            U = Q @ U

            Sigma = torch.clamp(Sigma[:rank] - 1.0 / mu, min=0.0)
            L = U[:, :rank] @ torch.diag(Sigma) @ V[:, :rank].t()

            #
            temp_S = A - L + (1.0 / mu) * Y
            S = torch.sign(temp_S) * torch.clamp(torch.abs(temp_S) - lam / mu, min=0.0)

            #
            Y = Y + mu * (A - L - S)

        # Eq (24)
        eps_rank = 1e-3
        valid_mask = (L >= eps_rank) & (a_mask > 0) & (bnp > 0)
        A_clean = L * valid_mask.float()
        return A_clean

    def forward(self, X_seq, A_StE, A_FE, A_SpE, BNP, A_mask):
        batch, time_steps, num_nodes, _ = X_seq.shape

        # Eq (10-12)
        A_final = self.gamma_StE * A_StE + self.gamma_FE * A_FE + self.gamma_SpE * A_SpE
        A_e = A_final * BNP
        A_clean = self.rpca_rsvd_denoising(A_e, BNP, A_mask)

        h_t = torch.zeros(batch, num_nodes, self.h_dim).to(X_seq.device)
        h_hist = []

        for t in range(time_steps):
            X = X_seq[:, t, :, :]

            #
            H_e = self.W_T(X) # Eq 26
            S_scales = [torch.relu(self.W_K[k](H_e)) for k in range(self.max_hop)]
            S_fused = torch.relu(self.phi(torch.cat(S_scales, dim=-1))) # Eq 29

            # Eq 30
            X_mean = X.mean(dim=-1, keepdim=True)
            flow_diff = torch.abs(X_mean - X_mean.transpose(1, 2))
            tau = torch.nn.functional.softplus(self.w_tau) + 1e-6
            W_msg = A_clean.unsqueeze(0) * torch.tanh(flow_diff / tau)

            Z = torch.relu(self.W_f(torch.matmul(W_msg, S_fused)))

            #
            Q, K = self.W_attn(Z), self.W_attn(Z)
            E = torch.exp(torch.matmul(Q, K.transpose(-1, -2))) * (A_clean > 0).float().unsqueeze(0)

            degrees = (A_clean > 0).float().sum(dim=-1, keepdim=True).unsqueeze(0) + 1e-6
            alpha = (A_clean > 0).float().unsqueeze(0) / degrees

            for _ in range(3): # M_iter = 3
                e_bar = torch.sum(E * alpha, dim=-1, keepdim=True) + 1e-6
                alpha = (E * alpha) / e_bar
                alpha = alpha / (torch.sum(alpha, dim=-1, keepdim=True) + 1e-6)

            H_spat = torch.matmul(alpha, self.W_a(Z))

            # Eq (34-35)
            g_t = self.W_g(torch.matmul(A_clean, H_spat))
            z_t = torch.sigmoid(self.W_z(g_t) + self.W_z_p(h_t))
            r_t = torch.sigmoid(self.W_r(g_t) + self.W_r_p(h_t))
            h_tilde = torch.tanh(self.W_h(g_t) + self.W_h_p(r_t * h_t))
            h_t = (1 - z_t) * h_t + z_t * h_tilde

            h_hist.append(h_t.unsqueeze(1))

        # Eq (36)
        H_conv = torch.cat(h_hist, dim=1).view(batch * num_nodes, time_steps, self.h_dim).transpose(1, 2)
        c3 = torch.relu(self.conv3(H_conv)[:, :, -1])
        c5 = torch.relu(self.conv5(H_conv)[:, :, -1])
        c7 = torch.relu(self.conv7(H_conv)[:, :, -1])

        # Eq (37)
        C_t = torch.relu(self.phi_c(torch.cat([c3, c5, c7], dim=1))).view(batch, num_nodes, self.h_dim)
        u_t = torch.cat([h_t, C_t], dim=-1)

        # Eq (38-39)
        Y_hat_list = [mlp(u_t) for mlp in self.mlps]
        Y_hat = torch.cat(Y_hat_list, dim=-1)

        return Y_hat

#
if __name__ == "__main__":
    # model = DTSSM(num_nodes=460, in_dim=1, h_dim=64, out_dim=64, horizons=4)
    # y_preds = model(X_seq, A_StE, A_FE, A_SpE, BNP, A_mask)
    pass