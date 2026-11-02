
# 2. MNL Parameter Calibration for Functional Edges

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# [INPUT]
INPUT_DISTANCE_MATRIX = ""
INPUT_ATTRACTIVENESS = ""
INPUT_EMPIRICAL_PROB = ""

class FunctionalGraphCalibrator:
    def __init__(self, d_ij, A_jt, P_empirical):
        self.d_ij = torch.tensor(d_ij, dtype=torch.float32)
        self.A_jt = torch.tensor(A_jt, dtype=torch.float32)
        self.P_emp = torch.tensor(P_empirical, dtype=torch.float32)

        # Eq (8)
        self.alpha_raw = torch.nn.Parameter(torch.tensor(1.0))
        self.mu_raw = torch.nn.Parameter(torch.tensor(5.0))

    def compute_choice_prob(self):
        # Eq (6)
        alpha = F.softplus(self.alpha_raw)
        mu = F.softplus(self.mu_raw)
        phi_d = 1.0 / (1.0 + torch.exp(-alpha * (self.d_ij - mu)))

        # Eq (9)
        psi_A = torch.log(self.A_jt + 1.0)

        # Eq (4)-(5)
        exp_V = phi_d * psi_A
        P_model = exp_V / (torch.sum(exp_V, dim=-1, keepdim=True) + 1e-8)
        return P_model

    def calibrate(self, max_iter=1000, lr=0.01):
        # Eq (7)
        optimizer = optim.Adam([self.alpha_raw, self.mu_raw], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            P_model = self.compute_choice_prob()

            # KL divergence calculation
            kl_loss = torch.sum(self.P_emp * torch.log(self.P_emp / (P_model + 1e-8)))
            kl_loss.backward()
            optimizer.step()

        return F.softplus(self.alpha_raw).item(), F.softplus(self.mu_raw).item()

# Execution Block
if __name__ == "__main__":
    # d_ij = np.load(INPUT_DISTANCE_MATRIX)
    # A_jt = np.load(INPUT_ATTRACTIVENESS)
    # P_emp = np.load(INPUT_EMPIRICAL_PROB)
    # calibrator = FunctionalGraphCalibrator(d_ij, A_jt, P_emp)
    # opt_alpha, opt_mu = calibrator.calibrate()
    pass