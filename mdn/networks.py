"""
Mixture density networks and their deterministic regression counterparts.

MDN1D / RegressionNet1D: scalar input/output, for the 2D synthetic toy task.
MDN / RegressionNet: general vector input/output, for the higher-dimensional
dataset. MDN(K=1) reduces to the single-Gaussian regression baseline.
Each pair shares a body architecture so the comparison isn't confounded by
capacity differences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_body(in_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )


class MDN1D(nn.Module):
    """Scalar input -> scalar multimodal output mixture density network."""

    def __init__(self, in_dim=1, hidden=64, K=3):
        super().__init__()
        self.body = _make_body(in_dim, hidden)
        self.alpha_head = nn.Linear(hidden, K)
        self.mu_head = nn.Linear(hidden, K)
        self.sigma_head = nn.Linear(hidden, K)

    def forward(self, x):
        z = self.body(x)
        alpha = F.softmax(self.alpha_head(z), dim=-1)   # (B, K), sums to 1
        mu = self.mu_head(z)                              # (B, K)
        sigma = F.softplus(self.sigma_head(z)) + 1e-4    # (B, K), > 0
        return alpha, mu, sigma


def mdn_nll_1d(alpha, mu, sigma, y):
    y = y.expand_as(mu)                                  # (B,1) -> (B,K)
    log_component = (-0.5 * ((y - mu) / sigma) ** 2
                      - torch.log(sigma)
                      - 0.5 * torch.log(torch.tensor(2 * torch.pi)))
    log_mix = torch.logsumexp(torch.log(alpha + 1e-12) + log_component, dim=-1)
    return -log_mix.mean()


class RegressionNet1D(nn.Module):
    """Deterministic regression baseline with the same body as MDN1D."""

    def __init__(self, in_dim=1, hidden=64, out_dim=1):
        super().__init__()
        self.body = _make_body(in_dim, hidden)
        self.out_head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        z = self.body(x)
        return self.out_head(z)


def _demo_1d():
    # Same multimodal data both models will be trained/compared on: an
    # inverted noisy sine, where one x maps to up to 3 plausible y's.
    torch.manual_seed(0)
    N = 2000
    y_true = torch.rand(N, 1) * 2 * torch.pi - torch.pi
    x = y_true + 0.3 * torch.sin(2 * y_true) + 0.05 * torch.randn(N, 1)

    mdn = MDN1D(in_dim=1, hidden=64, K=3)
    reg = RegressionNet1D(in_dim=1, hidden=64, out_dim=1)
    mdn_opt = torch.optim.Adam(mdn.parameters(), lr=1e-3)
    reg_opt = torch.optim.Adam(reg.parameters(), lr=1e-3)

    for epoch in range(2000):
        alpha, mu, sigma = mdn(x)
        mdn_loss = mdn_nll_1d(alpha, mu, sigma, y_true)
        mdn_opt.zero_grad()
        mdn_loss.backward()
        mdn_opt.step()

        y_pred = reg(x)
        reg_loss = F.mse_loss(y_pred, y_true)
        reg_opt.zero_grad()
        reg_loss.backward()
        reg_opt.step()

        if epoch % 200 == 0:
            print(f"epoch {epoch:4d}  mdn nll {mdn_loss.item():.4f}  reg mse {reg_loss.item():.4f}")


# ---------------------------------------------------------------------------
# General (higher-dimensional) case: vector input, vector output.
# MDN(K=1) is the single-Gaussian regression baseline; RegressionNet is the
# deterministic (non-probabilistic) baseline. Both share _make_body_relu so
# capacity is matched between all three models being compared.
# ---------------------------------------------------------------------------

def _make_body_relu(in_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )


class MDN(nn.Module):
    """General mixture density network: vector input -> vector multimodal
    output. K=1 reduces to a single-Gaussian regression model."""

    def __init__(self, in_dim, out_dim, hidden=128, K=5, rank=4, eps=1e-3):
        super().__init__()
        self.K, self.d, self.m, self.eps = K, out_dim, rank, eps
        self.body = _make_body_relu(in_dim, hidden)
        self.alpha_head = nn.Linear(hidden, K)
        self.mu_head = nn.Linear(hidden, K * out_dim)
        self.r_head = nn.Linear(hidden, K * out_dim)          # diagonal precision term
        self.L_head = nn.Linear(hidden, K * out_dim * rank)   # low-rank correction

    def forward(self, x):
        B = x.shape[0]
        z = self.body(x)
        alpha = F.softmax(self.alpha_head(z), dim=-1)                  # (B, K)
        mu = self.mu_head(z).view(B, self.K, self.d)                    # (B, K, d)
        r = self.r_head(z).view(B, self.K, self.d)                      # (B, K, d)
        L = self.L_head(z).view(B, self.K, self.d, self.m)              # (B, K, d, m)

        diag_part = F.relu(r) + self.eps                                 # (B, K, d), strictly > 0
        Lambda = torch.diag_embed(diag_part) + L @ L.transpose(-1, -2)   # (B, K, d, d), guaranteed PD
        return alpha, mu, Lambda


def mdn_nll(alpha, mu, Lambda, y):
    B, K, d = mu.shape
    y = y.unsqueeze(1).expand(-1, K, -1)                     # (B, K, d)
    dist = torch.distributions.MultivariateNormal(loc=mu, precision_matrix=Lambda)
    log_component = dist.log_prob(y)                          # (B, K)
    log_mix = torch.logsumexp(torch.log(alpha + 1e-12) + log_component, dim=-1)
    return -log_mix.mean()


class RegressionNet(nn.Module):
    """General deterministic regression baseline, same body as MDN."""

    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.body = _make_body_relu(in_dim, hidden)
        self.out_head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        z = self.body(x)
        return self.out_head(z)


def _demo_general():
    # Synthetic higher-dimensional stand-in: 2 distinct linear modes per x,
    # so the target is genuinely multimodal in a multi-dimensional output space.
    torch.manual_seed(0)
    N, in_dim, out_dim = 4000, 10, 5
    x = torch.randn(N, in_dim)
    mode = (torch.rand(N) < 0.5).float().unsqueeze(-1)
    W1, W2 = torch.randn(in_dim, out_dim), torch.randn(in_dim, out_dim)
    y = mode * (x @ W1) + (1 - mode) * (x @ W2) + 0.1 * torch.randn(N, out_dim)

    mdn = MDN(in_dim=in_dim, out_dim=out_dim, hidden=128, K=4, rank=2)
    reg = RegressionNet(in_dim=in_dim, out_dim=out_dim, hidden=128)
    mdn_opt = torch.optim.Adam(mdn.parameters(), lr=1e-3)
    reg_opt = torch.optim.Adam(reg.parameters(), lr=1e-3)

    for epoch in range(1000):
        alpha, mu, Lambda = mdn(x)
        mdn_loss = mdn_nll(alpha, mu, Lambda, y)
        mdn_opt.zero_grad()
        mdn_loss.backward()
        mdn_opt.step()

        y_pred = reg(x)
        reg_loss = F.mse_loss(y_pred, y)
        reg_opt.zero_grad()
        reg_loss.backward()
        reg_opt.step()

        if epoch % 100 == 0:
            print(f"epoch {epoch:4d}  mdn nll {mdn_loss.item():.4f}  reg mse {reg_loss.item():.4f}")


if __name__ == "__main__":
    _demo_1d()
    _demo_general()
