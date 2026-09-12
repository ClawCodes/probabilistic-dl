"""
1D mixture density network and its deterministic regression counterpart,
sharing the same body architecture so the two are comparable on equal footing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_body(in_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
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


def _demo():
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


if __name__ == "__main__":
    _demo()
