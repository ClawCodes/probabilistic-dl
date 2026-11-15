from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalVAE(nn.Module):
    """Gaussian conditional VAE with a diagonal-normal encoder and decoder."""

    def __init__(self, x_dim: int, cond_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.x_dim, self.cond_dim, self.latent_dim = x_dim, cond_dim, latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + cond_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, x_dim),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat((x, c), dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        return self.decoder(torch.cat((z, c), dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int | None = None):
        n = len(c) if n is None else n
        z = torch.randn(n, self.latent_dim, device=c.device)
        return self.decode(z, c)


def kl_per_dimension(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x,c) || N(0, I)) for each sample and latent dimension, shape (batch, latent_dim)."""
    return -0.5 * (1 + logvar - mu.square() - logvar.exp())


def vae_loss(reconstruction, target, mu, logvar, beta: float = 0.015):
    """MSE reconstruction (mean over data dims) + beta * KL.

    The KL is summed over latent dimensions (and averaged over the batch), so
    beta has the same meaning for every latent size. Averaging over latent
    dimensions would silently scale the KL weight by 1/latent_dim.
    """
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction="mean")
    kl = kl_per_dimension(mu, logvar).sum(dim=1).mean()
    return reconstruction_loss + beta * kl, reconstruction_loss, kl
