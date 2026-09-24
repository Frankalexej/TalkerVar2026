"""Two-factor VaDE and a supervised classifier sharing its encoder dimensions."""
import math

import torch
from torch import nn

from src.config import ModelConfig


def mlp(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    return nn.Sequential(*layers)


class TwoFactorVaDE(nn.Module):
    """(phoneme category, speaker category) -> continuous z -> acoustic x.

    Categories are inferred through Gaussian mixture responsibilities. There are
    no label classifier heads and no label loss. The category names describe the
    intended interpretation, which unsupervised training need not recover.
    """

    def __init__(self, input_dim: int, n_phonemes: int, n_speakers: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_phonemes, self.n_speakers = n_phonemes, n_speakers
        self.encoder = mlp(input_dim, config.hidden_dims)
        width = config.hidden_dims[-1] if config.hidden_dims else input_dim
        self.z_mean = nn.Linear(width, config.latent_dim)
        self.z_logvar = nn.Linear(width, config.latent_dim)
        self.decoder = nn.Sequential(
            mlp(config.latent_dim, tuple(reversed(config.hidden_dims))),
            nn.Linear(config.hidden_dims[0] if config.hidden_dims else config.latent_dim, input_dim),
        )
        k, s, d = n_phonemes, n_speakers, config.latent_dim
        self.global_mean = nn.Parameter(torch.zeros(d))
        self.phoneme_effect = nn.Parameter(torch.randn(k, d) * 0.2)
        self.speaker_effect = nn.Parameter(torch.randn(s, d) * 0.2)
        self.interaction = nn.Parameter(torch.zeros(k, s, d))
        self.component_logvar = nn.Parameter(torch.zeros(k, s, d))
        self.phoneme_logits = nn.Parameter(torch.zeros(k), requires_grad=config.learn_mixture_weights)
        self.speaker_logits = nn.Parameter(torch.zeros(s), requires_grad=config.learn_mixture_weights)

    def encode(self, x):
        h = self.encoder(x)
        return self.z_mean(h), self.z_logvar(h).clamp(-10, 6)

    def centered_interaction(self):
        h = self.interaction
        return h - h.mean(0, keepdim=True) - h.mean(1, keepdim=True) + h.mean((0, 1), keepdim=True)

    def component_means(self):
        a = self.phoneme_effect - self.phoneme_effect.mean(0)
        b = self.speaker_effect - self.speaker_effect.mean(0)
        return self.global_mean + a[:, None] + b[None, :] + self.centered_interaction()

    def log_weights(self):
        return (self.phoneme_logits.log_softmax(0)[:, None]
                + self.speaker_logits.log_softmax(0)[None, :]).flatten()

    def responsibilities(self, z):
        """p(c_phoneme, c_speaker | z); accepts [..., latent_dim]."""
        means = self.component_means().flatten(0, 1)
        logvar = self.component_logvar.clamp(-8, 6).flatten(0, 1)
        log_density = -0.5 * (math.log(2 * math.pi) + logvar
                              + (z.unsqueeze(-2) - means).square() * (-logvar).exp()).sum(-1)
        return (log_density + self.log_weights()).softmax(-1)

    def loss(self, x, *, generator=None):
        """Negative mean-field VaDE ELBO plus interaction mean-square penalty.

        q(c|x) is approximated by averaging p(c|z) over encoder samples.
        Gaussian KL terms are analytic. Reconstruction uses Monte Carlo.
        Gradients flow through responsibilities (no detach or hard assignments).
        This is the recovered toy's VaDE-style objective, not the alternative
        collapsed structured-posterior KL(q(z|x) || p(z)).
        """
        mean, logvar = self.encode(x)
        eps = torch.randn((self.config.train_mc_samples, *mean.shape),
                          device=x.device, generator=generator)
        z = mean[None] + (0.5 * logvar).exp()[None] * eps
        q = self.responsibilities(z).mean(0)
        reconstruction = self.decoder(z)
        variance = self.config.observation_variance
        nll = 0.5 * (((reconstruction - x[None]).square() / variance)
                     + math.log(2 * math.pi * variance)).sum(-1).mean()
        prior_mean = self.component_means().flatten(0, 1)
        prior_logvar = self.component_logvar.clamp(-8, 6).flatten(0, 1)
        kl_per_component = 0.5 * (
            prior_logvar[None] - logvar[:, None] - 1
            + (logvar.exp()[:, None] + (mean[:, None] - prior_mean[None]).square())
            * (-prior_logvar[None]).exp()
        ).sum(-1)
        gaussian_kl = (q * kl_per_component).sum(-1).mean()
        categorical_kl = (q * (q.clamp_min(1e-12).log() - self.log_weights())).sum(-1).mean()
        interaction_mse = self.centered_interaction().square().mean()
        loss = nll + gaussian_kl + categorical_kl + self.config.interaction_weight * interaction_mse
        return loss, {"loss": loss, "reconstruction_nll": nll, "gaussian_kl": gaussian_kl,
                      "categorical_kl": categorical_kl, "interaction_mse": interaction_mse}

    @torch.no_grad()
    def category_probabilities(self, x, generator=None):
        mean, logvar = self.encode(x)
        eps = torch.randn((self.config.eval_mc_samples, *mean.shape),
                          device=x.device, generator=generator)
        z = mean[None] + (0.5 * logvar).exp()[None] * eps
        return self.responsibilities(z).mean(0).reshape(-1, self.n_phonemes, self.n_speakers)


class VowelClassifier(nn.Module):
    """Matched encoder and latent-width bottleneck, followed by two label heads."""

    def __init__(self, input_dim: int, n_phonemes: int, n_speakers: int, config: ModelConfig):
        super().__init__()
        self.encoder = mlp(input_dim, config.hidden_dims)
        width = config.hidden_dims[-1] if config.hidden_dims else input_dim
        self.bottleneck = nn.Linear(width, config.latent_dim)
        self.phoneme_head = nn.Linear(config.latent_dim, n_phonemes)
        self.speaker_head = nn.Linear(config.latent_dim, n_speakers)

    def forward(self, x):
        z = self.bottleneck(self.encoder(x))
        return self.phoneme_head(z), self.speaker_head(z)

    def loss(self, x, y):
        phoneme, speaker = self(x)
        lp = nn.functional.cross_entropy(phoneme, y[:, 0])
        ls = nn.functional.cross_entropy(speaker, y[:, 1])
        return lp + ls, {"loss": lp + ls, "phoneme_ce": lp, "speaker_ce": ls}
