"""Two-factor VaDE and a supervised classifier sharing its encoder dimensions."""
import math

import torch
from torch import nn

from src.config import ModelConfig, SharedWModelConfig, NeuralModelConfig


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


class NeuralVaDE(TwoFactorVaDE):
    """The original statistical model with configurable hidden-layer activation.

    Inherit the exact same loss, prior, parameterization and inference methods.
    Width/depth are already supplied by ModelConfig.hidden_dims. Replacing ReLU
    modules changes only the encoder/decoder hidden nonlinearity, with identical
    parameter initialization for a fixed seed and architecture.
    """

    def __init__(self, input_dim, n_phonemes, n_speakers, config: NeuralModelConfig):
        super().__init__(input_dim, n_phonemes, n_speakers, config)
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}
        if config.activation not in activations:
            raise ValueError(f"Unknown activation: {config.activation}")
        if config.activation != "relu":
            def replace(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.ReLU):
                        setattr(module, name, activations[config.activation]())
                    else:
                        replace(child)
            replace(self.encoder)
            replace(self.decoder)


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


class SharedWGMVAE(TwoFactorVaDE):
    """Paper-inspired GMVAE with one shared auxiliary w and two categories.

    Generative model: p(w) p(c_phi) p(c_s) p(z|w,c_phi,c_s) p(x|z).
    Recognition: q(z|x) q(w|x) p(c_phi,c_s|z,w).

    Paper notation (Dilokthanakul et al., arXiv:1611.02648v2):
      paper y -> our x; paper continuous x -> our z;
      paper categorical z -> our (c_phi,c_s); paper w -> our w.

    Inherit the original encoder, decoder, static component parameters and prior
    weights without editing TwoFactorVaDE. A zero-initialized conditional head
    initially reproduces its static mixture. All components receive the SAME w
    sample for a token. w is not a separate variable per category or per speaker.
    """

    def __init__(self, input_dim: int, n_phonemes: int, n_speakers: int, config: SharedWModelConfig):
        if config.w_dim < 1 or config.objective not in ("paper", "structured"):
            raise ValueError("Use w_dim >= 1 and objective='paper' or 'structured'.")
        super().__init__(input_dim, n_phonemes, n_speakers, config)
        width = config.hidden_dims[-1] if config.hidden_dims else input_dim
        self.w_mean = nn.Linear(width, config.w_dim)
        self.w_logvar = nn.Linear(width, config.w_dim)
        # Start q(w|x) at its N(0,I) prior; it can subsequently depend on x.
        for head in (self.w_mean, self.w_logvar):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        layers = []
        width = config.w_dim
        for next_width in config.prior_hidden_dims:
            layers.extend([nn.Linear(width, next_width), nn.Tanh()])
            width = next_width
        self.prior_network = nn.Sequential(*layers)
        self.prior_head = nn.Linear(width, 2 * n_phonemes * n_speakers * config.latent_dim)
        nn.init.zeros_(self.prior_head.weight)
        nn.init.zeros_(self.prior_head.bias)

    def posterior_parameters(self, x):
        h = self.encoder(x)
        return (self.z_mean(h), self.z_logvar(h).clamp(-10, 6),
                self.w_mean(h), self.w_logvar(h).clamp(-10, 6))

    def sample_latents(self, x, n_samples, generator=None):
        z_mean, z_logvar, w_mean, w_logvar = self.posterior_parameters(x)
        eps_z = torch.randn((n_samples, *z_mean.shape), device=x.device, generator=generator)
        eps_w = torch.randn((n_samples, *w_mean.shape), device=x.device, generator=generator)
        z = z_mean[None] + (0.5 * z_logvar).exp()[None] * eps_z
        w = w_mean[None] + (0.5 * w_logvar).exp()[None] * eps_w
        return z, w, (z_mean, z_logvar, w_mean, w_logvar)

    def conditional_parameters(self, w):
        """Return means/logvariances/interactions with shape [..., K_phi,K_s,D].

        The network emits w-dependent residuals on the static mixture. Any mean
        grid decomposes into global + row + column + double-centered interaction;
        penalize the TOTAL interaction at the sampled w, not just the static h.
        Variances remain full diagonal component variances, as in the old model.
        """
        shape = (*w.shape[:-1], 2, self.n_phonemes, self.n_speakers, self.config.latent_dim)
        mean_delta, logvar_delta = self.prior_head(self.prior_network(w)).reshape(shape).unbind(-4)
        delta_interaction = (mean_delta - mean_delta.mean(-3, keepdim=True)
                             - mean_delta.mean(-2, keepdim=True)
                             + mean_delta.mean((-3, -2), keepdim=True))
        means = self.component_means() + mean_delta
        logvar = (self.component_logvar + logvar_delta).clamp(-8, 6)
        interaction = self.centered_interaction() + delta_interaction
        return means, logvar, interaction

    @staticmethod
    def component_log_densities(z, means, logvar):
        # Flatten the two category axes while preserving sample and batch axes.
        means, logvar = means.flatten(-3, -2), logvar.flatten(-3, -2)
        return -0.5 * (math.log(2 * math.pi) + logvar
                       + (z.unsqueeze(-2) - means).square() * (-logvar).exp()).sum(-1)

    def responsibilities(self, z, w):
        """Joint p(c_phi,c_s|z,w), normalized over K_phi*K_s components."""
        means, logvar, _ = self.conditional_parameters(w)
        return (self.component_log_densities(z, means, logvar) + self.log_weights()).softmax(-1)

    def loss(self, x, *, generator=None):
        """Paper Eq. 5 estimator or exact structured-posterior ELBO estimator.

        'paper': sample-wise responsibilities weight analytic Gaussian component
        KLs, following Eq. 5. This is NOT an unbiased estimator of the exact ELBO
        for q(z|x)q(w|x)p(c|z,w), because responsibilities depend on sampled z.
        It is retained explicitly for the requested paper-style comparison.

        'structured': keep log q(z|x)-log p(z|w,c) inside the sample expectation.
        Summing this term plus categorical KL is exactly log q(z|x)-log p(z|w).
        No category sampling, detached responsibilities, or label loss is used.
        """
        z, w, (z_mean, z_logvar, w_mean, w_logvar) = self.sample_latents(
            x, self.config.train_mc_samples, generator)
        means, logvar, interaction = self.conditional_parameters(w)
        log_component = self.component_log_densities(z, means, logvar)
        log_responsibility = (log_component + self.log_weights()).log_softmax(-1)
        responsibility = log_responsibility.exp()
        categorical_kl = (responsibility * (log_responsibility - self.log_weights())).sum(-1).mean()
        if self.config.objective == "paper":
            prior_mean, prior_logvar = means.flatten(-3, -2), logvar.flatten(-3, -2)
            component_kl = 0.5 * (
                prior_logvar - z_logvar[None, :, None] - 1
                + (z_logvar.exp()[None, :, None] + (z_mean[None, :, None] - prior_mean).square())
                * (-prior_logvar).exp()
            ).sum(-1)
            conditional_prior = (responsibility * component_kl).sum(-1).mean()
        else:
            log_qz = -0.5 * (math.log(2 * math.pi) + z_logvar[None]
                             + (z - z_mean[None]).square() * (-z_logvar[None]).exp()).sum(-1)
            conditional_prior = (responsibility * (log_qz[..., None] - log_component)).sum(-1).mean()
        w_kl = 0.5 * (w_mean.square() + w_logvar.exp() - 1 - w_logvar).sum(-1).mean()
        variance = self.config.observation_variance
        nll = 0.5 * ((self.decoder(z) - x[None]).square() / variance
                     + math.log(2 * math.pi * variance)).sum(-1).mean()
        interaction_mse = interaction.square().mean()
        loss = nll + conditional_prior + categorical_kl + w_kl + self.config.interaction_weight * interaction_mse
        return loss, {"loss": loss, "reconstruction_nll": nll, "conditional_prior": conditional_prior,
                      "categorical_kl": categorical_kl, "w_kl": w_kl, "interaction_mse": interaction_mse}

    @torch.no_grad()
    def category_probabilities(self, x, generator=None):
        z, w, _ = self.sample_latents(x, self.config.eval_mc_samples, generator)
        return self.responsibilities(z, w).mean(0).reshape(-1, self.n_phonemes, self.n_speakers)
