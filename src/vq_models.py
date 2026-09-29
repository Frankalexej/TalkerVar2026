"""Nonlinear dual-codebook VQ-VAE; labels enter only the auxiliary speaker loss."""
import torch
from torch import nn
from src.models import mlp


class VectorQuantizer(nn.Module):
    def __init__(self, size, dim, beta=0.25):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)
        self.beta = beta
        nn.init.uniform_(self.embedding.weight, -1/size, 1/size)

    def forward(self, h):
        e = self.embedding.weight
        distances = h.square().sum(1, keepdim=True) + e.square().sum(1)[None] - 2*h@e.T
        k = distances.argmin(1)
        quantized = self.embedding(k)
        codebook = (h.detach() - quantized).square().mean()
        commitment = (h - quantized.detach()).square().mean()
        straight_through = h + (quantized - h).detach()
        return straight_through, k, codebook, commitment


class DualVQVAE(nn.Module):
    def __init__(self, n_phonemes, config):
        super().__init__()
        dim, widths = config.embedding_dim, config.hidden_dims
        self.encoder_phi = nn.Sequential(mlp(5, widths), nn.Linear(widths[-1], dim))
        self.encoder_s = nn.Sequential(mlp(5, widths), nn.Linear(widths[-1], dim))
        self.vq_phi = VectorQuantizer(n_phonemes, dim, config.commitment)
        self.vq_s = VectorQuantizer(10, dim, config.commitment)
        self.decoder = nn.Sequential(mlp(2*dim, tuple(reversed(widths))), nn.Linear(widths[0], 5))
        # Present in both conditions for identical initialization/RNG. At lambda=0
        # it is not optimized; its accuracy is intentionally not reported.
        self.speaker_classifier = nn.Linear(dim, 10)

    def forward(self, x):
        hp, hs = self.encoder_phi(x), self.encoder_s(x)
        zp, kp, cp, bp = self.vq_phi(hp)
        zs, ks, cs, bs = self.vq_s(hs)
        return {'reconstruction': self.decoder(torch.cat([zp, zs], 1)),
                'h_phi': hp, 'h_s': hs, 'z_phi': zp, 'z_s': zs, 'k_phi': kp, 'k_s': ks,
                'speaker_logits': self.speaker_classifier(hs),
                'codebook_phi': cp, 'commitment_phi': bp, 'codebook_s': cs, 'commitment_s': bs}

    def loss(self, x, speakers, speaker_weight):
        out = self(x)
        recon = (x - out['reconstruction']).square().mean()
        ce = nn.functional.cross_entropy(out['speaker_logits'], speakers)
        vp = out['codebook_phi'] + self.vq_phi.beta*out['commitment_phi']
        vs = out['codebook_s'] + self.vq_s.beta*out['commitment_s']
        total = recon + vp + vs
        if speaker_weight:
            total = total + speaker_weight*ce
        components = {k: out[k] for k in ('codebook_phi', 'commitment_phi', 'codebook_s', 'commitment_s')}
        components.update(loss=total, reconstruction=recon, vq_phi=vp, vq_s=vs, speaker_ce=ce)
        return total, components, out
