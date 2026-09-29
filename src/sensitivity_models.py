"""Speaker CE on the straight-through quantized speaker vector."""
import torch
from src.vq_models import DualVQVAE, VectorQuantizer


class QuantizedSpeakerVQVAE(DualVQVAE):
    def __init__(self, config, k_phi, k_s):
        # Create all MLPs identically first: neural initialization is identical
        # across capacities for a given seed, despite different embedding tables.
        super().__init__(5, config)
        if k_phi != 5:
            self.vq_phi = VectorQuantizer(k_phi, config.embedding_dim, config.commitment)
        if k_s != 10:
            self.vq_s = VectorQuantizer(k_s, config.embedding_dim, config.commitment)

    def forward(self, x):
        hp, hs = self.encoder_phi(x), self.encoder_s(x)
        zp, kp, cp, bp = self.vq_phi(hp)
        zs, ks, cs, bs = self.vq_s(hs)
        return {'reconstruction': self.decoder(torch.cat([zp,zs],1)),
                'h_phi':hp, 'h_s':hs, 'z_phi':zp, 'z_s':zs, 'k_phi':kp, 'k_s':ks,
                'speaker_logits':self.speaker_classifier(zs),
                'codebook_phi':cp, 'commitment_phi':bp, 'codebook_s':cs, 'commitment_s':bs}
