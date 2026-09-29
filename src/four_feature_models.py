"""Four-dimensional VQ and algebraically equivalent memory-efficient VaDE."""
import math
import torch
from torch import nn
from src.models import mlp, TwoFactorVaDE
from src.vq_models import VectorQuantizer
from src.sensitivity_models import QuantizedSpeakerVQVAE


class FourFeatureVQVAE(QuantizedSpeakerVQVAE):
    def __init__(self,config,k_phi,k_s):
        nn.Module.__init__(self)
        d,w = config.embedding_dim,config.hidden_dims
        self.encoder_phi = nn.Sequential(mlp(4,w),nn.Linear(w[-1],d))
        self.encoder_s = nn.Sequential(mlp(4,w),nn.Linear(w[-1],d))
        # Fixed-size initial tables preserve identical neural initialization
        # across capacity, following the preceding sensitivity implementation.
        self.vq_phi = VectorQuantizer(5,d,config.commitment)
        self.vq_s = VectorQuantizer(10,d,config.commitment)
        self.decoder = nn.Sequential(mlp(2*d,tuple(reversed(w))),nn.Linear(w[0],4))
        self.speaker_classifier = nn.Linear(d,10)
        self.vq_phi = VectorQuantizer(k_phi,d,config.commitment)
        self.vq_s = VectorQuantizer(k_s,d,config.commitment)


class EfficientTwoFactorVaDE(TwoFactorVaDE):
    """Same parameters, MC draws and ELBO; expand quadratic forms using GEMM.

    Avoid [MC,batch,K*K,latent_dim] broadcast temporaries at K=64.
    This is a computational reassociation, not a new probabilistic model.
    """
    def responsibilities(self,z):
        mean = self.component_means().flatten(0,1)
        lv = self.component_logvar.clamp(-8,6).flatten(0,1)
        precision = (-lv).exp()
        quadratic = z.square() @ precision.T - 2*z @ (mean*precision).T
        quadratic = quadratic + (mean.square()*precision+lv+math.log(2*math.pi)).sum(-1)
        return (-0.5*quadratic+self.log_weights()).softmax(-1)

    def loss(self,x,*,generator=None):
        mean,lv = self.encode(x)
        eps = torch.randn((self.config.train_mc_samples,*mean.shape),device=x.device,generator=generator)
        z = mean[None]+(0.5*lv).exp()[None]*eps
        q = self.responsibilities(z).mean(0)
        variance = self.config.observation_variance
        nll = 0.5*((self.decoder(z)-x[None]).square()/variance+math.log(2*math.pi*variance)).sum(-1).mean()
        pm = self.component_means().flatten(0,1)
        pl = self.component_logvar.clamp(-8,6).flatten(0,1)
        precision = (-pl).exp()
        kl = 0.5*((lv.exp()+mean.square())@precision.T-2*mean@(pm*precision).T
                  +(pl+pm.square()*precision).sum(-1)[None]-(lv+1).sum(-1)[:,None])
        gaussian = (q*kl).sum(-1).mean()
        categorical = (q*(q.clamp_min(1e-12).log()-self.log_weights())).sum(-1).mean()
        interaction = self.centered_interaction().square().mean()
        total = nll+gaussian+categorical+self.config.interaction_weight*interaction
        return total,{'loss':total,'reconstruction_nll':nll,'gaussian_kl':gaussian,
                      'categorical_kl':categorical,'interaction_mse':interaction}
