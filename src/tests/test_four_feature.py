import unittest
import numpy as np
import torch
from src.config import ModelConfig
from src.models import TwoFactorVaDE
from src.four_feature_config import FourFeatureConfig
from src.four_feature_models import FourFeatureVQVAE, EfficientTwoFactorVaDE
from src.four_feature_data import prepare_four_feature_data


class FourFeatureTests(unittest.TestCase):
    def test_gmvae_objective_and_gradients(self):
        torch.manual_seed(42)
        cfg = ModelConfig(hidden_dims=(12,12),latent_dim=4)
        a = TwoFactorVaDE(4,16,16,cfg).double()
        b = EfficientTwoFactorVaDE(4,16,16,cfg).double()
        b.load_state_dict(a.state_dict())
        x = torch.randn(7,4,dtype=torch.float64)
        la,ca = a.loss(x,generator=torch.Generator().manual_seed(8))
        lb,cb = b.loss(x,generator=torch.Generator().manual_seed(8))
        for key in ca:
            torch.testing.assert_close(ca[key],cb[key],rtol=1e-9,atol=1e-10)
        la.backward(); lb.backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-10)

    def test_four_dimensional_vq_and_capacity_matching(self):
        cfg = FourFeatureConfig()
        torch.manual_seed(42); a = FourFeatureVQVAE(cfg,16,16)
        torch.manual_seed(42); b = FourFeatureVQVAE(cfg,64,64)
        for name,p in a.state_dict().items():
            if not name.startswith('vq_'):
                torch.testing.assert_close(p,b.state_dict()[name],rtol=0,atol=0)
        x = torch.randn(9,4)
        out = a(x)
        self.assertEqual(out['reconstruction'].shape,x.shape)
        torch.testing.assert_close(out['speaker_logits'],a.speaker_classifier(out['z_s']))

    def test_k64_float32_inference_equivalence(self):
        torch.manual_seed(43)
        cfg = ModelConfig(hidden_dims=(12,12))
        a = TwoFactorVaDE(4,64,64,cfg)
        b = EfficientTwoFactorVaDE(4,64,64,cfg)
        b.load_state_dict(a.state_dict())
        x = torch.randn(5,4)
        p = a.category_probabilities(x,generator=torch.Generator().manual_seed(9))
        q = b.category_probabilities(x,generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(p,q,rtol=2e-5,atol=1e-8)

    def test_manifest_and_preprocessing(self):
        d = prepare_four_feature_data(FourFeatureConfig())
        self.assertEqual(d.metadata['split_counts'],{'train':36863,'val':4608,'test_seen':4612,'test_unseen':13686})
        self.assertNotIn('f0_median_hz',d.frame)
        for p,x in d.x.items():
            self.assertEqual(x.shape,(d.metadata['split_counts'][p],4))
            self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(d.x['train'].mean(0),0,atol=1e-6)


if __name__=='__main__':
    unittest.main()
