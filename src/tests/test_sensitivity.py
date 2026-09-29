import unittest
import numpy as np
import torch
from src.sensitivity_config import SensitivityConfig
from src.sensitivity_models import QuantizedSpeakerVQVAE
from src.sensitivity_metrics import fit_many_to_one
from src.raw_clusterability import fit_speaker_normalizer,speaker_normalize
from src.training import seed_everything
from src.vq_training import initialize


class SensitivityTests(unittest.TestCase):
    def test_classifier_reads_quantized_vector(self):
        m = QuantizedSpeakerVQVAE(SensitivityConfig(),64,64)
        out = m(torch.randn(20,5))
        torch.testing.assert_close(out['speaker_logits'],m.speaker_classifier(out['z_s']))
        torch.testing.assert_close(out['z_s'],m.vq_s.embedding(out['k_s']))

    def test_post_quantization_ce_gradient_route(self):
        m = QuantizedSpeakerVQVAE(SensitivityConfig(),64,64)
        out = m(torch.randn(20,5))
        torch.nn.functional.cross_entropy(out['speaker_logits'],torch.arange(20)%10).backward()
        self.assertIsNotNone(m.encoder_s[1].weight.grad)
        self.assertIsNotNone(m.speaker_classifier.weight.grad)
        self.assertIsNone(m.encoder_phi[1].weight.grad)
        self.assertIsNone(m.vq_s.embedding.weight.grad)

    def test_capacity_preserves_pretrained_neural_parameters(self):
        cfg = SensitivityConfig(hidden_dims=(16,16),pretrain_epochs=1)
        x = np.random.default_rng(1).normal(size=(100,5)).astype('float32')
        states = []
        for size in ((5,10),(64,64)):
            seed_everything(42,1)
            model = QuantizedSpeakerVQVAE(cfg,*size)
            initialize(model,x,cfg,42)
            states.append({k:v for k,v in model.state_dict().items() if not k.startswith('vq_')})
        for name in states[0]:
            torch.testing.assert_close(states[0][name],states[1][name],rtol=0,atol=0)

    def test_many_to_one_and_empty_code_fallback(self):
        mapping,empty = fit_many_to_one(np.array([0,0,1,2]),np.array([1,1,1,0]),4,2)
        np.testing.assert_array_equal(mapping,[1,1,0,1])
        np.testing.assert_array_equal(empty,[3])

    def test_normalization_uses_frozen_training_stats(self):
        x = np.array([[0.,1.],[2.,3.],[20.,5.],[22.,7.]])
        s = np.array([0,0,1,1])
        stats = fit_speaker_normalizer(x,s)
        z = speaker_normalize(x,s,stats)
        np.testing.assert_allclose(z,[[-1,-1],[1,1],[-1,-1],[1,1]])
        np.testing.assert_allclose(speaker_normalize(np.array([[5.,7.]]),np.array([0]),stats),[[4.,5.]])
        with self.assertRaises(ValueError):
            speaker_normalize(np.array([[5.,7.]]),np.array([2]),stats)


if __name__ == '__main__':
    unittest.main()
