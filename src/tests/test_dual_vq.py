"""Scientific invariants: gradient routing, matched initialization and split leakage."""
import unittest
import numpy as np
import torch
from src.vq_models import DualVQVAE, VectorQuantizer
from src.vq_config import VQConfig
from src.vq_data import prepare_vq_data
from src.vq_training import initialize, state_digest
from src.training import seed_everything
from src.evaluation import fit_alignment


class VQTests(unittest.TestCase):
    def test_straight_through_routes_reconstruction_gradient_to_encoder(self):
        vq = VectorQuantizer(5, 8)
        h = torch.randn(12, 8, requires_grad=True)
        z, k, cb, commitment = vq(h)
        torch.testing.assert_close(z, vq.embedding(k))
        z.sum().backward()
        torch.testing.assert_close(h.grad, torch.ones_like(h))
        self.assertIsNone(vq.embedding.weight.grad)

    def test_codebook_and_commitment_detach_opposite_sides(self):
        vq = VectorQuantizer(5, 8)
        h = torch.randn(12, 8, requires_grad=True)
        _, _, cb, commitment = vq(h)
        cb.backward(retain_graph=True)
        self.assertIsNone(h.grad)
        self.assertIsNotNone(vq.embedding.weight.grad)
        vq.embedding.weight.grad = None
        commitment.backward()
        self.assertIsNotNone(h.grad)
        self.assertIsNone(vq.embedding.weight.grad)

    def test_condition_B_ignores_speaker_targets(self):
        model = DualVQVAE(5, VQConfig())
        x = torch.randn(12, 5)
        a = model.loss(x, torch.zeros(12, dtype=torch.long), 0)[0]
        b = model.loss(x, torch.ones(12, dtype=torch.long), 0)[0]
        torch.testing.assert_close(a, b)
        a.backward()
        self.assertIsNone(model.speaker_classifier.weight.grad)

    def test_supervision_reaches_only_speaker_branch_directly(self):
        model = DualVQVAE(5, VQConfig())
        x = torch.randn(12, 5)
        ce = torch.nn.functional.cross_entropy(model(x)['speaker_logits'], torch.arange(12)%10)
        ce.backward()
        self.assertIsNotNone(model.encoder_s[1].weight.grad)
        self.assertIsNone(model.encoder_phi[1].weight.grad)
        self.assertIsNone(model.vq_s.embedding.weight.grad)

    def test_initialization_reproducible(self):
        config = VQConfig(hidden_dims=(16,16), pretrain_epochs=1)
        x = np.random.default_rng(123).normal(size=(40,5)).astype('float32')
        hashes = []
        for _ in range(2):
            seed_everything(42,1)
            m = DualVQVAE(5,config)
            initialize(m,x,config,42)
            hashes.append(state_digest(m))
        self.assertEqual(*hashes)

    def test_mapping_is_frozen_even_when_test_semantics_reverse(self):
        mapping = fit_alignment(np.array([1,1,0,0]),np.array([0,0,1,1]),2)
        np.testing.assert_array_equal(mapping, [1,0])
        # A new test permutation must not retroactively improve test accuracy.
        self.assertEqual(float((mapping[np.array([0,1])] == np.array([0,1])).mean()),0)


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare_vq_data(VQConfig())

    def test_disjoint_tokens_and_speakers(self):
        d = self.data
        ix = np.concatenate(list(d.indices.values()))
        self.assertEqual(len(np.unique(ix)),len(d.frame))
        for part in ('train','val','test_seen'):
            self.assertEqual(set(d.frame.iloc[d.indices[part]].speaker_id),set(d.speakers))
        self.assertFalse(set(d.frame.iloc[d.indices['test_unseen']].speaker_id) & set(d.speakers))

    def test_preprocessing_uses_training_only(self):
        d = self.data
        raw = d.frame.iloc[d.indices['train']][list(VQConfig().data.features)].to_numpy(float)
        raw[~np.isfinite(raw)|(raw<=0)] = np.nan
        np.testing.assert_allclose(d.imputer.statistics_,np.nanmedian(raw,axis=0))
        np.testing.assert_allclose(d.scaler.mean_,d.imputer.transform(raw).mean(0))
        self.assertEqual(d.metadata['absent_training_phonemes'],[])


if __name__ == '__main__':
    unittest.main()
