import unittest

import numpy as np
import torch

from src.config import DataConfig, ModelConfig
from src.data import prepare_data
from src.evaluation import fit_alignment
from src.models import TwoFactorVaDE


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = TwoFactorVaDE(5, 5, 10, ModelConfig(hidden_dims=(16,), latent_dim=3))

    def test_normalization_and_marginals(self):
        q = self.model.category_probabilities(torch.randn(9, 5))
        self.assertEqual(q.shape, (9, 5, 10))
        torch.testing.assert_close(q.sum((1, 2)), torch.ones(9))
        self.assertTrue(torch.isfinite(q).all())

    def test_interaction_has_zero_row_and_column_means(self):
        with torch.no_grad():
            self.model.interaction.normal_()
        h = self.model.centered_interaction()
        torch.testing.assert_close(h.mean(0), torch.zeros_like(h.mean(0)), atol=1e-6, rtol=0)
        torch.testing.assert_close(h.mean(1), torch.zeros_like(h.mean(1)), atol=1e-6, rtol=0)

    def test_gaussian_kl_matches_torch_distribution(self):
        x = torch.randn(7, 5)
        generator = torch.Generator().manual_seed(4)
        _, components = self.model.loss(x, generator=generator)
        mean, logvar = self.model.encode(x)
        eps = torch.randn((self.model.config.train_mc_samples, *mean.shape),
                          generator=torch.Generator().manual_seed(4))
        z = mean[None] + (0.5 * logvar).exp()[None] * eps
        q = self.model.responsibilities(z).mean(0)
        posterior = torch.distributions.Normal(mean[:, None], (0.5 * logvar).exp()[:, None])
        prior = torch.distributions.Normal(self.model.component_means().flatten(0, 1),
                  (0.5 * self.model.component_logvar.clamp(-8, 6)).exp().flatten(0, 1))
        expected = (q * torch.distributions.kl_divergence(posterior, prior).sum(-1)).sum(-1).mean()
        torch.testing.assert_close(components["gaussian_kl"], expected)

    def test_loss_and_gradients_are_finite(self):
        loss, terms = self.model.loss(torch.randn(11, 5))
        self.assertTrue(all(torch.isfinite(term) for term in terms.values()))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_hungarian_mapping(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        clusters = np.array([2, 2, 0, 0, 1, 1])
        mapping = fit_alignment(truth, clusters, 3)
        np.testing.assert_array_equal(mapping[clusters], truth)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare_data(DataConfig())

    def test_all_selected_rows_are_used_once(self):
        data = self.data
        joined = np.concatenate(list(data.indices.values()))
        np.testing.assert_array_equal(np.sort(joined), np.arange(len(data.frame)))
        self.assertEqual(len(np.unique(joined)), len(joined))
        self.assertEqual(len(data.speakers), 10)
        self.assertEqual(data.frame.source_row.nunique(), len(data.frame))
        for part, x in data.x.items():
            self.assertTrue(np.isfinite(x).all())
            self.assertEqual(x.shape[1], 5)
            self.assertEqual(len(np.unique(data.y[part][:, 1])), 10)

    def test_preprocessing_is_fit_on_training_only(self):
        data = self.data
        raw = data.frame[data.metadata["features"]].to_numpy(float)
        raw[~np.isfinite(raw) | (raw <= 0)] = np.nan
        np.testing.assert_allclose(data.imputer.statistics_, np.nanmedian(raw[data.indices["train"]], axis=0))
        np.testing.assert_allclose(data.x["train"].mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(data.x["train"].std(0), 1, atol=1e-4)

    def test_selection_and_split_are_reproducible(self):
        repeated = prepare_data(DataConfig())
        self.assertEqual(self.data.speakers, repeated.speakers)
        for part in self.data.indices:
            np.testing.assert_array_equal(self.data.indices[part], repeated.indices[part])


if __name__ == "__main__":
    unittest.main()
