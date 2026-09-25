"""Numerical checks of the two-category extension of the paper's shared w."""
import unittest

import torch

from src.config import SharedWModelConfig
from src.models import SharedWGMVAE, TwoFactorVaDE


class SharedWTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.config = SharedWModelConfig(hidden_dims=(16,), latent_dim=3, w_dim=2,
                                         prior_hidden_dims=(8,), train_mc_samples=3)
        self.model = SharedWGMVAE(5, 5, 10, self.config)
        self.x = torch.randn(7, 5)

    def generator(self):
        return torch.Generator().manual_seed(44)

    def test_zero_conditional_head_reproduces_static_prior(self):
        static = TwoFactorVaDE(5, 5, 10, self.config)
        static.load_state_dict({k: v for k, v in self.model.state_dict().items()
                                if k in static.state_dict()})
        z, w = torch.randn(4, 7, 3), torch.randn(4, 7, 2)
        means, logvar, _ = self.model.conditional_parameters(w)
        torch.testing.assert_close(means, static.component_means().expand_as(means))
        torch.testing.assert_close(logvar, static.component_logvar.expand_as(logvar))
        torch.testing.assert_close(self.model.responsibilities(z, w), static.responsibilities(z))

    def test_one_w_conditions_all_components(self):
        with torch.no_grad():
            self.model.prior_head.weight.normal_(std=0.1)
        means, logvar, interaction = self.model.conditional_parameters(torch.tensor([[0., 0.], [1., 0.]]))
        self.assertEqual(means.shape, (2, 5, 10, 3))
        self.assertFalse(torch.allclose(means[0], means[1]))
        self.assertFalse(torch.allclose(logvar[0], logvar[1]))
        self.assertGreater(interaction.square().mean().item(), 0)
        for dim in (-3, -2):
            torch.testing.assert_close(interaction.mean(dim), torch.zeros_like(interaction.mean(dim)), atol=1e-6, rtol=0)

    def test_joint_probabilities_and_marginals(self):
        q = self.model.category_probabilities(self.x, self.generator())
        self.assertEqual(q.shape, (7, 5, 10))
        torch.testing.assert_close(q.sum((1, 2)), torch.ones(7))

    def test_paper_conditional_prior_matches_distribution_kl(self):
        _, terms = self.model.loss(self.x, generator=self.generator())
        z, w, (zm, zv, wm, wv) = self.model.sample_latents(self.x, 3, self.generator())
        means, logvar, _ = self.model.conditional_parameters(w)
        q = torch.distributions.Normal(zm[None, :, None], (0.5 * zv).exp()[None, :, None])
        p = torch.distributions.Normal(means.flatten(-3, -2), (0.5 * logvar).exp().flatten(-3, -2))
        kl = torch.distributions.kl_divergence(q, p).sum(-1)
        expected = (self.model.responsibilities(z, w) * kl).sum(-1).mean()
        torch.testing.assert_close(terms["conditional_prior"], expected)

    def test_exact_structured_objective_equals_collapsed_mixture(self):
        self.config.objective = "structured"
        with torch.no_grad():
            self.model.prior_head.weight.normal_(std=0.05)
        _, terms = self.model.loss(self.x, generator=self.generator())
        z, w, (zm, zv, _, _) = self.model.sample_latents(self.x, 3, self.generator())
        means, logvar, _ = self.model.conditional_parameters(w)
        q = torch.distributions.Normal(zm, (0.5 * zv).exp())
        log_p = torch.logsumexp(self.model.component_log_densities(z, means, logvar)
                               + self.model.log_weights(), dim=-1)
        expected = (q.log_prob(z).sum(-1) - log_p).mean()
        torch.testing.assert_close(terms["conditional_prior"] + terms["categorical_kl"], expected)

    def test_w_kl_matches_standard_normal_kl(self):
        with torch.no_grad():
            self.model.w_mean.bias.fill_(0.4)
            self.model.w_logvar.bias.fill_(-0.6)
        _, terms = self.model.loss(self.x, generator=self.generator())
        _, _, mean, logvar = self.model.posterior_parameters(self.x)
        q = torch.distributions.Normal(mean, (0.5 * logvar).exp())
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
        torch.testing.assert_close(terms["w_kl"], torch.distributions.kl_divergence(q, p).sum(-1).mean())

    def test_both_objectives_have_finite_gradients(self):
        with torch.no_grad():
            self.model.prior_head.weight.normal_(std=0.05)
        for objective in ("paper", "structured"):
            with self.subTest(objective=objective):
                self.config.objective = objective
                self.model.zero_grad(set_to_none=True)
                loss, terms = self.model.loss(self.x, generator=self.generator())
                self.assertTrue(all(torch.isfinite(v) for v in terms.values()))
                loss.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(self.model.w_mean.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
