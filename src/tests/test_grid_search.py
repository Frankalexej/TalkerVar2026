from dataclasses import asdict
import unittest

import torch

from src.config import ModelConfig, NeuralModelConfig
from src.grid_search import build_trial_config, selection_score
from src.models import NeuralVaDE, TwoFactorVaDE
from src.search_config import GridSearchConfig, grid_candidates


class GridTests(unittest.TestCase):
    def test_neural_class_inherits_statistical_methods_unchanged(self):
        for name in ("loss", "encode", "component_means", "centered_interaction",
                     "log_weights", "responsibilities", "category_probabilities"):
            self.assertIs(getattr(NeuralVaDE, name), getattr(TwoFactorVaDE, name))

    def test_relu_matches_original_state_and_loss(self):
        torch.manual_seed(12)
        original = TwoFactorVaDE(5, 5, 10, ModelConfig(hidden_dims=(16,)))
        torch.manual_seed(12)
        neural = NeuralVaDE(5, 5, 10, NeuralModelConfig(hidden_dims=(16,), activation="relu"))
        for key in original.state_dict():
            torch.testing.assert_close(original.state_dict()[key], neural.state_dict()[key], atol=0, rtol=0)
        x = torch.randn(7, 5)
        a, _ = original.loss(x, generator=torch.Generator().manual_seed(3))
        b, _ = neural.loss(x, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_tanh_changes_only_nonlinearities(self):
        config = NeuralModelConfig(hidden_dims=(16, 16), activation="tanh")
        model = NeuralVaDE(5, 5, 10, config)
        self.assertEqual(sum(isinstance(m, torch.nn.Tanh) for m in model.modules()), 4)
        self.assertEqual(sum(isinstance(m, torch.nn.ReLU) for m in model.modules()), 0)
        loss, _ = model.loss(torch.randn(9, 5))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_grid_has_24_settings_plus_original_control(self):
        candidates = grid_candidates(GridSearchConfig())
        self.assertEqual(len(candidates), 25)
        self.assertEqual(candidates[0]["candidate_id"], "c000")
        self.assertEqual(candidates[0]["batch_size"], 512)

    def test_statistical_grid_change_rejected(self):
        search = GridSearchConfig()
        search.grid["latent_dim"] = [2, 8]
        with self.assertRaises(ValueError):
            grid_candidates(search)

    def test_trial_preserves_statistical_hyperparameters(self):
        search = GridSearchConfig()
        for candidate in grid_candidates(search):
            trial = build_trial_config(search, candidate, 43, "screen")
            for key, value in asdict(search.base.model).items():
                if key != "hidden_dims":
                    self.assertEqual(getattr(trial.model, key), value)
            self.assertEqual(trial.data, search.base.data)
            self.assertEqual(trial.training.pretrain_epochs, search.base.training.pretrain_epochs)

    def test_selection_score_uses_both_accuracies(self):
        self.assertAlmostEqual(selection_score({"phoneme": {"accuracy": 0.8},
                                               "speaker": {"accuracy": 0.2}}), 0.4)


if __name__ == "__main__":
    unittest.main()
