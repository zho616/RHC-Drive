"""Direct conditions: physical inputs, padding, train/inference and checkpoints."""
import copy
import unittest
from unittest.mock import patch

import torch
import gymnasium as gym
import numpy as np

from context import direct_condition, ORACLE_KEYS, ORACLE_SCALE
from dreamer import Dreamer
from test_rhc_ablation import paired_inputs, variant_config, grad_norm


class DirectConditionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_exact_values_padding_and_no_parameter_ids(self):
        data, _, _ = paired_inputs()
        values = (1400., 35., 1.2, 4.5, 1.8, 2.5, 1.6, .35, 800., 100.)
        for key, value in zip(ORACLE_KEYS, values):
            data[key] = torch.full((2, 7, 1), value)
        oracle = direct_condition(data, 'oracle')
        torch.testing.assert_close(oracle[0, 0], torch.tensor(values) / torch.tensor(ORACLE_SCALE))
        data['log_chassis_model_id'].fill_(999)
        torch.testing.assert_close(direct_condition(data, 'oracle'), oracle)
        data['ctx_mask'][:, :, :2] = False
        raw = direct_condition(data, 'raw_history')
        self.assertEqual(raw.shape, (2, 7, 35))
        data['ctx_history'][~data['ctx_mask']] = float('nan')
        torch.testing.assert_close(direct_condition(data, 'raw_history'), raw)
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue((raw[..., :12] == 0).all())
        torch.testing.assert_close(raw[..., -5:], data['ctx_mask'].float())
        swapped = data.clone()
        swapped['ctx_history'] = data['ctx_history'].flip(-2)
        swapped['ctx_mask'] = data['ctx_mask'].flip(-1)
        self.assertFalse(torch.equal(raw, direct_condition(swapped, 'raw_history')))

    def test_both_conditions_train_and_roundtrip_without_auxiliary_branches(self):
        data, obs, act = paired_inputs()
        for key, value in zip(ORACLE_KEYS, ORACLE_SCALE):
            data[key] = torch.full((2, 7, 1), value)
            obs.spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        for source, width in [('oracle', 10), ('raw_history', 35)]:
            with self.subTest(source=source):
                cfg = variant_config('B')
                cfg.model.context.update(dict(source=source, wm_grad_scale=0., aux_weight=0.))
                with patch('context.HistoryEncoder', side_effect=AssertionError('encoder used')), \
                     patch('context.ResidualDynamics', side_effect=AssertionError('predictor used')):
                    model = Dreamer(copy.deepcopy(cfg.model), obs, act)
                    self.assertFalse(hasattr(model, 'context_encoder'))
                    self.assertFalse(hasattr(model, 'context_dynamics'))
                    self.assertEqual(model.context_dim, width)
                    _, metrics = model._cal_grad(data, model.rssm.initial(2))
                    self.assertFalse(any(k.startswith('loss/context_') for k in metrics))
                    self.assertGreater(grad_norm(model.rssm), 0)
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    model._scaler.step(model._optimizer)
                    model._scaler.update()
                    restored = Dreamer(copy.deepcopy(cfg.model), obs, act)
                    restored.load_state_dict(model.state_dict(), strict=True)
                    torch.manual_seed(7)
                    a, _ = model.act(data[:, 2], model.get_initial_state(2), eval=True)
                    torch.manual_seed(7)
                    b, _ = restored.act(data[:, 2], restored.get_initial_state(2), eval=True)
                    torch.testing.assert_close(a, b)
                    self.assertTrue(torch.isfinite(b).all())
                    # The prior/imagination path is condition-free in Shallow.
                    z, h = restored.rssm.initial(2)
                    embed = restored.encoder(data[:, 2])
                    reset = torch.zeros(2, 1, dtype=torch.bool)
                    condition = restored._context_condition(data[:, 2])
                    _, _, logits = restored.rssm.obs_step(z, h, a, embed, reset, condition)
                    _, _, other = restored.rssm.obs_step(z, h, a, embed, reset, condition + 2.)
                    self.assertFalse(torch.equal(logits, other))


if __name__ == '__main__':
    unittest.main()
