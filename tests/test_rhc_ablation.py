"""Ablation routing, real removal of inactive branches, and legacy Full invariance."""
import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from context import SWAP_PROFILE_KEYS, auxiliary_loss, context_swap_loss, predict_next_state, state_scale
from dreamer import Dreamer
from test_context import config, inputs


def variant_config(name):
    cfg = config()
    history, response, exchange, wm = {
        'A': (False, False, False, 0.), 'B': (True, False, False, 1.),
        'C': (True, True, False, 0.), 'D': (True, True, True, 0.)}[name]
    cfg.model.context.update(dict(enabled=history, use_history_context=history,
        use_response_loss=response, use_context_exchange=exchange,
        injection='dali_shallow', actor_critic=False, aux_weight=1.,
        swap_weight=1. if exchange else 0., wm_grad_scale=wm, log_grad_norms=True))
    return cfg


def paired_inputs():
    data, _, act = inputs()
    data['ctx_episode_uid'] = torch.tensor([[0, 1], [0, 2]])[:, None].expand(2, 7, 2)
    for key in SWAP_PROFILE_KEYS: data[key] = torch.ones(2, 7, 1)
    obs = gym.spaces.Dict({k: gym.spaces.Box(-np.inf, np.inf, shape=tuple(v.shape[2:]), dtype=np.float32)
                          for k, v in data.items() if k not in ('action', 'reward')})
    return data, obs, act


def grad_norm(module):
    return sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None) ** .5


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_horizon_one_matches_single_transition_and_keeps_exchange(self):
        data, obs, act = paired_inputs()
        cfg = variant_config('D'); cfg.model.context.prediction_horizon = 1
        model = Dreamer(copy.deepcopy(cfg.model), obs, act)
        context = model.context_encoder(data['ctx_history'], data['ctx_mask'])
        context = torch.where(data['is_first'].reshape(2, 7, 1), 0., context)
        value, count = auxiliary_loss(model.context_dynamics, data, context, prediction_horizon=1)
        state = data['ctx_ego_state'].float()
        event = (data['is_terminal'] | data['is_last'] | data['ctx_collision']).reshape(2, 7)
        valid = ~event[:, :-1] & ~event[:, 1:] & ~data['is_first'].reshape(2, 7)[:, 1:]
        prediction = predict_next_state(model.context_dynamics, context[:, :-1], state[:, :-1],
                                        data['ctx_prev_action'][:, 1:].clamp(-1, 1))
        error = ((prediction-state[:, 1:]).abs()/state.new_tensor(state_scale(state.shape[-1]))).sum(-1)
        torch.testing.assert_close(value, error[valid].mean(), rtol=0, atol=0)
        self.assertEqual(count, valid.sum())
        gradients = torch.autograd.grad(value, list(model.context_encoder.parameters()), retain_graph=True)
        self.assertGreater(sum(float(g.square().sum()) for g in gradients), 0)
        self.assertTrue(all(g is None for g in torch.autograd.grad(value, list(model.rssm.parameters()), allow_unused=True)))
        before = {k: p.detach().clone() for k, p in model.context_encoder.named_parameters()}
        with patch('context.auxiliary_loss', wraps=auxiliary_loss) as response:
            _, metrics = model._cal_grad(data, model.rssm.initial(2))
        self.assertEqual(response.call_args.kwargs['prediction_horizon'], 1)
        self.assertIn('loss/context_swap', metrics)
        self.assertGreater(float(metrics['context/swap_pairs']), 0)
        self.assertEqual(model.context_wm_grad_scale, 0)
        self.assertGreater(grad_norm(model.context_encoder), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        model._scaler.step(model._optimizer); model._scaler.update()
        self.assertGreater(sum(float((p.detach()-before[k]).square().sum()) for k,p in model.context_encoder.named_parameters()), 0)
        restored = Dreamer(cfg.model, obs, act); restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(restored.context_encoder.position.shape, (5, 64))
        self.assertEqual(restored.context_encoder.mean.out_features, 8)
        action, _ = restored.act(data[:, 1], restored.get_initial_state(2), eval=True)
        self.assertTrue(torch.isfinite(action).all())

    def test_default_horizon_matches_frozen_five_step_loss_and_gradients(self):
        path = Path(__file__).resolve().parents[1]/'logdir/rhc_context_ablation_seed1_20261026/source/context.py'
        if not path.exists(): self.skipTest('frozen context source unavailable')
        spec = importlib.util.spec_from_file_location('frozen_context', path)
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
        data, obs, act = paired_inputs(); model = Dreamer(variant_config('D').model, obs, act)
        context = model.context_encoder(data['ctx_history'], data['ctx_mask'])
        parameters = list(model.context_encoder.parameters()) + list(model.context_dynamics.parameters())
        expected, count = legacy.auxiliary_loss(model.context_dynamics, data, context)
        old_grad = torch.autograd.grad(expected, parameters, retain_graph=True)
        for kwargs in ({}, {'prediction_horizon': 5}):
            actual, actual_count = auxiliary_loss(model.context_dynamics, data, context, **kwargs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(count, actual_count)
            for a,b in zip(old_grad, torch.autograd.grad(actual, parameters, retain_graph=True)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                auxiliary_loss(model.context_dynamics, data, context, prediction_horizon=invalid)

    def test_four_variants_update_and_roundtrip(self):
        data, obs, act = paired_inputs()
        for name in 'ABCD':
            with self.subTest(variant=name):
                torch.manual_seed(17)
                model = Dreamer(variant_config(name).model, obs, act)
                initial = model.rssm.initial(2)
                before = {k: p.detach().clone() for k, p in model._named_params.items() if k.startswith('context_encoder.')}
                _, metrics = model._cal_grad(data, initial)
                self.assertTrue(all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v)))
                self.assertEqual('loss/context_aux' in metrics, name in 'CD')
                self.assertEqual('loss/context_swap' in metrics, name == 'D')
                self.assertEqual(hasattr(model, 'context_dynamics'), name in 'CD')
                self.assertEqual(hasattr(model, 'context_encoder'), name != 'A')
                self.assertGreater(grad_norm(model.rssm), 0)
                if name != 'A': self.assertGreater(grad_norm(model.context_encoder), 0)
                model._scaler.step(model._optimizer); model._scaler.update()
                if before:
                    delta = sum(float((p.detach()-before[k]).square().sum()) for k, p in model._named_params.items() if k in before)
                    self.assertGreater(delta, 0)
                saved = io.BytesIO(); torch.save(model.state_dict(), saved); saved.seek(0)
                restored = Dreamer(variant_config(name).model, obs, act)
                restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
                torch.manual_seed(33); action, _ = restored.act(data[:, 1], restored.get_initial_state(2), eval=True)
                self.assertEqual(action.shape, (2, 2)); self.assertTrue(torch.isfinite(action).all())

    def test_history_only_removes_predictor_and_auxiliary_calls(self):
        data, obs, act = paired_inputs()
        with patch('context.ResidualDynamics', side_effect=AssertionError('inactive predictor instantiated')), \
             patch('context.auxiliary_loss', side_effect=AssertionError('inactive response loss called')), \
             patch('context.context_swap_loss', side_effect=AssertionError('inactive exchange called')), \
             patch('context.refresh_context', side_effect=AssertionError('inactive response rollout called')):
            model = Dreamer(variant_config('B').model, obs, act)
            _, metrics = model._cal_grad(data, model.rssm.initial(2))
            self.assertGreater(grad_norm(model.context_encoder), 0)
            self.assertGreater(float(metrics['context/wm_input_grad_norm']), 0)
            self.assertTrue(model.rssm.context_input_grad)
        cfg = variant_config('B'); cfg.model.context.wm_grad_scale = 0
        with self.assertRaisesRegex(ValueError, 'no training signal'):
            Dreamer(cfg.model, obs, act)

    def test_c_d_response_identical_and_auxiliaries_isolated(self):
        data, obs, act = paired_inputs()
        values = []
        for name in 'CD':
            torch.manual_seed(10)
            model = Dreamer(variant_config(name).model, obs, act)
            context = model.context_encoder(data['ctx_history'], data['ctx_mask'])
            context = torch.where(data['is_first'].reshape(2, 7, 1), 0., context)
            response, _ = auxiliary_loss(model.context_dynamics, data, context)
            exchange, _ = context_swap_loss(model.context_dynamics, data, context)
            values.append(response.detach())
            for objective in (response, exchange):
                gs = torch.autograd.grad(objective, list(model.rssm.parameters()), retain_graph=True, allow_unused=True)
                self.assertTrue(all(g is None for g in gs))
                gs = torch.autograd.grad(objective, list(model.context_encoder.parameters()), retain_graph=True)
                self.assertGreater(sum(float(g.square().sum()) for g in gs), 0)
            z, h, q = model.rssm.observe(model.encoder(data), data['action'], model.rssm.initial(2), data['is_first'], context)
            gs = torch.autograd.grad(q.square().mean(), list(model.context_encoder.parameters()), allow_unused=True)
            self.assertTrue(all(g is None for g in gs))
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)

        # The explicit switch must really disable Exchange even if an inherited
        # positive legacy weight remains in the configuration.
        cfg = variant_config('C'); cfg.model.context.swap_weight = 1.
        with patch('context.context_swap_loss', side_effect=AssertionError('disabled exchange called')):
            model = Dreamer(cfg.model, obs, act)
            _, metrics = model._cal_grad(data, model.rssm.initial(2))
            self.assertNotIn('loss/context_swap', metrics)

    def test_legacy_full_values_and_gradients_unchanged(self):
        path = Path(__file__).resolve().parents[1] / 'logdir/tintersection_roundabout_B_seed1_20261025/sources/source_1/dreamer.py'
        if not path.exists(): self.skipTest('frozen legacy implementation unavailable')
        spec = importlib.util.spec_from_file_location('legacy_rhc_dreamer', path)
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
        data, obs, act = paired_inputs()
        models = []
        for cls in (legacy.Dreamer, Dreamer):
            torch.manual_seed(23)
            models.append(cls(copy.deepcopy(variant_config('D').model), obs, act))
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
        outputs = []
        for model in models:
            torch.manual_seed(57)
            outputs.append(model._cal_grad(data, model.rssm.initial(2))[1])
        for key in outputs[0]: torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
        for key, param in models[0]._named_params.items():
            other = models[1]._named_params[key]
            if param.grad is None: self.assertIsNone(other.grad)
            else: torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
