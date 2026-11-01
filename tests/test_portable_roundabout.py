"""Portable configs must match actual local experiments and preserve split boundaries."""
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf

from scripts.train_roundabout_ablation import build_config, METHODS, prepare, verify

ROOT = Path(__file__).resolve().parents[1]


def flat(value, prefix=''):
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result.update(flat(item, prefix + key + '.'))
        else:
            result[prefix + key] = item
    return result


class PortableRoundaboutTests(unittest.TestCase):
    def test_routes_and_seed_offsets(self):
        for seed in (0, 1, 2, 7):
            for method in METHODS:
                c = build_config(ROOT, method, seed, Path('/tmp/portable_example'))
                self.assertEqual(c.env.seed + c.env.start_seed, 0)
                self.assertEqual(c.env.seed + c.env.eval_start_seed, 10000)
                self.assertEqual(c.env.steps, 500000)
                self.assertEqual(c.env.env_num, 8)
                self.assertEqual(c.batch_size, 16)
                self.assertEqual(c.batch_length, 64)
                self.assertEqual(c.env.train_ratio, 512)
                self.assertEqual(c.env.map_block, 'O')
                self.assertEqual(c.model.context.injection, 'dali_shallow')
                self.assertEqual(c.model.context.enabled, method != 'baseline')
                self.assertEqual(c.model.context.wm_grad_scale, 1. if method == 'history_only' else 0.)
                self.assertEqual(c.model.context.use_context_exchange, method in ('full_h1', 'full_h5'))
                self.assertEqual(c.model.context.prediction_horizon, 1 if method == 'full_h1' else 5)
                self.assertEqual(c.env.get('oracle_parameters', False), method == 'oracle')

    def test_exact_match_to_frozen_local_production_configs(self):
        folder = ROOT / 'logdir/roundabout_five_ablations_3seeds_20261030_retry1/configs'
        if not folder.exists():
            self.skipTest('Local frozen experiments are not distributed with the source repository')
        for seed in (0, 1, 2):
            for number, method in enumerate(METHODS[:5], 1):
                suffix = '10' if method == 'oracle' else ''
                reference = OmegaConf.load(folder / f'seed{seed}_{number}_{method}{suffix}.yaml')
                actual = build_config(ROOT, method, seed, Path('/tmp/portable_example'))
                a = flat(OmegaConf.to_container(reference, resolve=True))
                b = flat(OmegaConf.to_container(actual, resolve=True))
                differences = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
                self.assertEqual(differences, {'logdir', 'hydra.run.dir'}, (seed, method, differences))

    def test_frozen_queue_is_independent_of_original_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            plan = prepare(ROOT, out, ['oracle', 'raw_history'], [2], 0, True)
            verify(out, plan)
            self.assertEqual(len(plan['jobs']), 2)
            self.assertTrue((out / 'source/vehicle_parameters.py').is_file())
            for job in plan['jobs']:
                self.assertEqual(job['budget'], 96)
                self.assertEqual(len(job['evaluations']), 3)
                self.assertTrue(str(job['train'][2]).startswith(str(out)))
            (out / 'source/context.py').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'Frozen files changed'):
                verify(out, plan)


if __name__ == '__main__':
    unittest.main()
