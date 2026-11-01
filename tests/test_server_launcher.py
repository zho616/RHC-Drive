"""Check cross-server experiment defaults without starting simulators or GPUs."""
import argparse
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
from scripts.train_server import build_config, METHODS


class ServerLauncherTests(unittest.TestCase):
    def arguments(self, **overrides):
        values = dict(task='nuplan', method='r2dreamer_context', output=Path('/tmp/dry_run'),
                      steps=None, seed=0, device='cuda:0', eval_every=None,
                      eval_episodes=None, data_root=Path('/tmp/converted_dataset'), smoke=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_shared_environment_and_context_contract(self):
        root = Path(__file__).resolve().parents[1]
        for task in ('nuplan', 'roundabout', 'tintersection'):
            environments = []
            for method in METHODS:
                with self.subTest(task=task, method=method):
                    c = build_config(root, self.arguments(task=task, method=method))
                    plain = OmegaConf.to_container(c, resolve=True)
                    self.assertEqual(c.env.steps, 1000000 if task == 'nuplan' else 500000)
                    self.assertEqual(c.trainer.eval_every, 5000 if task == 'nuplan' else 10000)
                    self.assertEqual(c.env.eval_episode_num, 100 if task == 'nuplan' else 20)
                    self.assertEqual(c.model.context.enabled, method.endswith('_context'))
                    self.assertEqual(c.model.context.injection, 'dali_shallow')
                    self.assertFalse(c.model.context.actor_critic)
                    env = plain['env']
                    env.pop('context_enabled')
                    environments.append(env)
            self.assertTrue(all(env == environments[0] for env in environments))

    def test_dataset_relocation_and_independent_seed(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            c = build_config(root, self.arguments(seed=7, data_root=Path(directory)))
            OmegaConf.resolve(c)
            self.assertEqual(c.env.seed, 7)
            self.assertEqual(c.env.seed + c.env.start_seed, 0)
            self.assertEqual(c.env.seed + c.env.eval_start_seed, 10000)
            self.assertEqual(c.env.scenario_train_dir, str(Path(directory) / 'train'))
            self.assertEqual(c.env.scenario_eval_dir, str(Path(directory) / 'val'))
            self.assertEqual(c.env.scenario_test_dir, str(Path(directory) / 'test'))


if __name__ == '__main__':
    unittest.main()
