"""The compact oracle reports actual geometry without changing vehicle dynamics."""
import copy
from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from envs.metadrive import MetaDrive
from vehicle_parameters import ORACLE_KEYS, read_vehicle_parameters

ROOT = Path(__file__).resolve().parents[1]


class VehicleParameterTests(unittest.TestCase):
    def test_native_geometry_readback_all_chassis_and_observation_invariance(self):
        cfg = OmegaConf.load(ROOT / 'configs/metadrive_roundabout_500k.yaml')
        cfg.seed = 0
        cfg.env.start_seed = 0
        cfg.env.eval_start_seed = 10000
        cfg.env.chassis.models = ['s', 'default', 'm', 'l', 'xl']
        base = MetaDrive(copy.deepcopy(cfg.env))
        try:
            standard = base.reset(seed=3)
            standard_next, _, _, _ = base.step(np.array([.1, .2], np.float32))
        finally:
            base.close()
        cfg.env.oracle_parameters = True
        env = MetaDrive(cfg.env)
        try:
            observed = env.reset(seed=3)
            np.testing.assert_array_equal(observed['state'], standard['state'])
            next_obs, _, _, _ = env.step(np.array([.1, .2], np.float32))
            np.testing.assert_allclose(next_obs['state'], standard_next['state'], atol=1e-6)
            np.testing.assert_array_equal(next_obs['ctx_history'], standard_next['ctx_history'])
            seen = set()
            for seed in range(25):
                obs = env.reset(seed=seed)
                vehicle = env._env.vehicle
                seen.add(env.chassis_parameters['model'])
                self.assertTrue(env.observation_space.contains(obs))
                self.assertTrue(set(ORACLE_KEYS).issubset(obs))
                self.assertEqual(obs['state'].shape, (259,))
                self.assertAlmostEqual(float(obs['log_chassis_wheelbase']),
                                       vehicle.FRONT_WHEELBASE + vehicle.REAR_WHEELBASE, places=5)
                self.assertAlmostEqual(float(obs['log_chassis_track']),
                                       2 * vehicle.LATERAL_TIRE_TO_CENTER, places=5)
                self.assertAlmostEqual(float(obs['log_chassis_wheel_radius']), vehicle.TIRE_RADIUS, places=6)
                for key, value in read_vehicle_parameters(vehicle).items():
                    self.assertEqual(obs[key], value)
                stepped, _, _, _ = env.step(np.array([.1, .2], np.float32))
                for key in ORACLE_KEYS:
                    self.assertEqual(stepped[key], obs[key])
            self.assertEqual(seen, {'s', 'default', 'm', 'l', 'xl'})
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
