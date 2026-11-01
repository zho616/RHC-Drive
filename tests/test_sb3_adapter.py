"""Protect Gymnasium timeout semantics and state-only observation boundary."""
import unittest
from types import SimpleNamespace
import numpy as np
import gymnasium as gym
from scripts.train_metadrive_sb3 import StateEnv

class AdapterTests(unittest.TestCase):
    def test_state_only_and_terminal_flags(self):
        for terminal in (True,False):
            env=StateEnv.__new__(StateEnv)
            obs={'state':np.zeros(259,np.float32),'is_terminal':terminal,'ctx_ego_state':np.ones(4)}
            class Native:
                _env=SimpleNamespace(vehicle=SimpleNamespace(speed_km_h=10.),current_seed=0)
                chassis_parameters={'model':'default'}
                def reset(self):return obs
                def step(self,action):return obs,1.,True,{'arrive_dest':terminal,'route_completion':.5}
            env.native=Native()
            state,info=env.reset(seed=12345)
            self.assertEqual(state.shape,(259,))
            state,reward,terminated,truncated,info=env.step(np.zeros(2))
            self.assertEqual(terminated,terminal);self.assertEqual(truncated,not terminal)
            self.assertEqual(info['episode_result']['steps'],1)
            self.assertEqual(info['episode_result']['timeout'],not terminal)
            self.assertEqual(info['episode_result']['reward'],1.)

if __name__=='__main__':unittest.main()
