"""Single-agent MetaDrive LiDAR observations in the Dreamer environment protocol."""

import gymnasium as gym
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.traffic_manager import PGTrafficManager


class RandomRoundaboutSpawnManager(BaseManager):
    """Choose a legal entrance and a reachable, different exit before agent reset."""

    PRIORITY = 5  # Map=0; agents/traffic=10.

    def reset(self):
        config = self.engine.global_config
        rng = np.random.default_rng(int(config["spawn_random_seed"]))
        current_map = self.engine.map_manager.current_map
        road_network = current_map.road_network
        first_socket = current_map.blocks[0].get_socket_list()[0]
        # First socket faces into the junction; other sockets face out of it.
        entries = [first_socket.positive_road]
        exits = [first_socket.negative_road]
        for socket in current_map.blocks[-1].get_socket_list():
            entries.append(socket.negative_road)
            exits.append(socket.positive_road)
        entry = int(rng.integers(len(entries)))
        exit_index = int(rng.choice([i for i in range(len(exits)) if i != entry]))
        lanes = entries[entry].get_lanes(road_network)
        lane = lanes[int(rng.integers(len(lanes)))]
        lower = float(config["spawn_longitude_min"])
        upper = min(float(config["spawn_longitude_max"]), lane.length - 5.0)
        if upper < lower:
            raise ValueError(f"Spawn range does not fit entrance lane of length {lane.length}")
        longitude = float(rng.uniform(lower, upper))
        lateral_limit = min(float(config["spawn_lateral_max"]), lane.width / 2 - 1.0)
        lateral = float(rng.uniform(-lateral_limit, lateral_limit))
        agent_config = config["agent_configs"][DEFAULT_AGENT]
        agent_config.update(dict(spawn_lane_index=lane.index, spawn_longitude=longitude,
                                 spawn_lateral=lateral, destination=exits[exit_index].end_node))
        self.last_spawn = dict(entry=entry, exit=exit_index, lane_index=lane.index,
                               longitude=longitude, lateral=lateral)


class MultiEntranceTrafficManager(PGTrafficManager):
    def get_vehicle_num(self):
        return len(self._traffic_vehicles)

    def _create_trigger_vehicles(self, map, traffic_density):
        super()._create_trigger_vehicles(map, traffic_density)
        # Native trigger waits for the original entrance. Other entrances would
        # otherwise encounter parked, inactive vehicles. Activate the ring traffic
        # from episode start while retaining native density and spawn locations.
        for block in self.block_triggered_vehicles:
            self._traffic_vehicles.extend(self.get_objects(block.vehicles).values())
        self.block_triggered_vehicles.clear()
        # Native traffic placement assumes the original fixed ego entrance.
        # A random ego position can overlap a background car at time zero.
        # Clear a local safety margin before the first physics step.
        agents = list(self.engine.agent_manager.episode_created_agents.values())
        overlapping = [vehicle for vehicle in self._traffic_vehicles
                       if any(np.linalg.norm(vehicle.position - agent.position) < 8.0 for agent in agents)]
        if overlapping:
            self.clear_objects([vehicle.id for vehicle in overlapping])
            self._traffic_vehicles = [vehicle for vehicle in self._traffic_vehicles if vehicle not in overlapping]


class RoundaboutEnv(MetaDriveEnv):
    """Additive driving reward; use any current same-direction reference lane."""

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(lateral_penalty=0.2, reward_mode="additive",
                           random_roundabout_spawn=False, spawn_random_seed=0,
                           spawn_longitude_min=5.0, spawn_longitude_max=25.0,
                           spawn_lateral_max=0.5, chassis_enabled=False,
                           chassis_spec=dict(models=["default"], mass_scales=[1.0], steering_scales=[1.0],
                                             friction_scales=[1.0]),
                           chassis_random_seed=0))
        config["vehicle_config"].update(dict(max_speed_km_h=80.0))
        return config

    def setup_engine(self):
        super().setup_engine()
        if self.config["random_roundabout_spawn"]:
            self.engine.register_manager("roundabout_spawn_manager", RandomRoundaboutSpawnManager())
            self.engine.update_manager("traffic_manager", MultiEntranceTrafficManager())

        if self.config["chassis_enabled"]:
            from envs.chassis import ChassisManager
            self.engine.register_manager("chassis_manager", ChassisManager())

    def reward_function(self, vehicle_id):
        vehicle = self.agents[vehicle_id]
        refs = vehicle.navigation.current_ref_lanes
        lane = vehicle.lane if vehicle.lane in refs else refs[0]
        long_last, _ = lane.local_coordinates(vehicle.last_position)
        long_now, lateral = lane.local_coordinates(vehicle.position)
        if self.config["reward_mode"] == "metadrive":
            # Delegate reward and event priority to the installed MetaDrive.
            reward, info = super().reward_function(vehicle_id)
            sign = 1 if vehicle.lane in refs or not vehicle.navigation.current_road.is_negative_road() else -1
            progress = self.config["driving_reward"] * (long_now - long_last) * sign
            speed = self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * sign
            event = (self._is_arrive_destination(vehicle) or self._is_out_of_road(vehicle)
                     or vehicle.crash_vehicle or vehicle.crash_object or vehicle.crash_sidewalk)
            info.update(
                reward_progress=0.0 if event else float(progress),
                reward_speed=0.0 if event else float(speed),
                reward_lateral=0.0,
                reward_terminal=float(reward) if event else 0.0,
                lateral_offset=float(lateral),
            )
            return reward, info
        if self.config["reward_mode"] == "paper":
            # Signed longitudinal progress, normalized speed, additive terminal.
            progress = self.config["driving_reward"] * (long_now - long_last)
            speed = self.config["speed_reward"] * vehicle.speed_km_h / vehicle.max_speed_km_h
            lateral_reward = 0.0
        else:
            heading = lane.heading_theta_at(long_now)
            correct_direction = float(np.dot(vehicle.heading, [np.cos(heading), np.sin(heading)]) > 0)
            progress = self.config["driving_reward"] * (long_now - long_last) * correct_direction
            speed = self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * correct_direction
            lateral_reward = -self.config["lateral_penalty"] * abs(lateral) / lane.width
        collision = any(
            getattr(vehicle, key, False)
            for key in ("crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk")
        )
        terminal = 0.0
        if self._is_arrive_destination(vehicle):
            terminal = self.config["success_reward"]
        elif collision:
            terminal = -self.config["crash_vehicle_penalty"]
        elif self._is_out_of_road(vehicle):
            terminal = -self.config["out_of_road_penalty"]
        reward = float(progress + speed + lateral_reward + terminal)
        return reward, {
            "step_reward": reward,
            "reward_progress": float(progress),
            "reward_speed": float(speed),
            "reward_lateral": float(lateral_reward),
            "reward_terminal": float(terminal),
            "lateral_offset": float(lateral),
            "route_completion": vehicle.navigation.route_completion,
        }

    def done_function(self, vehicle_id):
        done, info = super().done_function(vehicle_id)
        vehicle = self.agents[vehicle_id]
        info.update(
            off_lane=not vehicle.on_lane,
            yellow_line=vehicle.on_yellow_continuous_line,
            white_line=vehicle.on_white_continuous_line,
        )
        return done, info


class MetaDrive(gym.Env):
    def __init__(self, config, env_id=0, evaluation=False, *, use_render=False):
        if config.task not in ("metadrive_roundabout", "metadrive_tintersection"):
            raise ValueError(f"Unsupported MetaDrive task: {config.task}")
        self._action_repeat = int(config.action_repeat)
        if self._action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        context_enabled = bool(getattr(config, "context_enabled", False))
        if context_enabled and self._action_repeat != 1:
            raise ValueError("Context history currently requires action_repeat=1")
        self._start_seed = int(config.seed) + int(config.eval_start_seed if evaluation else config.start_seed)
        self._num_scenarios = int(config.eval_num_scenarios if evaluation else config.num_scenarios)
        if self._num_scenarios < 1:
            raise ValueError("num_scenarios must be positive")
        self._rng = np.random.default_rng(int(config.seed) + env_id + (1000000 if evaluation else 0))
        self._evaluation = evaluation
        self._env_id = env_id
        self._context_episode_counter = 0
        self._eval_stride = min(int(getattr(config, "eval_env_num", config.eval_episode_num)),
                                int(config.eval_episode_num))
        self._eval_next_index = env_id
        chassis = getattr(config, "chassis", None)
        self._chassis_enabled = bool(chassis and chassis.enabled)
        self._oracle_parameters = bool(getattr(config, 'oracle_parameters', False))
        self._oracle_values = None
        if self._oracle_parameters and not self._chassis_enabled:
            raise ValueError('Vehicle-parameter conditioning requires chassis randomization')
        chassis_spec = None
        if self._chassis_enabled:
            from envs.chassis import validate_spec
            prefix = "eval_" if evaluation else ""
            chassis_spec = {key: list(getattr(chassis, prefix + key)) for key in (
                "models", "mass_scales", "steering_scales")}
            chassis_spec["friction_scales"] = list(getattr(chassis, prefix + "friction_scales", [1.0]))
            validate_spec(chassis_spec)
            # Dedicated stream avoids coupling chassis to maps/spawn positions.
            self._chassis_seed = int(config.seed)
            self._chassis_rng = np.random.default_rng(np.random.SeedSequence([
                self._chassis_seed, int(env_id), 72109]))
        reward_mode = str(getattr(config.reward, "mode", "additive"))
        if reward_mode not in ("additive", "metadrive", "paper"):
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        map_block = str(getattr(config, "map_block", "O"))
        if map_block not in ("O", "T"):
            raise ValueError(f"Unsupported MetaDrive map block: {map_block}")
        self._env = RoundaboutEnv(
            dict(
                # Explicit generation config allows lane_num to be customized.
                # Native block ID, following the initial straight road.
                map_config=dict(type="block_sequence", config=map_block, lane_num=int(config.lane_num)),
                start_seed=self._start_seed,
                num_scenarios=self._num_scenarios,
                store_map=False,
                use_render=bool(use_render),
                image_observation=False,
                traffic_density=float(config.traffic_density),
                traffic_mode=config.traffic_mode,
                random_traffic=False,
                horizon=int(config.time_limit),
                truncate_as_terminate=False,
                decision_repeat=int(config.decision_repeat),
                out_of_road_done=bool(config.termination.out_of_road_done),
                on_continuous_line_done=bool(config.termination.on_continuous_line_done),
                on_broken_line_done=bool(config.termination.on_broken_line_done),
                out_of_route_done=bool(config.termination.out_of_route_done),
                crash_vehicle_done=True,
                crash_object_done=True,
                random_roundabout_spawn=bool(getattr(getattr(config, "spawn", None), "randomize", False)),
                random_spawn_lane_index=not bool(getattr(getattr(config, "spawn", None), "randomize", False)),
                spawn_longitude_min=float(getattr(getattr(config, "spawn", None), "longitude_min", 5.0)),
                spawn_longitude_max=float(getattr(getattr(config, "spawn", None), "longitude_max", 25.0)),
                spawn_lateral_max=float(getattr(getattr(config, "spawn", None), "lateral_max", 0.5)),
                driving_reward=float(config.reward.progress),
                reward_mode=reward_mode,
                speed_reward=float(config.reward.speed),
                lateral_penalty=float(config.reward.lateral),
                success_reward=float(config.reward.success),
                crash_vehicle_penalty=float(config.reward.crash),
                crash_object_penalty=float(config.reward.crash),
                out_of_road_penalty=float(config.reward.out_of_road),
                vehicle_config=dict(
                    max_speed_km_h=float(config.reward.max_speed_km_h),
                    lidar=dict(
                        num_lasers=int(config.lidar.num_lasers),
                        distance=float(config.lidar.distance),
                        num_others=int(config.lidar.num_others),
                    ),
                ),
                chassis_enabled=self._chassis_enabled,
                chassis_spec=chassis_spec or dict(models=["default"], mass_scales=[1.0], steering_scales=[1.0],
                                                 friction_scales=[1.0]),
                log_level=40,
            )
        )
        self.action_space = self._env.action_space
        spaces = {"state": self._env.observation_space}
        for key in ("is_first", "is_last", "is_terminal"):
            spaces[key] = gym.spaces.Box(0, 1, shape=(), dtype=np.bool_)
        for key in ("log_success", "log_crash", "log_out_of_road", "log_route_completion", "log_cost"):
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32)
        for key in ("off_lane", "yellow_line", "white_line", "crash_sidewalk"):
            spaces[f"log_{key}"] = gym.spaces.Box(0, 1, shape=(), dtype=np.float32)
        for key in ("progress", "speed", "lateral", "terminal"):
            spaces[f"log_reward_{key}"] = gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32)
        if self._chassis_enabled:
            from envs.chassis import LOG_KEYS
            for key in LOG_KEYS:
                spaces[f"log_chassis_{key}"] = gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32)
        if self._oracle_parameters:
            from vehicle_parameters import ORACLE_KEYS
            for key in ORACLE_KEYS:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32)
        self._context_history = None
        if context_enabled:
            from envs.ego_history import EgoHistory
            self._context_history = EgoHistory(
                self._env.config["physics_world_step_size"] * self._env.config["decision_repeat"],
                include_kinematics=bool(getattr(config, "context_kinematics", True)))
            state_dim = 4 if self._context_history.include_kinematics else 2
            for key, shape in (("ctx_ego_state", (state_dim,)), ("ctx_prev_action", (2,)), ("ctx_history", (5, state_dim + 2))):
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
            spaces["ctx_mask"] = gym.spaces.Box(0, 1, shape=(5,), dtype=np.bool_)
            spaces["ctx_collision"] = gym.spaces.Box(0, 1, shape=(), dtype=np.bool_)
            spaces["ctx_episode_uid"] = gym.spaces.Box(0, np.iinfo(np.int64).max, shape=(2,), dtype=np.int64)
        self.observation_space = gym.spaces.Dict(spaces)

    @staticmethod
    def _obs(state, first=False, terminated=False, truncated=False, info=None, cost=0.0):
        info = info or {}
        last = bool(terminated or truncated)
        obs = {
            "state": np.asarray(state, dtype=np.float32),
            "is_first": first,
            "is_last": last,
            "is_terminal": bool(terminated),
            "log_success": np.float32(last and info.get("arrive_dest", False)),
            "log_crash": np.float32(last and info.get("crash", False)),
            "log_out_of_road": np.float32(last and info.get("out_of_road", False)),
            "log_route_completion": np.float32(info.get("route_completion", 0.0) if last else 0.0),
            "log_cost": np.float32(cost),
        }
        for key in ("off_lane", "yellow_line", "white_line", "crash_sidewalk"):
            obs[f"log_{key}"] = np.float32(last and info.get(key, False))
        for key in ("progress", "speed", "lateral", "terminal"):
            obs[f"log_reward_{key}"] = np.float32(info.get(f"reward_{key}", 0.0))
        return obs

    def reset(self, *, seed=None, options=None):
        # Training samples randomly; evaluation workers interleave fixed seeds.
        explicit_seed = seed is not None
        if seed is None:
            if self._evaluation:
                seed = self._start_seed + self._eval_next_index % self._num_scenarios
                self._eval_next_index += self._eval_stride
            else:
                seed = int(self._rng.integers(self._start_seed, self._start_seed + self._num_scenarios))
        if not self._start_seed <= seed < self._start_seed + self._num_scenarios:
            raise ValueError("Scenario seed is outside this environment's train/eval split")
        # Evaluation repeats identical routes/positions per map. Training samples
        # new routes/positions even when a map seed is drawn again.
        self._env.config["spawn_random_seed"] = (
            int(seed) + 2000000 if self._evaluation or explicit_seed
            else int(self._rng.integers(0, 2**31 - 1))
        )
        if getattr(self, "_chassis_enabled", False):
            chassis_seed = (
                int(np.random.SeedSequence([self._chassis_seed, int(seed), 98117]).generate_state(1)[0])
                if self._evaluation or explicit_seed
                else int(self._chassis_rng.integers(0, 2**31 - 1))
            )
            self._env.config["chassis_random_seed"] = chassis_seed
        state, _ = self._env.reset(seed=seed)
        if self._oracle_parameters:
            from vehicle_parameters import read_vehicle_parameters
            self._oracle_values = read_vehicle_parameters(self._env.vehicle)
        self._context_episode_counter += 1
        obs = self._obs(state, first=True)
        if getattr(self, "_context_history", None) is not None:
            obs.update(self._context_history.observe(self._env.vehicle))
        self._add_chassis_logs(obs)
        return obs

    @property
    def chassis_parameters(self):
        if not getattr(self, "_chassis_enabled", False):
            return None
        values = dict(self._env.engine.chassis_manager.current)
        if self._oracle_parameters and self._oracle_values is not None:
            values.update({key.removeprefix('log_chassis_'): float(value)
                           for key, value in self._oracle_values.items()})
        return values

    def _add_chassis_logs(self, obs):
        if getattr(self, "_context_history", None) is not None:
            obs["ctx_episode_uid"] = np.array(
                [self._env_id, self._context_episode_counter], dtype=np.int64)
        if getattr(self, "_chassis_enabled", False):
            from envs.chassis import LOG_KEYS
            current = self.chassis_parameters
            obs.update({f"log_chassis_{key}": np.float32(current[key]) for key in LOG_KEYS})
        if self._oracle_parameters:
            obs.update(self._oracle_values)

    def begin_evaluation(self):
        self._eval_next_index = self._env_id

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Invalid MetaDrive action: {action}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward, cost = 0.0, 0.0
        components = {f"reward_{key}": 0.0 for key in ("progress", "speed", "lateral", "terminal")}
        for _ in range(self._action_repeat):
            state, rew, terminated, truncated, info = self._env.step(action)
            reward += float(rew)
            cost += float(info.get("cost", 0.0))
            for key in components:
                components[key] += float(info.get(key, 0.0))
            if terminated or truncated:
                break
        info = {**info, **components}
        obs = self._obs(state, terminated=terminated, truncated=truncated, info=info, cost=cost)
        if getattr(self, "_context_history", None) is not None:
            obs.update(self._context_history.observe(self._env.vehicle, action))
        self._add_chassis_logs(obs)
        if getattr(self, "_chassis_enabled", False):
            info["chassis"] = self.chassis_parameters
        return obs, reward, bool(terminated or truncated), info

    def close(self):
        self._env.close()

    def render(self, *args, **kwargs):
        """Forward display rendering without changing the LiDAR observation."""
        return self._env.render(*args, **kwargs)
