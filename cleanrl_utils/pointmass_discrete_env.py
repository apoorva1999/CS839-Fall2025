import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register as gym_register
from cleanrl_utils.pointmass_continuous_env import PointMassContinuousEnv


class PointMassDiscreteEnv(PointMassContinuousEnv):
    """A continuous point-mass agent in a 2D plane with obstacles.

    - Observation: continuous vector [agent_x, agent_y, goal_x, goal_y, obstacle_info...]
    - Action space: Box(2,) with continuous velocity commands in x and y directions [-max_velocity, max_velocity]
    - Reward: step penalty, goal reward, collision penalty, out-of-bounds penalty
    - Episode ends when agent reaches goal or on time limit
    - Agent has a circular collision radius and cannot pass through rectangular obstacles
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        super().__init__()

        # TODO: Decide how many discrete actions you need.
        # Action space: some number of discrete actions
        self.action_space = spaces.Discrete(5)

        # Initialize with random positions
        self._reset_positions()

    def step(self, action):
        assert self.action_space.contains(action)
        action_to_control = {
            0: (np.array([ self.max_velocity,  0.0]), "right"),  # Right
            1: (np.array([- self.max_velocity,  0.0]), "left"),  # Left
            2: (np.array([ 0.0,  self.max_velocity]), "up"),  # Up
            3: (np.array([ 0.0, -self.max_velocity]), "down"),  # Down
            4: (np.array([ 0.0,  0.0]), "no-op"),  # No-op
        }

        # TODO: TURN DISCRETE ACTIONS INTO A CONTINUOUS ONE.
        control = action_to_control[action][0] # placeholder logic


        # Debug: Print before calling parent step

        prev_distance = np.linalg.norm(self._agent_pos - self._goal_pos)
        steps = self._steps

        if steps % 100 == 0:
            print(f"DEBUG: Before step - Agent pos: {self._agent_pos}, move: {action_to_control[action][1]}, goal pos: {self._goal_pos}, distance: {prev_distance}")

        _, _, terminated, truncated, info = super().step(control)
        observation = self._get_obs()
        new_distance = np.linalg.norm(self._agent_pos - self._goal_pos)
        reward = self.get_reward(prev_distance, new_distance, terminated)
        # reward -= 0.1  
        if steps % 100 == 0:
            print(f"DEBUG: After step - Agent pos: {self._agent_pos}, goal pos: {self._goal_pos}, distance: {new_distance}, reward: {reward}")
       

        return observation, reward, terminated, truncated, info

    def get_reward(self, prev_distance, new_distance, terminated):
        progress = (prev_distance - new_distance) 
        reward = progress * 100.0
        reward -= 0.05
        collided = self._check_collision(self._agent_pos, self.agent_radius)
        out_of_bounds = self._check_out_of_bounds(self._agent_pos, self.agent_radius)
        if collided or out_of_bounds:
            reward -= 2.0  # collision penalty
        if terminated and new_distance < self.goal_radius:
            reward += 100.0  # goal bonus

        return reward


    def _get_obs(self) -> np.ndarray:
        """Get normalized observation vector."""
        obs = super()._get_obs()

        # TODO: CHANGE THE OBSERVATION IF YOU LIKE.

        return obs

    def close(self):
        pass


def make_env(
    world_width: float = 10.0,
    world_height: float = 10.0,
    agent_radius: float = 0.2,
    goal_radius: float = 0.3,
    max_velocity: float = 2.0,
    max_episode_steps: int = 500,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> PointMassDiscreteEnv:
    return PointMassDiscreteEnv(seed=seed, render_mode=render_mode)


# --- Gymnasium registration helpers ---
DEFAULT_ENV_ID = "PointMassDiscrete-v0"


def register_pointmass_discrete_env(env_id: str = DEFAULT_ENV_ID, **kwargs) -> None:
    """Register the PointMassContinuousEnv with Gymnasium."""
    gym_register(
        id=env_id,
        entry_point="cleanrl_utils.pointmass_discrete_env:PointMassDiscreteEnv",
        kwargs=kwargs,
        max_episode_steps=None,
    )
