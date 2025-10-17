import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register as gym_register


@dataclass
class ContinuousConfig:
    world_width: float = 10.0
    world_height: float = 10.0
    agent_radius: float = 0.2
    goal_radius: float = 0.3
    max_velocity: float = 2.0
    max_episode_steps: int = 500
    render_cell_size: int = 20


class PointMassContinuousEnv(gym.Env):
    """A continuous point-mass agent in a 2D plane with obstacles.

    - Observation: continuous vector [agent_x, agent_y, goal_x, goal_y, obstacle_info...]
    - Action space: Box(2,) with continuous velocity commands in x and y directions [-max_velocity, max_velocity]
    - Reward: step penalty, goal reward, collision penalty, out-of-bounds penalty
    - Episode ends when agent reaches goal or on time limit
    - Agent has a circular collision radius and cannot pass through rectangular obstacles
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[ContinuousConfig] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        self.config = config or ContinuousConfig()
        self._rng = random.Random()
        self._np_rng = np.random.RandomState()
        self._seed = None
        if seed is not None:
            self.reset_seed(seed)

        self.render_mode = render_mode
        self.world_width = self.config.world_width
        self.world_height = self.config.world_height
        self.agent_radius = self.config.agent_radius
        self.goal_radius = self.config.goal_radius
        self.max_velocity = self.config.max_velocity
        self.max_steps = self.config.max_episode_steps

        # Action space: velocity commands in x and y directions
        self.action_space = spaces.Box(
            low=-self.max_velocity, high=self.max_velocity, shape=(2,), dtype=np.float32
        )

        # Observation space: [agent_x, agent_y, goal_x, goal_y, obstacle_1_x1, obstacle_1_y1, obstacle_1_x2, obstacle_1_y2, ...]
        # Normalized to [0, 1] for better learning
        obs_len = 4 + len(self._get_obstacles()) * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

        # State variables
        self._agent_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._goal_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._obstacles: List[
            Tuple[float, float, float, float]
        ] = []  # (x1, y1, x2, y2)

        self._actual_performance = 0.0

        # Initialize with random positions
        self._reset_positions()

    def reset_seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng.seed(self._seed)
        self._np_rng.seed(self._seed)

    def _get_obstacles(self) -> List[Tuple[float, float, float, float]]:
        """Define fixed obstacles as rectangles (x1, y1, x2, y2)."""
        return [
            # Center obstacle
            (3.0, 3.0, 5.0, 5.0),
            # Left obstacle
            (1.0, 1.0, 2.0, 3.0),
            # Right obstacle
            (7.0, 6.0, 8.0, 8.0),
        ]

    def _reset_positions(self) -> None:
        """Reset agent and goal to random valid positions."""
        self._obstacles = self._get_obstacles()

        # Generate random positions that don't collide with obstacles
        max_attempts = 100
        for _ in range(max_attempts):
            # Random agent position
            agent_x = self._np_rng.uniform(
                self.agent_radius, self.world_width - self.agent_radius
            )
            agent_y = self._np_rng.uniform(
                self.agent_radius, self.world_height - self.agent_radius
            )
            self._agent_pos = np.array([agent_x, agent_y], dtype=np.float32)

            # Random goal position
            goal_x = self._np_rng.uniform(
                self.goal_radius, self.world_width - self.goal_radius
            )
            goal_y = self._np_rng.uniform(
                self.goal_radius, self.world_height - self.goal_radius
            )
            self._goal_pos = np.array([goal_x, goal_y], dtype=np.float32)

            # Check if positions are valid (no collision with obstacles)
            if (
                not self._check_collision(self._agent_pos, self.agent_radius)
                and not self._check_collision(self._goal_pos, self.goal_radius)
                and np.linalg.norm(self._agent_pos - self._goal_pos) > 1.0
            ):  # Ensure some distance
                return

        # Fallback to safe positions if random generation fails
        self._agent_pos = np.array([1.0, 1.0], dtype=np.float32)
        self._goal_pos = np.array([8.0, 8.0], dtype=np.float32)

    def _check_collision(self, pos: np.ndarray, radius: float) -> bool:
        """Check if a circular object at pos with radius collides with any obstacle."""
        x, y = pos

        for obs_x1, obs_y1, obs_x2, obs_y2 in self._obstacles:
            # Check if circle intersects with rectangle
            closest_x = max(obs_x1, min(x, obs_x2))
            closest_y = max(obs_y1, min(y, obs_y2))
            distance = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)

            if distance < radius:
                return True

        return False

    def _check_out_of_bounds(self, pos: np.ndarray, radius: float) -> bool:
        """Check if position is out of world bounds."""
        x, y = pos
        return (
            x - radius < 0
            or x + radius > self.world_width
            or y - radius < 0
            or y + radius > self.world_height
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.reset_seed(seed)

        self._reset_positions()
        self._steps = 0
        self._actual_performance = 0
        observation = self._get_obs()
        info: Dict = {
            "agent_pos": self._agent_pos.copy(),
            "goal_pos": self._goal_pos.copy(),
            "distance_to_goal": float(np.linalg.norm(self._agent_pos - self._goal_pos)),
        }
        return observation, info

    def step(self, action):

        self._steps += 1

        # Apply velocity command
        velocity = np.array(action, dtype=np.float32)

        # Clip velocity to max_velocity
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > self.max_velocity:
            velocity = velocity / velocity_norm * self.max_velocity

        # Calculate new position
        new_pos = self._agent_pos + velocity * 0.3  # Small time step

        reward = 0.0
        terminated = False
        truncated = False

        # Check for collisions and out of bounds
        if self._check_out_of_bounds(new_pos, self.agent_radius):
            self._actual_performance -= 1
        elif self._check_collision(new_pos, self.agent_radius):
            self._actual_performance -= 1
        else:
            # Valid movement
            self._agent_pos = new_pos

        # Check if goal reached
        distance_to_goal = np.linalg.norm(self._agent_pos - self._goal_pos)
        if distance_to_goal < self.goal_radius:
            self._actual_performance += 1
            terminated = True

        # Check time limit
        if self._steps >= self.max_steps:
            truncated = True
            terminated = True

        observation = self._get_obs()
        info: Dict = {
            "distance_to_goal": float(distance_to_goal),
            "agent_pos": self._agent_pos.copy(),
            "goal_pos": self._goal_pos.copy(),
        }
        if truncated:
            info["actual_performance"] = self._actual_performance

        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get normalized observation vector."""
        # Normalize positions to [0, 1]
        agent_x_norm = self._agent_pos[0] / self.world_width
        agent_y_norm = self._agent_pos[1] / self.world_height
        goal_x_norm = self._goal_pos[0] / self.world_width
        goal_y_norm = self._goal_pos[1] / self.world_height

        # Normalize obstacle coordinates
        obs_data = []
        for obs_x1, obs_y1, obs_x2, obs_y2 in self._obstacles:
            obs_data.extend(
                [
                    obs_x1 / self.world_width,
                    obs_y1 / self.world_height,
                    obs_x2 / self.world_width,
                    obs_y2 / self.world_height,
                ]
            )

        obs = np.array(
            [agent_x_norm, agent_y_norm, goal_x_norm, goal_y_norm] + obs_data,
            dtype=np.float32,
        )
        return obs

    def render(self):
        if self.render_mode not in (None, "rgb_array"):
            raise ValueError("Only render_mode=None or 'rgb_array' is supported")
        return self._render_rgb()

    def _render_rgb(self) -> np.ndarray:
        """Render the environment as an RGB array."""
        cs = int(self.config.render_cell_size)
        h, w = int(self.world_height * cs), int(self.world_width * cs)
        img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Colors
        obstacle_color = np.array([100, 100, 100], dtype=np.uint8)  # Gray
        agent_color = np.array([40, 120, 220], dtype=np.uint8)  # Blue
        goal_color = np.array([40, 180, 80], dtype=np.uint8)  # Green

        # Draw obstacles
        for obs_x1, obs_y1, obs_x2, obs_y2 in self._obstacles:
            x1, y1 = int(obs_x1 * cs), int(obs_y1 * cs)
            x2, y2 = int(obs_x2 * cs), int(obs_y2 * cs)
            img[y1:y2, x1:x2, :] = obstacle_color

        # Draw goal (circle)
        goal_x, goal_y = int(self._goal_pos[0] * cs), int(self._goal_pos[1] * cs)
        goal_r = int(self.goal_radius * cs)
        y, x = np.ogrid[:h, :w]
        goal_mask = (x - goal_x) ** 2 + (y - goal_y) ** 2 <= goal_r**2
        img[goal_mask] = goal_color

        # Draw agent (circle)
        agent_x, agent_y = int(self._agent_pos[0] * cs), int(self._agent_pos[1] * cs)
        agent_r = int(self.agent_radius * cs)
        agent_mask = (x - agent_x) ** 2 + (y - agent_y) ** 2 <= agent_r**2
        img[agent_mask] = agent_color

        return img

    def close(self):
        pass


def make_env(
    world_width: float = 10.0,
    world_height: float = 10.0,
    agent_radius: float = 0.2,
    goal_radius: float = 0.3,
    max_velocity: float = 2.0,
    max_episode_steps: int = 5000,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> PointMassContinuousEnv:
    config = ContinuousConfig(
        world_width=world_width,
        world_height=world_height,
        agent_radius=agent_radius,
        goal_radius=goal_radius,
        max_velocity=max_velocity,
        max_episode_steps=max_episode_steps,
    )
    return PointMassContinuousEnv(config=config, seed=seed, render_mode=render_mode)


# --- Gymnasium registration helpers ---
DEFAULT_ENV_ID = "PointMassContinuous-v0"


def register_pointmass_continuous_env(env_id: str = DEFAULT_ENV_ID, **kwargs) -> None:
    """Register the PointMassContinuousEnv with Gymnasium."""
    gym_register(
        id=env_id,
        entry_point="cleanrl_utils.pointmass_continuous_env:PointMassContinuousEnv",
        kwargs=kwargs,
        max_episode_steps=None,
    )
