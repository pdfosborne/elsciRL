"""Gym wrapper utilities for instruction-following reward shaping."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from elsciRL.environment_setup.gym_wrapper_abstract import RewardWrapper


class InstructionRewardWrapper(RewardWrapper):
    """Adds adapter-derived instruction rewards to a Gym environment."""

    def __init__(self, env, reward_fn: Optional[Callable[[np.ndarray | None, Dict], float]] = None):
        super().__init__(env)
        self.reward_fn = reward_fn

    def reward(self, reward):
        if self.reward_fn is None:
            return reward
        obs = getattr(self.env, "last_obs", None)
        info = getattr(self.env, "last_info", {})
        shaped_reward = self.reward_fn(obs, info)
        if shaped_reward is None:
            return reward
        return reward + shaped_reward
