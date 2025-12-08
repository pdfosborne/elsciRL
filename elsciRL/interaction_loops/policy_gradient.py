import time
from typing import Optional

import numpy as np
from PIL import Image
from gymnasium.wrappers import TimeLimit

from elsciRL.environment_setup.imports import ImportHelper
from elsciRL.environment_setup.results_table import ResultsTable
from elsciRL.environment_setup.elsciRL_info import elsciRLInfo
from elsciRL.experiments.experiment_utils.config_utils import ensure_dir


def apply_action_limit(env, max_steps: Optional[int]):
    """Wrap env with a TimeLimit so runaway episodes truncate after max_steps."""

    if not max_steps or max_steps <= 0:
        return env
    if isinstance(env, TimeLimit):
        env._max_episode_steps = (
            min(env._max_episode_steps, max_steps)
            if getattr(env, "_max_episode_steps", None)
            else max_steps
        )
        return env
    try:
        return TimeLimit(env, max_episode_steps=max_steps)
    except Exception:
        setattr(env, "_elsci_max_episode_steps", max_steps)
        return env


def _normalize_render_stack(render_stack):
    """Convert renderer outputs to PIL Images so GIF saving works consistently."""

    normalized = []
    for frame in render_stack or []:
        if frame is None:
            continue
        if hasattr(frame, "save"):
            normalized.append(frame)
        elif isinstance(frame, np.ndarray):
            normalized.append(Image.fromarray(frame.astype(np.uint8)))
    return normalized


class PolicyGradientInteractionLoop:
    """Interaction loop tailored for policy-gradient Gym agents."""

    def __init__(self, Engine, Adapters: dict, local_setup_info: dict):
        self.setup_info = local_setup_info
        self.agent_type = local_setup_info['agent_type']
        Imports = ImportHelper(local_setup_info)
        self.agent, _, self.agent_name, self.agent_state_adapter = Imports.agent_info(Adapters)
        (
            self.num_train_episodes,
            self.num_test_episodes,
            self.training_action_cap,
            self.testing_action_cap,
            self.reward_signal,
        ) = Imports.parameter_info()
        self.train = Imports.training_flag()
        self.live_env, self.observed_states, self.experience_sampling = Imports.live_env_flag()

        self.env = Engine(local_setup_info)
        max_steps = self.training_action_cap if self.train else self.testing_action_cap
        self.env = apply_action_limit(self.env, max_steps)
        self.start_obs = self.env.reset()
        self.results = ResultsTable(local_setup_info)
        self.elsciRL = elsciRLInfo(self.observed_states, self.experience_sampling)

    def episode_loop(self, render: bool = False, render_save_dir: Optional[str] = None):
        num_episodes = self.num_train_episodes if self.train else self.num_test_episodes
        table_results = self._run_rollout(
            agent=self.agent,
            env=self.env,
            agent_name=self.agent_name,
            num_episodes=num_episodes,
            results_table=self.results,
            train=self.train,
            training_action_cap=self.training_action_cap,
            testing_action_cap=self.testing_action_cap,
            render=render,
            render_save_dir=render_save_dir,
        )
        return table_results

    @staticmethod
    def _run_rollout(
        agent,
        env,
        agent_name: str,
        num_episodes: int,
        results_table,
        train: bool,
        training_action_cap: Optional[int] = None,
        testing_action_cap: Optional[int] = None,
        render: bool = False,
        render_save_dir: Optional[str] = None,
    ):
        action_limit = training_action_cap if train else testing_action_cap
        env = apply_action_limit(env, action_limit)

        episode_render = []
        for episode in range(num_episodes):
            start_time = time.time()
            if train:
                total_steps = action_limit if action_limit and action_limit > 0 else 1
                agent.learn(total_steps=total_steps)
            reward, actions, _, render_stack = agent.test(env, render=render)
            end_time = time.time()

            if actions:
                if isinstance(actions[0], np.int64):
                    actions = [action.item() for action in actions]
                elif isinstance(actions[0], np.ndarray):
                    actions = [action.item() for action in actions]

            results_table.results_per_episode(
                agent_name,
                None,
                episode,
                len(actions),
                reward,
                (end_time - start_time),
                actions,
                0,
                0,
            )
            if render and render_stack:
                episode_render.extend(_normalize_render_stack(render_stack))

        table_results = results_table.results_table_format()
        if render and episode_render:
            ensure_dir(render_save_dir or "renders")
            episode_render[0].save(
                f"{render_save_dir or 'renders'}/{agent_name}_policy.gif",
                save_all=True,
                append_images=episode_render[1:],
                optimize=False,
                duration=200,
                loop=1,
            )
        return table_results

    @classmethod
    def policy_rollout(
        cls,
        agent,
        env,
        agent_name: str,
        num_episodes: int,
        results_table,
        render: bool = False,
        render_save_dir: Optional[str] = None,
        action_limit: Optional[int] = None,
    ):
        return cls._run_rollout(
            agent=agent,
            env=env,
            agent_name=agent_name,
            num_episodes=num_episodes,
            results_table=results_table,
            train=False,
            training_action_cap=None,
            testing_action_cap=action_limit,
            render=render,
            render_save_dir=render_save_dir,
        )
