# TODO: Simplify and remove sub-goals/elsciRL tracking/live_env/exp sampling
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from gymnasium.wrappers import TimeLimit
# ------ Imports -----------------------------------------
# Agent Setup
from elsciRL.environment_setup.imports import ImportHelper
# Evaluation standards
from elsciRL.environment_setup.results_table import ResultsTable
from elsciRL.environment_setup.elsciRL_info import elsciRLInfo
# Non-gym interaction loop setup
from elsciRL.interaction_loops.standard import StandardInteractionLoop
from elsciRL.experiments.experiment_utils.config_utils import ensure_dir


def _apply_action_limit(env, max_steps: int | None):
    """Wrap env with a TimeLimit so runaway episodes truncate after max_steps."""

    if not max_steps or max_steps <= 0:
        return env
    if isinstance(env, TimeLimit):
        env._max_episode_steps = min(env._max_episode_steps, max_steps)
        return env
    try:
        return TimeLimit(env, max_episode_steps=max_steps)
    except Exception:
        # Fall back to manual attribute hints if wrapper fails (non-gym envs)
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


class GymInteractionLoop:
    """Interaction Loop for standard environments.
    REQUIRES:
        - Engine: Environment engine defined with elsciRLAI format
        - Adapters: Dictionary of local adapters with unique names: {"name_1": Adapter_1, "name_2": Adapter_2,...}
        - local_setup_info: Dictionary of local setup info (i.e. local config file)
    """
    def __init__(self, Engine, Adapters:dict, local_setup_info: dict):
        # Define agent type for interaction process, call alternative if not gym agent
        if local_setup_info['agent_type'].split('_')[0] == "SB3":
            self.gym_agent = True
            Imports = ImportHelper(local_setup_info)
            self.agent, self.agent_type, self.agent_name, self.agent_state_adapter = Imports.agent_info(Adapters)
            self.num_train_episodes, self.num_test_episodes, self.training_action_cap, self.testing_action_cap, self.reward_signal = Imports.parameter_info()  
            self.train = Imports.training_flag()
            # --- INIT env from engine
            self.env = Engine(local_setup_info)
            max_steps = self.training_action_cap if self.train else self.testing_action_cap
            self.env = _apply_action_limit(self.env, max_steps)
            self.start_obs = self.env.reset()
            # ---
            # --- PRESET elsciRL INFO
            # Agent
            # Training or testing phase flag
            # --- elsciRL
            self.live_env, self.observed_states, self.experience_sampling = Imports.live_env_flag()
            # Results formatting
            self.results = ResultsTable(local_setup_info)
            # elsciRL input function
            # - We only want to init trackers on first batch otherwise it resets knowledge
            self.elsciRL = elsciRLInfo(self.observed_states, self.experience_sampling)
        else:
            # --- Used for initialisation default interaction loop as alternative
            self.gym_agent = False
            self.interaction = StandardInteractionLoop(Engine, Adapters, local_setup_info)
            self.start_obs = self.interaction.start_obs
            self.results = ResultsTable(local_setup_info)

    def episode_loop(self, render:bool=False, render_save_dir:str=None):
        if self.gym_agent:
            # Mode selection (already initialized)
            if self.train:
                number_episodes = self.num_train_episodes
            else:
                number_episodes = self.num_test_episodes
            
            episode_render = []
            print("\n Episode Interaction Loop: ")
            if self.train:
                for episode in tqdm(range(0, number_episodes)):
                    start_time = time.time()
                    # Can force the agent to train on a single episode
                    # Very time consuming to do this
                    self.agent.learn(total_steps=self.training_action_cap)
                    end_time = time.time()
                    reward, actions, states, render_stack = self.agent.test(self.env, render=render)
                    episode_render.append(render_stack)
                    # Need to get values from actions
                    # TODO: Ensure all agents output int directly to solve this
                    if isinstance(actions[0], np.int64):
                        actions = [action.item() for action in actions]
                    elif isinstance(actions[0], np.ndarray):
                        actions = [action.item() for action in actions]

                    

                    self.results.results_per_episode(self.agent_name, None, episode, len(actions), 
                                                    reward, (end_time-start_time), actions, 0, 0)  
            else:
                for episode in tqdm(range(0, number_episodes)):
                    start_time = time.time()
                    # Evaluate fixed policy on single episode
                    reward, actions, states, render_stack = self.agent.test(self.env, render=render)
                    # Need to get values from actions
                    # TODO: Ensure all agents output int directly to solve this
                    if isinstance(actions[0], np.int64):
                        actions = [action.item() for action in actions]
                    elif isinstance(actions[0], np.ndarray):
                        actions = [action.item() for action in actions]

                    episode_render.append(render_stack)
                    end_time = time.time()
                    self.results.results_per_episode(self.agent_name, None, episode, len(actions), 
                                                    reward, (end_time-start_time), actions, 0, 0)
            table_results = self.results.results_table_format()
            # Output GIF image of all episode frames
            if render and render_stack:
                frames = _normalize_render_stack(render_stack)
                if frames:
                    frames[0].save(
                        render_save_dir + '/render.gif',
                        save_all=True,
                        append_images=frames[1:],
                        optimize=False,
                        duration=200,
                        loop=1,
                    )
        else:
            table_results = self.interaction.episode_loop()
            self.agent = self.interaction.agent
            self.results = self.interaction.results
            self.elsciRL = self.interaction.elsciRL

        return table_results

    @staticmethod
    def policy_rollout(
        agent,
        env,
        agent_name: str,
        num_episodes: int,
        results_table,
        render: bool = False,
        render_save_dir: str | None = None,
        action_limit: int | None = None,
    ):
        """Execute a pre-configured policy-gradient agent on a Gym env and log results."""
        env = _apply_action_limit(env, action_limit)
        episode_render = []
        for episode in range(num_episodes):
            start_time = time.time()
            reward, actions, states, render_stack = agent.test(env, render=render)
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