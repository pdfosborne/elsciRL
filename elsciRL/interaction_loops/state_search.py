# TODO: Simplify and remove sub-goals/elsciRL tracking/live_env/exp sampling
import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, cpu_count
# ------ Imports -----------------------------------------
# Agent Setup
from elsciRL.environment_setup.imports import ImportHelper

# Evaluation standards
from elsciRL.environment_setup.results_table import ResultsTable
from elsciRL.environment_setup.elsciRL_info import elsciRLInfo


class StateSearchInteractionLoop:
    """Interaction Loop for standard environments.
    REQUIRES:
        - Engine: Environment engine defined with elsciRLAI format
        - Adapters: Dictionary of local adapters with unique names: {"name_1": Adapter_1, "name_2": Adapter_2,...}
        - local_setup_info: Dictionary of local setup info (i.e. local config file)
    """
    
    def __init__(self, Engine, Adapters: dict, local_setup_info: dict):
        # --- INIT state space from engine
        self.agent_adapter_name = local_setup_info['agent_type'] + "_" + local_setup_info['adapter_select']
        self.engine = Engine(local_setup_info)
        self.start_obs = self.engine.reset()
        # --- PRESET elsciRL INFO
        # Agent
        Imports = ImportHelper(local_setup_info)
        self.agent, self.agent_type, self.agent_name, self.agent_state_adapter = (
            Imports.agent_info(Adapters)
        )
        (
            self.num_train_episodes,
            self.num_test_episodes,
            self.training_action_cap,
            self.testing_action_cap,
            self.reward_signal,
        ) = Imports.parameter_info()

        # Training or testing phase flag
        self.train = Imports.training_flag()

        # Mode selection (already initialized)
        if self.train:
            self.number_episodes = self.num_train_episodes
        else:
            self.number_episodes = self.num_test_episodes
        # --- elsciRL
        self.live_env, self.observed_states = (
            Imports.live_env_flag()
        )
        # Results formatting
        self.results = ResultsTable(local_setup_info)
        # elsciRL input function
        # - We only want to init trackers on first batch otherwise it resets knowledge
        self.elsciRL = elsciRLInfo(self.observed_states)
        

    def episode_loop(self, render: bool = False, render_save_dir: str = None):
        # RENDER AND SUB-GOALS REMOVED COMPLETELY SO SAVE RUN-TIME
        # Parallel processing for faster episode runs
        parallel = Parallel(n_jobs=cpu_count(True), prefer="threads", verbose=0)
        print("\n Episode Interaction Loop: ")
        for episode in parallel(tqdm(range(0, self.number_episodes))):
            action_history = []
            # ---
            # Start observation is used instead of .reset()  fn so that this can be overridden for repeat analysis from the same start pos
            obs = self.engine.reset(start_obs=self.start_obs)
            legal_moves = self.engine.legal_move_generator(obs)

            # LLM agents need to pass the state as a string
            if self.agent_type.split("_")[0] == "LLM":
                state = self.agent_state_adapter.adapter(
                state=obs,
                legal_moves=legal_moves,
                episode_action_history=action_history,
                encode=False,
            )
            else:
                state = self.agent_state_adapter.adapter(
                    state=obs,
                    legal_moves=legal_moves,
                    episode_action_history=action_history,
                    encode=True,
                )
            # ---
            start_time = time.time()
            episode_reward: int = 0
            # ---
            for action in range(0, self.training_action_cap):
                if self.live_env:
                    # Agent takes action
                    legal_moves = self.engine.legal_move_generator(obs)
                    agent_action = self.agent.policy(state, legal_moves)

                    if isinstance(agent_action, np.int64):
                        action_history.append(agent_action.item())
                    else:
                        action_history.append(agent_action)

                    next_obs, reward, terminated, _ = self.engine.step(
                        state=obs, action=agent_action
                    )
                    
                    # Can override reward per action with small negative punishment
                    if reward == 0:
                        reward = self.reward_signal[1]

                    legal_moves = self.engine.legal_move_generator(next_obs)
                    # LLM agents need to pass the state as a string
                    if self.agent_type.split("_")[0] == "LLM":
                        next_state = self.agent_state_adapter.adapter(
                        state=next_obs,
                        legal_moves=legal_moves,
                        episode_action_history=action_history,
                        encode=False,
                    )
                    else:
                        next_state = self.agent_state_adapter.adapter(
                            state=next_obs,
                            legal_moves=legal_moves,
                            episode_action_history=action_history,
                            encode=True,
                        )
                    # elsciRL trackers
                    self.elsciRL.observed_state_tracker(
                        engine_observation=next_obs,
                        language_state=self.agent_state_adapter.adapter(
                            state=next_obs,
                            legal_moves=legal_moves,
                            episode_action_history=action_history,
                            encode=False,
                        ),
                    )

                episode_reward += reward
                if terminated:
                    break
                else:
                    state = next_state
                    if self.live_env:
                        obs = next_obs

            # If action limit reached
            if not terminated:
                reward = self.reward_signal[2]

            end_time = time.time()
            try:
                agent_results = self.agent.q_result()
            except:
                agent_results = [0, 0]

            if self.live_env:
                self.results.results_per_episode(
                    self.agent_name,
                    None,
                    episode,
                    action,
                    episode_reward,
                    (end_time - start_time),
                    action_history,
                    agent_results[0],
                    agent_results[1],
                )
            # Check if action language mapping is working
            if self.agent_type.split("_")[0] == "LLM":
                print(f"\n ++++++++++++ \n Action language mapping knowledge: {self.agent.action_language_mapping_encoder}")
        # Output GIF image of all episode frames
        return self.results.results_table_format()
