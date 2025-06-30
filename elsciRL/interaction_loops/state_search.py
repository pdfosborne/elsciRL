import time
import numpy as np
from tqdm import tqdm

# ------ Imports -----------------------------------------
# Agent Setup
from elsciRL.environment_setup.imports import ImportHelper

# Evaluation standards
from elsciRL.environment_setup.results_table import ResultsTable
from elsciRL.environment_setup.elsciRL_info import elsciRLInfo


def episode_loop(Engine, Adapters: dict, local_setup_info: dict, number_episodes: int = 1000, 
                 batch_number: int = 0, observed_states: dict = {}) -> dict:
    # --- INIT state space from engine
    agent_adapter_name = local_setup_info['agent_type'] + "_" + local_setup_info['adapter_select']
    engine = Engine(local_setup_info)
    start_obs = engine.reset()
    # --- PRESET elsciRL INFO
    # Agent
    Imports = ImportHelper(local_setup_info)
    agent, agent_type, agent_name, agent_state_adapter = (
        Imports.agent_info(Adapters)
    )
    (
        num_train_episodes,
        num_test_episodes,
        training_action_cap,
        testing_action_cap,
        reward_signal,
    ) = Imports.parameter_info()

    # Training or testing phase flag
    train = Imports.training_flag()

    # Mode selection (already initialized)
    # --- elsciRL
    live_env, observed_states_flag = (
        Imports.live_env_flag()
    )
    # Results formatting
    results = ResultsTable(local_setup_info)
    # elsciRL input function
    # - We only want to init trackers on first batch otherwise it resets knowledge
    elsciRL = elsciRLInfo(observed_states)
    # RENDER AND SUB-GOALS REMOVED COMPLETELY SO SAVE RUN-TIME
    
    for episode in tqdm(range(0, number_episodes)):
        action_history = []
        # ---
        # Start observation is used instead of .reset()  fn so that this can be overridden for repeat analysis from the same start pos
        obs = engine.reset(start_obs=start_obs)
        legal_moves = engine.legal_move_generator(obs)

        # LLM agents need to pass the state as a string
        if agent_type.split("_")[0] == "LLM":
            state = agent_state_adapter.adapter(
            state=obs,
            legal_moves=legal_moves,
            episode_action_history=action_history,
            encode=False,
        )
        else:
            state = agent_state_adapter.adapter(
                state=obs,
                legal_moves=legal_moves,
                episode_action_history=action_history,
                encode=True,
            )
        # ---
        start_time = time.time()
        episode_reward: int = 0
        # ---
        for action in range(0, training_action_cap):
            if live_env:
                # Agent takes action
                legal_moves = engine.legal_move_generator(obs)
                agent_action = agent.policy(state, legal_moves)

                if isinstance(agent_action, np.int64):
                    action_history.append(agent_action.item())
                else:
                    action_history.append(agent_action)

                next_obs, reward, terminated, _ = engine.step(
                    state=obs, action=agent_action
                )
                
                # Can override reward per action with small negative punishment
                if reward == 0:
                    reward = reward_signal[1]

                # Only update observed states if not already observed
                if next_obs not in observed_states:
                    legal_moves = engine.legal_move_generator(next_obs)
                    # LLM agents need to pass the state as a string
                    if agent_type.split("_")[0] == "LLM":
                        next_state = agent_state_adapter.adapter(
                        state=next_obs,
                        legal_moves=legal_moves,
                        episode_action_history=action_history,
                        encode=False,
                    )
                    else:
                        next_state = agent_state_adapter.adapter(
                            state=next_obs,
                            legal_moves=legal_moves,
                            episode_action_history=action_history,
                            encode=True,
                        )
                    # elsciRL trackers
                    # TODO: Consider adding prior action history to the tracker so that we can 
                    #  transform observed data across adapters without loss of information
                    observed_states = elsciRL.observed_state_tracker(
                        engine_observation=next_obs,
                        language_state=agent_state_adapter.adapter(
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
                if live_env:
                    obs = next_obs

        # If action limit reached
        if not terminated:
            reward = reward_signal[2]

        end_time = time.time()
        try:
            agent_results = agent.q_result()
        except:
            agent_results = [0, 0]

        if live_env:
            results.results_per_episode(
                agent_name,
                None,
                episode,
                action,
                episode_reward,
                (end_time - start_time),
                action_history,
                agent_results[0],
                agent_results[1],
            )
    # Output GIF image of all episode frames
    return observed_states
