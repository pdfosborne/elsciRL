# TODO: Simplify and remove sub-goals/elsciRL tracking/live_env/exp sampling
import time
import numpy as np
from tqdm import tqdm
from PIL import Image  # Used to generate GIF

# ------ Imports -----------------------------------------
# Agent Setup
from elsciRL.environment_setup.imports import ImportHelper

# Evaluation standards
from elsciRL.environment_setup.results_table import ResultsTable
from elsciRL.environment_setup.elsciRL_info import elsciRLInfo


class StandardInteractionLoop:
    """Interaction Loop for standard environments.
    REQUIRES:
        - Engine: Environment engine defined with elsciRLAI format
        - Adapters: Dictionary of local adapters with unique names: {"name_1": Adapter_1, "name_2": Adapter_2,...}
        - local_setup_info: Dictionary of local setup info (i.e. local config file)
    """

    def __init__(self, Engine, Adapters: dict, local_setup_info: dict):
        # --- INIT state space from engine
        self.engine = Engine(local_setup_info)
        self.start_obs = self.engine.reset()
        # if self.engine.render() is not None:
        #     self.render = self.engine.render()
        # else:
        #     self.render = None
        # ---
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
        # --- elsciRL
        self.live_env, self.observed_states, self.experience_sampling = (
            Imports.live_env_flag()
        )
        # Results formatting
        self.results = ResultsTable(local_setup_info)
        # elsciRL input function
        # - We only want to init trackers on first batch otherwise it resets knowledge
        self.elsciRL = elsciRLInfo(self.observed_states, self.experience_sampling)
        # Env start position for instr input
        # Enable sub-goals
        if (
            (local_setup_info["sub_goal"] is not None)
            & (local_setup_info["sub_goal"] != ["None"])
            & (local_setup_info["sub_goal"] != "None")
        ):
            self.sub_goal: list = local_setup_info["sub_goal"]
        else:
            self.sub_goal: list = None
        # New: multi sub-goal paths used for continuous instruction plans
        # - Would not be defined by config files but rather experiment module so dont need to define in terms of local_setup_info
        self.multi_sub_goal: dict = None

    def episode_loop(self, render: bool = False, render_save_dir: str = None):
        # Mode selection (already initialized)
        if self.train:
            number_episodes = self.num_train_episodes
        else:
            number_episodes = self.num_test_episodes

        sub_goal_tracker = (
            {}
        )  # Used to track sub-goal completion -> re-call most common end state for start of next with EXACT chaining
        sub_goal_best = 0
        self.sub_goal_end = None
        # NEW: Animation of Learning
        if render:
            render_stack = []
        print("\n Episode Interaction Loop: ")
        for episode in tqdm(range(0, number_episodes)):
            action_history = []
            # ---
            # Start observation is used instead of .reset()  fn so that this can be overridden for repeat analysis from the same start pos
            obs = self.engine.reset(start_obs=self.start_obs)
            if render:
                render_stack.append(Image.fromarray(self.engine.render().astype("uint8")))
            legal_moves = self.engine.legal_move_generator(obs)
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
            sub_goal_reward_obtained = False
            multi_sub_goal_tracker = (
                []
            )  # Used to tracked known sub-goal completion and reduce print clutter
            multi_goal_reward_obtained = []

            for action in range(0, self.training_action_cap):
                if self.live_env:
                    # Agent takes action
                    legal_moves = self.engine.legal_move_generator(obs)
                    agent_action = self.agent.policy(state, legal_moves)
                    action_history.append(agent_action)

                    next_obs, reward, terminated, _ = self.engine.step(
                        state=obs, action=agent_action
                    )
                    if render:
                        render_stack.append(
                            Image.fromarray(self.engine.render().astype("uint8"))
                        )

                    # Can override reward per action with small negative punishment
                    if reward == 0:
                        reward = self.reward_signal[1]

                    legal_moves = self.engine.legal_move_generator(next_obs)
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

                    # MUST COME BEFORE SUB-GOAL CHECK OR 'TERMINAL STATES' WILL BE FALSE
                    self.elsciRL.experience_sampling_add(
                        state, agent_action, next_state, reward, terminated
                    )
                    # Trigger end on sub-goal if defined
                    if self.sub_goal:
                        if (type(self.sub_goal) == type("")) | (
                            type(self.sub_goal) == type(0)
                        ):
                            if next_obs == self.sub_goal:
                                # Sub-goal no longer terminates episode but only gets reward once
                                if sub_goal_reward_obtained is False:
                                    reward = self.reward_signal[0]
                                    sub_goal_reward_obtained = True
                                else:
                                    reward = 0
                                # terminated = True
                                # Required if we want exact sub-goal end position for next instruction
                                if next_obs not in sub_goal_tracker:
                                    sub_goal_tracker[next_obs] = 1
                                else:
                                    sub_goal_tracker[next_obs] = (
                                        sub_goal_tracker[next_obs] + 1
                                    )
                                if sub_goal_tracker[next_obs] >= sub_goal_best:
                                    self.sub_goal_end = next_obs
                                    sub_goal_best = sub_goal_tracker[next_obs]

                        elif (type(self.sub_goal) == type(list(""))) | (
                            type(self.sub_goal) == type([0])
                        ):
                            if next_obs in self.sub_goal:
                                # Sub-goal no longer terminates episode but only gets reward once
                                if sub_goal_reward_obtained is False:
                                    reward = self.reward_signal[0]
                                    sub_goal_reward_obtained = True
                                else:
                                    reward = 0
                                # terminated = True
                                # Required if we want exact sub-goal end position for next instruction
                                if next_obs not in sub_goal_tracker:
                                    sub_goal_tracker[next_obs] = 1
                                else:
                                    sub_goal_tracker[next_obs] = (
                                        sub_goal_tracker[next_obs] + 1
                                    )
                                if sub_goal_tracker[next_obs] >= sub_goal_best:
                                    self.sub_goal_end = next_obs
                                    sub_goal_best = sub_goal_tracker[next_obs]
                                # print("---------------------------------")
                                # print("Sub-Goal Reached: ", next_obs)
                        else:
                            print(
                                "Sub-Goal Type ERROR: The input sub-goal type must be a str/int or list(str/int)."
                            )
                    # New: multi sub-goal completion for continuous chaining
                    # - If sub-task completed, obtain scaled reward based on instruction scale
                    if self.multi_sub_goal:
                        for multi_goal in self.multi_sub_goal:
                            if next_obs in self.multi_sub_goal[multi_goal]["sub_goal"]:
                                # Only get reward for first visit in episode
                                if multi_goal not in multi_goal_reward_obtained:
                                    multi_goal_reward_scale = self.multi_sub_goal[
                                        multi_goal
                                    ]["reward_scale"]
                                    reward = (
                                        self.reward_signal[0] * multi_goal_reward_scale
                                    )
                                if multi_goal not in multi_sub_goal_tracker:
                                    multi_sub_goal_tracker.append(multi_goal)
                                    print(
                                        "\n - Multi sub-goal completed: ",
                                        multi_goal,
                                        " -> reward after scaling = ",
                                        reward,
                                    )
                                break  # Assume state is unique to completing one sub-step (first used)
                else:
                    # Experience Sampling
                    legal_moves = self.elsciRL.experience_sampling_legal_actions(state)
                    # Unknown state, have no experience to sample from so force break episode
                    if legal_moves == None:
                        break

                    agent_action = self.agent.policy(state, legal_moves)
                    next_state, reward, terminated = (
                        self.elsciRL.experience_sampling_step(state, agent_action)
                    )

                if self.train:
                    self.agent.learn(state, next_state, reward, agent_action)
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

        # Output GIF image of all episode frames
        if render:
            render_stack[0].save(
                render_save_dir + "/render.gif",
                save_all=True,
                append_images=render_stack[1:],
                optimize=False,
                duration=200,
                loop=1,
            )

        if self.sub_goal:
            if sub_goal_best > 0:
                print(
                    "\n \t - Most Common Sub-Goal Reached: ",
                    self.sub_goal_end,
                    "\n ---------- ",
                )
            else:
                print("\n \t - Sub-Goal Not Reached \n ---------- ")
        return self.results.results_table_format()
