# TODO: Simplify and remove sub-goals/elsciRL tracking/live_env/exp sampling
import time
import numpy as np
from tqdm import tqdm
# ------ Imports -----------------------------------------
# Agent Setup
from elsciRL.environment_setup.imports import ImportHelper
# Evaluation standards
from elsciRL.environment_setup.results_table import ResultsTable
from elsciRL.environment_setup.elsciRL_info import elsciRLInfo
# Non-gym interaction loop setup
from elsciRL.interaction_loops.standard import StandardInteractionLoop

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
            # --- INIT env from engine
            self.env = Engine(local_setup_info)
            self.start_obs = self.env.reset()
            # ---
            # --- PRESET elsciRL INFO
            # Agent
            Imports = ImportHelper(local_setup_info)
            self.agent, self.agent_type, self.agent_name, self.agent_state_adapter = Imports.agent_info(Adapters)
            self.num_train_episodes, self.num_test_episodes, self.training_action_cap, self.testing_action_cap, self.reward_signal = Imports.parameter_info()  
            # Training or testing phase flag
            self.train = Imports.training_flag()
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
                    self.results.results_per_episode(self.agent_name, None, episode, len(actions), 
                                                    reward, (end_time-start_time), actions, 0, 0)  
            else:
                for episode in tqdm(range(0, number_episodes)):
                    start_time = time.time()
                    # Evaluate fixed policy on single episode
                    reward, actions, states, render_stack = self.agent.test(self.env, render=render)
                    episode_render.append(render_stack)
                    end_time = time.time()
                    self.results.results_per_episode(self.agent_name, None, episode, len(actions), 
                                                    reward, (end_time-start_time), actions, 0, 0)
            table_results = self.results.results_table_format()
            # Output GIF image of all episode frames
            if render:
                render_stack[0].save(render_save_dir+'/render.gif', save_all=True, append_images=render_stack[1:], optimize=False, duration=200, loop=1)
        else:
            table_results = self.interaction.episode_loop()
            self.agent = self.interaction.agent
            self.results = self.interaction.results
            self.elsciRL = self.interaction.elsciRL

        return table_results