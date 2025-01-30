import os
import pandas as pd
import numpy as np
import json
import random

# ------ Interaction Protocol -----------------------------------
from elsciRL.interaction_loops.standard import StandardInteractionLoop
# ------ Experiment Import --------------------------------------
from elsciRL.evaluation.standard_report import Evaluation
# ------ Agent Imports -----------------------------------------
# Universal Agents
from elsciRL.agents.agent_abstract import Agent, QLearningAgent
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.neural_q_agent import NeuralQLearningAgent
# ------ Gym Experiement ----------------------------------------
from elsciRL.experiments.GymExperiment import GymExperiment
from elsciRL.agents.stable_baselines.DQN import SB_DQN
from elsciRL.agents.stable_baselines.PPO import SB_PPO
from elsciRL.agents.stable_baselines.A2C import SB_A2C

# TODO: Enable any number of the same agent types with varying parameters
AGENT_TYPES = {
    "Qlearntab": TableQLearningAgent,
    "DQN": NeuralQLearningAgent,
    "DQN_2": NeuralQLearningAgent,
    "DQN_language": NeuralQLearningAgent,
    "Random": random,
    "SB3_DQN": SB_DQN,
    "SB3_PPO": SB_PPO,
    "SB3_A2C": SB_A2C
}

PLAYER_PARAMS = {
    "Qlearntab": ["alpha", "gamma", "epsilon"],
    "DQN": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "DQN_2": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "DQN_language": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "Random": [],
    "SB3_DQN": ["policy"],
    "SB3_PPO": ["policy"],
    "SB3_A2C": ["policy"]
}

# This is the main run functions for elsciRL to be imported
# Defines the train/test operators and imports all the required agents and experiment functions ready to be used
# The local main.py file defines the [adapters, configs, environment] to be input

# This should be where the environment is initialized and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead
# -> This means we have multiple main.py types (e.g. with/without convergence measure) so should create a directory and finalise naming for this

class elsciRLOptimize:
    def __init__(self, Config:dict, LocalConfig:dict, 
                 Engine, Adapters:dict,
                 save_dir:str, show_figures:str, window_size:float,
                 instruction_path: dict=None, predicted_path: dict=None, instruction_episode_ratio:float=0.1,
                 instruction_chain:bool=False, instruction_chain_how:str='None'):
        self.ExperimentConfig = Config
        self.LocalConfig = LocalConfig
        try:
            self.setup_info = self.ExperimentConfig['data'] | self.LocalConfig['data']
        except:
            self.setup_info = self.ExperimentConfig | self.LocalConfig
        self.engine = Engine
        self.adapters = Adapters
        self.env = StandardInteractionLoop
        
        # New: Flag for if using gym agents to optimize instead
        # - Generates reward signal from instructions that is passed to gym eng translator
        if self.setup_info['agent_select'][0].split('_')[0] == "SB3":
            self.is_gym_agent = True
            self.sub_goal_reward = self.setup_info['reward_signal'][0]
            self.gym_exp = GymExperiment(Config=self.ExperimentConfig, ProblemConfig=self.LocalConfig, 
                    Engine=self.engine, Adapters=self.adapters,
                    save_dir=save_dir, show_figures = 'No', window_size=0.1)
            # Get start position to start instr chains
            train_setup_info = self.setup_info.copy()
            agent_type = "Qlearntab" # Force agent to Qlearntab for compatibility
            adapter = train_setup_info["adapter_select"][0]
            # ----- Agent parameters
            agent_parameters = train_setup_info["agent_parameters"][agent_type]
            train_setup_info['agent_type'] = agent_type
            train_setup_info['agent_name'] = str(agent_type) + '_' + str(adapter) + '_' + str(agent_parameters)
            train_setup_info['adapter_select'] = adapter
            agent = AGENT_TYPES[agent_type](**agent_parameters)
            train_setup_info['agent'] = agent
            train_setup_info['train'] = True
            train_setup_info['live_env'] = True
            train_setup_info['training_results'] = False
            train_setup_info['observed_states'] = False
            train_setup_info['experience_sampling'] = False
            live_env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=train_setup_info)
            self.start_obs = live_env.start_obs
        else:
            self.is_gym_agent = False


        if not predicted_path:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.save_dir = save_dir+'/Instr_Experiment'
        else:
            save_dir_extra = save_dir.split("/")[-1]
            save_dir = '/'.join(save_dir.split("/")[:-1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.save_dir = save_dir+'/'+save_dir_extra
        self.show_figures = show_figures
        if (self.show_figures.lower() != 'y')|(self.show_figures.lower() != 'yes'):
            print("Figures will not be shown and only saved.")    

        self.training_setups: dict = {}
        # New instruction learning
        # New for when we have predicted goal states for training
        if predicted_path:
            self.instruction_path = predicted_path
            if instruction_path:
                self.testing_path = instruction_path
            else:
                self.testing_path = predicted_path
        else:
            self.instruction_path = instruction_path
            self.testing_path = False   
            # New - instruction chaining
            # - i.e. when learning the env should not reset to original start position
            # - instead, next instruction starts from where previous ended
            self.instruction_chain = instruction_chain
            if instruction_chain_how == 'None':
                self.instruction_chain_how = 'Random'
            else:
                self.instruction_chain_how = instruction_chain_how

            # new - store agents cross training repeats for completing the same start-end goal
            self.trained_agents: dict = {}
            self.num_training_seeds = self.setup_info['number_training_seeds']
            # new - config input defines the re-use of trained agents for testing: 'best' or 'all'
            self.test_agent_type = self.setup_info['test_agent_type']
            self.analysis = Evaluation(window_size=window_size)
            # Defines the number of episodes used for sub-instructions
            self.instruction_episode_ratio = instruction_episode_ratio

        # Instruction Knowledge from Search
        self.known_instructions = list(self.instruction_path.keys())
        self.known_instructions_dict = {}
        # Extract sub-goal completion for each instruction based on search results
        agent_adapter_i = None
        for instr in self.known_instructions:
            start = instr.split("---")[0]
            end = instr.split("---")[1]
            for n, agent_type in enumerate(self.setup_info['agent_select']):
                adapter = self.setup_info["adapter_select"][n]
                agent_adapter = agent_type+'_'+adapter
                if agent_adapter not in self.known_instructions_dict:
                    self.known_instructions_dict[agent_adapter] = {}
                
                if agent_adapter in self.instruction_path[instr]:
                    count = self.instruction_path[instr][agent_adapter]['count']
                    if start not in self.known_instructions_dict[agent_adapter]:
                        self.known_instructions_dict[agent_adapter][start] = {}
                        if end not in self.known_instructions_dict[agent_adapter][start]:
                            self.known_instructions_dict[agent_adapter][start][end] = count
                else:
                    # Supplement alternative search agent for this
                    # - we need it to match agent_adapter lookup for later calls so simply copies the search knowledge
                    # Search agent+adapter is now independent from optimization agent, 
                    #  - will default to match 
                    #  - but if optimization agent not seen in search then alternative must be used
                    agent_adapter_list = {}
                    i = 0
                    for item in self.instruction_path[instr]:
                        agent_adapter_list[str(i)] = item
                        i+=1
                    
                    if (i>1)&(agent_adapter_i is None):
                        print("\n Agent + Adapter not used in instruction search, please select the search agent:")
                        print(agent_adapter_list)
                        agent_adapter_i = input("\t - Select the id number of the default search agent+adapter you wish to use:    ")
                    elif (i>1)&(agent_adapter_i is not None):
                        agent_adapter_i = agent_adapter_i
                    else:
                        print("Only one agent used in instruction search, defaulting to this.")
                        print(agent_adapter_list)
                        agent_adapter_i = '0'
                    agent_adapter_copy = agent_adapter_list[agent_adapter_i]
                    # Copy knowledge of chosen search agent+adapter
                    # - Instruction path to define sub_goal list
                    self.instruction_path[instr][agent_adapter] = self.instruction_path[instr][agent_adapter_copy].copy()
                    # - Known instructions dict to define meta-MDP planner
                    count = self.instruction_path[instr][agent_adapter_copy]['count']
                    if start not in self.known_instructions_dict[agent_adapter]:
                        self.known_instructions_dict[agent_adapter][start] = {}
                        if end not in self.known_instructions_dict[agent_adapter][start]:
                            self.known_instructions_dict[agent_adapter][start][end] = count
                    
        print("-----")
        print("Known human instruction inputs. ")
        print(self.known_instructions_dict)
        print(" - ")
        self.total_num_instructions = 0
        for instr in self.instruction_path:
            print("\n \t - ", instr, " -> ", list(self.instruction_path[instr].keys()))
            self.total_num_instructions+=1
        
        
                
    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Added gym based agents as selection
        if self.is_gym_agent:
            env_start = self.start_obs 
            start = str(env_start).split(".")[0]
            goal = start + "---" + "GOAL"
            print("Long-term Goal: ", goal)
            # ---- 
            # New reward signal passed to engine to generate gym reward
            reward_signal = {}
            while True:
                max_count = 0
                # Go through path and extract a a reward signal for each sub-instruction
                # If search cant use agent, it will default to Qlearntab
                agent_adapter = (self.setup_info["agent_select"][0]+'_'+self.setup_info["adapter_select"][0])
                if agent_adapter not in self.known_instructions_dict:
                    agent_adapter = "Qlearntab"+"_"+self.setup_info["adapter_select"][0]
                if start in self.known_instructions_dict[agent_adapter]:
                    for end in self.known_instructions_dict[agent_adapter][start]:
                        if self.known_instructions_dict[agent_adapter][start][end] > max_count:
                            max_count = self.known_instructions_dict[agent_adapter][start][end]
                            instr = start + "---" + end
                            print("Sub-instr: ", instr) 

                    sub_goal = self.instruction_path[instr][agent_adapter]['sub_goal']
                    # Get reward signal for each sub-goal
                    # - Sub-goals are lists of all matching env labels
                    for sg in sub_goal:
                        reward_signal[sg] = self.sub_goal_reward#*np.round(1/i, 4) # e.g. r=1 --> 1/2, 1/3, 1/4, ..
                    # ---
                    start = end
                    prior_instr = instr
                else:
                    break     
            # Apply signal using dict:= {obs:reward, obs:reward, ...}
            print("GYM REWARD SIGNAL: ", reward_signal )
            self.gym_exp.reward_signal = reward_signal
            self.training_setups = self.gym_exp.train()   
                       
        else:
            for n, agent_type in enumerate(self.setup_info['agent_select']):
                # We are adding then overriding some inputs from general configs for experimental setups
                train_setup_info = self.setup_info.copy()
                # TODO: fix experience sampling
                if train_setup_info['experience_sample_batch_ratio']>0:
                    print("NOTE - Experience Sampling feature not currently implemented and will not be used")
                    train_setup_info['experience_sample_batch_ratio'] = 0
                # ----- State Adapter Choice
                adapter = train_setup_info["adapter_select"][n]
                # ----- Agent parameters
                agent_parameters = train_setup_info["agent_parameters"][agent_type]
                train_setup_info['agent_type'] = agent_type
                train_setup_info['agent_name'] = str(agent_type) + '_' + str(adapter) + '_' + str(agent_parameters)
                train_setup_info['adapter_select'] = adapter
                # ----- Sub-Goal
                # - If we have setup dict to include agent_adapter specific location of sub-goals
                #   i.e. {instr:{env_code:{agent_adapter:{sub_goal:'ENV_CODE', sim_score:0.8}}, action_cap:5}}
                #   Otherwise is standard user defined input {instr:{env_code:'ENV_CODE', action_cap:5}}
                # -----
                # Repeat training
                train_setup_info['train'] = True
                number_training_episodes = train_setup_info['number_training_episodes']
                number_training_repeats = self.ExperimentConfig["number_training_repeats"]
                print("Training Agent " + str(agent_type) + " for " + str(number_training_repeats) + " repeats")
                if str(agent_type) + '_' + str(adapter) not in self.trained_agents:
                    self.trained_agents[str(agent_type) + '_' + str(adapter)] = {}

                seed_recall = {}
                seed_results_connection = {}
                if self.num_training_seeds <1:
                    self.num_training_seeds = 1
                for seed_num in range(0,self.num_training_seeds):
                    if self.num_training_seeds > 1:
                        print("------")
                        print("- Seed Num: ", seed_num)
                    # -------------------------------------------------------------------------------
                    # Initialise Environment
                    # Environment now init here and called directly in experimental setup loop
                    # - Observed states, experience sampling passed over seeds but not training repeats
                    if seed_num==0:
                        train_setup_info['training_results'] = False
                        train_setup_info['observed_states'] = False
                        train_setup_info['experience_sampling'] = False
                    else:
                        train_setup_info['training_results'] = False
                        train_setup_info['observed_states'] = observed_states_stored.copy()
                        train_setup_info['experience_sampling'] = experience_sampling_stored.copy()
                    # --- 
                    setup_num:int = 0
                    temp_agent_store = {}
                    for training_repeat in range(1,(number_training_repeats+1)):
                        if number_training_repeats > 1:
                            print("------")
                            print("- Repeat Num: ", training_repeat)
                        setup_num+=1
                        
                        # ----- init agent
                        player = AGENT_TYPES[agent_type](**agent_parameters)
                        train_setup_info['agent'] = player
                        
                        # init live environment
                        train_setup_info['live_env'] = True
                        live_env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=train_setup_info)
                        # ----------------
                        #  Start obs reset between seeds but fixed for repeat
                        if training_repeat > 1:
                            live_env.start_obs = env_start

                        env_start = live_env.start_obs
                        goal = str(env_start).split(".")[0] + "---" + "GOAL"
                        print("Long-term Goal: ", goal)
                        if goal in seed_recall:
                            setup_num = seed_recall[goal]
                        else:
                            seed_recall[goal] = 1
                        # Load results table from previous seeds to continue output graphs
                        if goal in seed_results_connection:
                            live_env.results.load(seed_results_connection[goal])
                    
                        # - Results save dir -> will override for same goal if seen in later seed
                        if self.num_training_seeds > 1:
                            agent_save_dir = self.save_dir+'/'+agent_type+'_'+adapter+'__training_results_'+str(goal)+'_'+str(setup_num) 
                        else:
                            agent_save_dir = self.save_dir+'/'+agent_type+'_'+adapter+'__training_results_'+str(setup_num)
                        if not os.path.exists(agent_save_dir):
                            os.mkdir(agent_save_dir)
                        
                        if train_setup_info['experience_sample_batch_ratio']>0:
                            live_sample_batch_size = int(number_training_episodes*train_setup_info['experience_sample_batch_ratio'])
                            live_sample_batch_count = int(1/train_setup_info['experience_sample_batch_ratio'])
                            # Train on Live system for limited number of total episodes   
                            live_env.num_train_episodes = live_sample_batch_size
                            print("-- Training with Simulated Batches, ", live_sample_batch_count, " total...")
                            # init simulated environment
                            train_setup_info['live_env'] = False
                            simulated_env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=train_setup_info)
                            simulated_env.start_obs = env_start

                            sample_batch_agent_store = {} # Need to continue learning across batches
                            sample_batch_results_store = {} # Pass over results table to next batch for the same instr TODO not working
                            for live_sample_batch in range(0, live_sample_batch_count-1):
                                print("--- Live Interaction Batch Num", live_sample_batch+1)
                                # ---- 
                                # Train for sub instr plan
                                start = str(env_start).split(".")[0]
                                i=0
                                while True:
                                    # Only allow insutrction up until total limi
                                    # - Prevents it being given more episodes than flat
                                    # - Prevents cyclic instruction paths
                                    if int(number_training_episodes*self.instruction_episode_ratio)<=(number_training_episodes-total_instr_episodes):
                                        # Override with trained agent if goal seen previously
                                        if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                            live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()
                                            # Reset exploration parameter between seeds so to not get 'trapped'
                                            live_env.agent.exploration_parameter_reset()
                                        break
                                    
                                    elif start in self.known_instructions_dict[(agent_type+'_'+adapter)]:
                                        i+=1
                                        max_count = 0
                                        for end in self.known_instructions_dict[(agent_type+'_'+adapter)][start]:
                                            if self.known_instructions_dict[(agent_type+'_'+adapter)][start][end] > max_count:
                                                max_count = self.known_instructions_dict[(agent_type+'_'+adapter)][start][end]
                                                instr = start + "---" + end
                                                print("Sub-instr: ", instr)
                                        # ---
                                        # Override trained agent with known instruction agent 
                                        if instr in sample_batch_agent_store:
                                            live_env.agent = sample_batch_agent_store[instr]
                                        elif instr in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                            live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][instr].clone()
                                        if instr in sample_batch_results_store:
                                            live_env.results.load(sample_batch_results_store[instr])
                                        
                                        # elif prior_instr:
                                        #     live_env.agent = live_env.agent.clone() # Start from agent from previous task
                                        # TODO: ADOPT AGENT OF MOST SIMILAR POLICY
                                        # ---
                                        sub_goal = self.instruction_path[instr][agent_type+'_'+adapter]['sub_goal']
                                        live_env.sub_goal = sub_goal
                                        live_env.agent.exploration_parameter_reset()
                                        training_results = live_env.episode_loop()
                                        sample_batch_agent_store[instr] = live_env.agent.clone()
                                        # Override episode numbers and store for next batch
                                        training_results['episode'] = training_results.index
                                        sample_batch_results_store[instr] = live_env.results.copy()
                                        # ---
                                        # Store instruction results
                                        instr_save_dir = agent_save_dir+'/'+str(i)+"-"+instr.replace(" ","").replace("/","_")
                                        if not os.path.exists(instr_save_dir):
                                            os.mkdir(instr_save_dir)

                                        # Produce training report with Analysis.py
                                        Return = self.analysis.train_report(training_results, instr_save_dir, self.show_figures)
                                        if instr not in temp_agent_store:
                                            temp_agent_store[instr] = {}
                                        temp_agent_store[instr][setup_num] = {'Return':Return,'agent':live_env.agent.clone()}
                                        # ---
                                        start = end
                                        live_env.results.reset() # Force reset so we don't get overlapping outputs
                                        #live_env.start_obs = sub_goal[0] # Set start position of env for next -> pick first sub-goal
                                    else:
                                        # if at least one sub-instr start with that for goal agent
                                        # if i > 1:
                                        #     live_env.agent = live_env.agent.clone() # Start from agent from previous task

                                        # if no instructions and first batch then adopt known agent
                                        # - Will then use this for next batches
                                        if i == 0 :
                                            if live_sample_batch == 0:
                                                # Override with trained agent if goal seen previously
                                                if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                                    live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()
                                        break
                                
                                # train for entire path 
                                #live_env.start_obs = env_start
                                live_env.sub_goal = None 
                                print("Goal: ", goal)
                                if goal in sample_batch_results_store:
                                    live_env.results.load(sample_batch_results_store[goal])
                                live_env.agent.exploration_parameter_reset()
                                training_results = live_env.episode_loop()
                                training_results['episode'] = training_results.index
                                sample_batch_results_store[goal] = live_env.results.copy()

                                # Train based on simulated exp             
                                simulated_env.num_train_episodes = number_training_episodes
                                # Connect live agent -> This should be linked so continues learning
                                simulated_env.agent = live_env.agent
                                simulated_env.episode_loop()
                            # Final batch doesn't require simulated exp after and we need result output
                            print("--- Final batch")
                            train_setup_info['live_env'] = True
                            train_setup_info['number_training_episodes'] = live_sample_batch_size
                            live_env.agent.exploration_parameter_reset()
                            training_results = live_env.episode_loop()

                            # Have to 'fix' output
                            training_results['episode'] = training_results.index
                            cumulative_r = 0
                            cumulative_r_lst = []
                            for r in training_results['episode_reward']:
                                cumulative_r+=r
                                cumulative_r_lst.append(cumulative_r)
                            training_results['cumulative_reward'] = cumulative_r_lst
                        else:
                            # ---- 
                            # Train for sub instr plan
                            # Set start position of env for next -> pick first sub-goal env label
                            start = str(env_start).split(".")[0]
                            i=0
                            total_instr_episodes = 0
                            instr_results = None
                            prior_instr = None
                            multi_sub_goal = {} # New multi-goal option -> needs to be defined in env
                            while True:
                                i+=1
                                if i > self.total_num_instructions:
                                    break
                                max_count = 0
                                # Only allow instruction up until total limit
                                # - Prevents it being given more episodes than flat
                                # - Prevents cyclic instruction paths
                                if int(number_training_episodes*self.instruction_episode_ratio)>=(number_training_episodes-total_instr_episodes):
                                    # Override with trained agent if goal seen previously
                                    if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                        live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()
                                        # Reset exploration parameter between seeds so to not get 'trapped'
                                        live_env.agent.exploration_parameter_reset()
                                    break
                                
                                elif start in self.known_instructions_dict[(agent_type+'_'+adapter)]:
                                    for end in self.known_instructions_dict[(agent_type+'_'+adapter)][start]:
                                        if self.known_instructions_dict[(agent_type+'_'+adapter)][start][end] > max_count:
                                            max_count = self.known_instructions_dict[(agent_type+'_'+adapter)][start][end]
                                            instr = start + "---" + end
                                            print("Sub-instr: ", instr)
                                            
                                    # Instructions use fewer episodes, lower bound to 10
                                    number_instr_episodes = int(number_training_episodes*self.instruction_episode_ratio)
                                    if number_instr_episodes<10:
                                        number_instr_episodes=10
                                    total_instr_episodes+=number_instr_episodes
                                    live_env.num_train_episodes = number_instr_episodes
                                    # ---
                                    # Override trained agent with known instruction agent
                                    if instr in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                        live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][instr].clone()
                                    # TODO: ADOPT AGENT OF MOST SIMILAR POLICY
                                    # ---
                                    # New: Allow env to start from prior instr end
                                    if self.instruction_chain:
                                        if prior_instr:
                                            # Select first env position from known sub-goal list
                                            if self.instruction_chain_how.lower() == 'first':
                                                env_sg_start = self.instruction_path[prior_instr][agent_type+'_'+adapter]['sub_goal'][0]
                                            elif self.instruction_chain_how.lower() == 'random':
                                                env_sg_start = random.choice(self.instruction_path[prior_instr][agent_type+'_'+adapter]['sub_goal'])
                                            elif self.instruction_chain_how.lower() == 'exact':
                                                try:
                                                    if live_env.sub_goal_end:
                                                        env_sg_start = live_env.sub_goal_end
                                                    else:
                                                        env_sg_start = random.choice(self.instruction_path[prior_instr][agent_type+'_'+adapter]['sub_goal'])
                                                except:
                                                    print("ERROR: To use the EXACT instruction chain, the environment must include the '.sub_goal_end' attribute.")
                                            elif (self.instruction_chain_how.lower() == 'continuous')|(self.instruction_chain_how.lower() == 'cont'):
                                                # Env start position is default (i.e. reset)
                                                env_sg_start = None
                                                # Define multi-sub-goals
                                                # - Scale prior instruction down based on current path by r = 1/x
                                                # - Reward override if same instr seen later to prevent cyclic loops that don't complete episode 
                                                if prior_instr not in multi_sub_goal:
                                                    multi_sub_goal[prior_instr] = {}
                                                    multi_sub_goal[prior_instr]['sub_goal'] = self.instruction_path[instr][agent_type+'_'+adapter]['sub_goal']
                                                multi_sub_goal[prior_instr]['reward_scale'] = np.round(1/i, 4) # e.g. r=1 --> 1/2, 1/3, 1/4, ...
                                                    
                                                try:
                                                    # This doesn't supercede .sub_goal so need both defined
                                                    live_env.multi_sub_goal = multi_sub_goal
                                                except:
                                                    print("ERROR: To use CONTINUOUS instruction chain, the environment must include the '.multi_sub_gial' attribute.")
                                            
                                            if env_sg_start:
                                                live_env.start_obs = env_sg_start
                                        
                                    sub_goal = self.instruction_path[instr][agent_type+'_'+adapter]['sub_goal']
                                    live_env.sub_goal = sub_goal
                                    live_env.agent.exploration_parameter_reset()
                                    if type(instr_results)==type(pd.DataFrame()):
                                        live_env.results.load(instr_results)
                                    training_results = live_env.episode_loop()
                                    training_results['episode'] = training_results.index
                                    # ---
                                    # Store instruction results
                                    instr_save_dir = agent_save_dir+'/'+str(i)+"-"+instr.replace(" ","").replace("/","_")
                                    if not os.path.exists(instr_save_dir):
                                        os.mkdir(instr_save_dir)

                                    # Produce training report with Analysis.py
                                    Return = self.analysis.train_report(training_results, instr_save_dir, self.show_figures)
                                    if instr not in temp_agent_store:
                                        temp_agent_store[instr] = {}
                                    temp_agent_store[instr][setup_num] = {'Return':Return,'agent':live_env.agent.clone()}
                                    # ---
                                    start = end
                                    prior_instr = instr
                                    # New: Dont reset results for each sub-instr so we show the training results with this included
                                    #live_env.results.reset() # Force reset so we don't get overlapping outputs
                                    instr_results =  live_env.results.copy()
                                else:
                                    # If no instructions then train for full goal
                                    if i == 1:
                                        # Override with trained agent if goal seen previously
                                        if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                            live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()
                                            # Reset exploration parameter between seeds so to not get 'trapped'
                                            live_env.agent.exploration_parameter_reset()
                                    break
                            # train for entire path 
                            if self.instruction_chain:
                                live_env.start_obs = env_start
                            # Number of episodes used reduced by those used for instructions (lower bounded)
                            if (number_training_episodes-total_instr_episodes)<int(number_training_episodes*self.instruction_episode_ratio):
                                if int(number_training_episodes*self.instruction_episode_ratio) < 10:
                                    live_env.num_train_episodes = 10
                                else:
                                    live_env.num_train_episodes = int(number_training_episodes*self.instruction_episode_ratio)
                            else:
                                live_env.num_train_episodes = number_training_episodes - total_instr_episodes
                            # Remove sub-goal
                            live_env.sub_goal = None
                            print("Goal: ", goal)
                            # Add instruction training to output chart
                            if type(instr_results)==type(pd.DataFrame()):
                                live_env.results.load(instr_results)
                            
                            training_results = live_env.episode_loop()
                            training_results['episode'] = training_results.index
                            
                        # Opponent now defined in local setup.py
                        # ----- Log training setup      
                        training_results.insert(loc=0, column='Repeat', value=setup_num)                    
                        # Produce training report with Analysis.py
                        Return = self.analysis.train_report(training_results, agent_save_dir, self.show_figures)
                        if goal not in temp_agent_store:
                            temp_agent_store[goal] = {}
                        temp_agent_store[goal][setup_num] = {'Return':Return,'agent':live_env.agent.clone()}
                                                                
                        # Extract trained agent from env and stored for re-call
                        # - Observed states and experience sampling from best repeat stored for next seed
                        if training_repeat == 1:
                            max_Return = Return
                            training_results_stored =  live_env.results.copy()
                            observed_states_stored = live_env.elsciRL.observed_states
                            experience_sampling_stored = live_env.elsciRL.experience_sampling
                        if Return > max_Return:
                            max_Return = Return
                            training_results_stored =  live_env.results.copy()
                            observed_states_stored = live_env.elsciRL.observed_states
                            experience_sampling_stored = live_env.elsciRL.experience_sampling
                        
                        seed_recall[goal] = seed_recall[goal] + 1
                    seed_results_connection[goal] = training_results_stored

                    for instr in temp_agent_store:
                        start_repeat_num = list(temp_agent_store[instr].keys())[0]
                        end_repeat_num = list(temp_agent_store[instr].keys())[-1]

                        if self.test_agent_type.lower() == 'best':
                            # Only save the best agent from repeated training
                            print("---------")
                            print("Selecting best agent from training repeats.")
                            best_return = temp_agent_store[instr][start_repeat_num]['Return']
                            best_agent = temp_agent_store[instr][start_repeat_num]['agent']
                            for repeat in range(start_repeat_num+1,end_repeat_num+1):
                                if temp_agent_store[instr][repeat]['Return']>best_return:
                                    best_return = temp_agent_store[instr][repeat]['Return']
                                    best_agent = temp_agent_store[instr][repeat]['agent']
                                    print(best_return)
                    
                            if instr not in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                self.trained_agents[str(agent_type) + '_' + str(adapter)][instr] = {}
                            self.trained_agents[str(agent_type) + '_' + str(adapter)][instr] = best_agent      
                        elif self.test_agent_type.lower() == 'all':
                            all_agents = []
                            for repeat in range(start_repeat_num,end_repeat_num+1):
                                agent = temp_agent_store[instr][repeat]['agent']
                                all_agents.append(agent)
                                
                            if instr not in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                self.trained_agents[str(agent_type) + '_' + str(adapter)][instr] = {}
                            self.trained_agents[str(agent_type) + '_' + str(adapter)][instr] = all_agents

                                            
                # Store last train_setup_info as collection of observed states and experience sampling
                self.training_setups['Training_Setup_'+str(agent_type) + '_' + str(adapter)] = train_setup_info
            if (number_training_repeats>1)|(self.num_training_seeds):
                self.analysis.training_variance_report(self.save_dir, self.show_figures)
                    
        #json.dump(self.training_setups) # TODO: Won't currently serialize this output to a json file
        return self.training_setups

    # TESTING PLAY
    def test(self, training_setups:str=None):
        # Added gym based agents as selection
        # - Strip reward signal from instructions for testing
        if self.is_gym_agent:
            self.gym_exp.reward_signal = None
            self.gym_exp.test()
        else:
            # Override input training setups with previously saved 
            if training_setups is None:
                training_setups = self.training_setups
            else:
                json.load(training_setups)

            for training_key in list(training_setups.keys()):    
                test_setup_info = training_setups[training_key]
                test_setup_info['train'] = False # Testing Phase
                print("----------")
                print("Testing results for trained agents in saved setup configuration:")
                number_training_repeats = test_setup_info['number_test_repeats']
                agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']
                
                # Only use the trained agent with best return
                if self.test_agent_type.lower()=='best':
                    for testing_repeat in range(0, test_setup_info['number_test_repeats']):  
                        # Re-init env for testing
                        env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=test_setup_info)
                        # ---
                        start_obs = env.start_obs
                        goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                        print("Flat agent Goal: ", goal)
                        # Override with trained agent if goal seen previously
                        if goal in self.trained_agents[test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                            print("Trained agent available for testing.")
                            env.agent = self.trained_agents[test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
                        else:
                            print("NO agent available for testing position.")
                        env.agent.epsilon = 0 # Remove random actions
                        # ---
                        # Testing generally is the agents replaying on the testing ENV
                        testing_results = env.episode_loop() 
                        test_save_dir = (self.save_dir+'/'+agent_adapter+'__testing_results_'+str(goal).split("/")[0]+"_"+str(testing_repeat))
                        if not os.path.exists(test_save_dir):
                            os.mkdir(test_save_dir)
                        # Produce training report with Analysis.py
                        Return = self.analysis.test_report(testing_results, test_save_dir, self.show_figures)
                        
                # Re-apply all trained agents with fixed policy
                elif self.test_agent_type.lower()=='all':
                    # All trained agents are used:
                    # - Repeats can be used to vary start position
                    # - But assumed environment is deterministic otherwise
                    # Re-init env for testing
                    for testing_repeat in range(0, test_setup_info['number_test_repeats']):
                        env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=test_setup_info)
                        # ---
                        start_obs = env.start_obs
                        goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                        print("Flat agent Goal: ", goal)
                        # Override with trained agent if goal seen previously
                        if goal in self.trained_agents[test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                            print("Trained agents available for testing.")
                            all_agents = self.trained_agents[test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
                        else:
                            print("NO agent available for testing position.")
                        
                        for ag,agent in enumerate(all_agents):
                            env.results.reset() # Reset results table for each agent
                            env.start_obs = start_obs
                            env.agent = agent
                            env.agent.epsilon = 0 # Remove random actions
                            agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']
                            # ---
                            # Testing generally is the agents replaying on the testing ENV
                            testing_results = env.episode_loop() 
                            test_save_dir = (self.save_dir+'/'+agent_adapter+'__testing_results_'+str(goal).split("/")[0]+"_"+"agent-"+str(ag)+"-repeat-"+str(testing_repeat))
                            if not os.path.exists(test_save_dir):
                                os.mkdir(test_save_dir)
                            # Produce training report with Analysis.py
                            Return = self.analysis.test_report(testing_results, test_save_dir, self.show_figures)

            # Path is the experiment save dir + the final instruction
            if (number_training_repeats>1)|(self.test_agent_type.lower()=='all'):
                self.analysis.testing_variance_report(self.save_dir, self.show_figures)


    def render_results(self, training_setups:str=None):
        """Apply fixed policy to render current decision making for limited number of episodes."""
        # Override input training setups with previously saved 
        if training_setups is None:
            training_setups = self.training_setups
        else:
            json.load(training_setups)

        for training_key in list(training_setups.keys()):    
            test_setup_info = training_setups[training_key]
            test_setup_info['train'] = False # Testing Phase
            print("----------")
            print("Rendering trained agent's policy:")
            agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']


            # Only use the trained agent with best return
            if self.test_agent_type.lower()=='best':
                for testing_repeat in range(0, test_setup_info['number_test_repeats']):  
                    # Re-init env for testing
                    env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=test_setup_info)
                    # ---
                    start_obs = env.start_obs
                    goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                    print("Flat agent Goal: ", goal)
                    # Override with trained agent if goal seen previously
                    if goal in self.trained_agents[test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                        print("Trained agent available for testing.")
                        env.agent = self.trained_agents[test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
                    else:
                        print("NO agent available for testing position.")
                    env.agent.epsilon = 0 # Remove random actions
                    # ---
                    # Testing generally is the agents replaying on the testing ENV
                    render_save_dir = self.save_dir+'/render_results'
                    if not os.path.exists(render_save_dir):
                        os.mkdir(render_save_dir)
                    render_results = env.episode_loop(render=True, render_save_dir=render_save_dir)
                    # Produce training report with Analysis.py
                    #Return = self.analysis.test_report(render_results, render_save_dir, self.show_figures)
                    
            # Re-apply all trained agents with fixed policy
            elif self.test_agent_type.lower()=='all':
                # All trained agents are used:
                # - Repeats can be used to vary start position
                # - But assumed environment is deterministic otherwise
                # Re-init env for testing
                for testing_repeat in range(0, test_setup_info['number_test_repeats']):
                    env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=test_setup_info)
                    # ---
                    start_obs = env.start_obs
                    goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                    print("Flat agent Goal: ", goal)
                    # Override with trained agent if goal seen previously
                    if goal in self.trained_agents[test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                        print("Trained agents available for testing.")
                        all_agents = self.trained_agents[test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
                    else:
                        print("NO agent available for testing position.")
                    
                    for ag,agent in enumerate(all_agents):
                        env.results.reset() # Reset results table for each agent
                        env.start_obs = start_obs
                        env.agent = agent
                        # only run once
                        env.num_train_repeat = 1
                        env.num_test_repeat = 1
                        env.agent.epsilon = 0 # Remove random actions
                        agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']
                        # ---
                        # Testing generally is the agents replaying on the testing ENV
                        render_save_dir = self.save_dir+'/render_results'
                        if not os.path.exists(render_save_dir):
                            os.mkdir(render_save_dir)
                        render_results = env.episode_loop(render=True, render_save_dir=render_save_dir)
                        # Produce training report with Analysis.py
                        #Return = self.analysis.test_report(render_results, render_save_dir, self.show_figures)

