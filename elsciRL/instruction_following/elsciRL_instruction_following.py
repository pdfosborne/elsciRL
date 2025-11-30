import os
import pandas as pd
import numpy as np
import json
import random

# ------ Interaction Protocol -----------------------------------
from elsciRL.interaction_loops.standard import StandardInteractionLoop
from elsciRL.interaction_loops.policy_gradient import PolicyGradientInteractionLoop
# ------ Experiment Import --------------------------------------
from elsciRL.evaluation.standard_report import Evaluation

from elsciRL.experiments.experiment_utils.agent_factory import AgentFactory
from elsciRL.experiments.experiment_utils.config_utils import ensure_dir, merge_configs
from elsciRL.experiments.experiment_utils.env_manager import EnvManager
from elsciRL.experiments.experiment_utils.result_manager import ResultManager
from elsciRL.experiments.training_procedures.default_exp_training import run_training_loop

# ------ Agent Imports -----------------------------------------

# Universal Agents
from elsciRL.agents.agent_abstract import Agent, QLearningAgent
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.DQN import DQNAgent
# ------ Gym Experiement ----------------------------------------
from elsciRL.experiments.GymExperiment import GymExperiment
from elsciRL.experiments.policy_gradient import PolicyGradientExperiment
from elsciRL.experiments.experiment_utils.render_current_results import render_current_result
# ------ LLM Agents ---------------------------------------------
from elsciRL.agents.LLM_agents.ollama_agent import LLMAgent as OllamaAgent
from elsciRL.agents.clean_rl.ppo import CleanRLPPO as PPOAgent
# ---------------------------------------------------------------
# TODO: COMPLETELY REWRITE THIS FILE TO USE THE NEW EXPERIMENT FRAMEWORK


# TODO: Enable any number of the same agent types with varying parameters
AGENT_TYPES = {
    "Qlearntab": TableQLearningAgent,
    "DQN": DQNAgent,
    "Random": random,
    "LLM_Ollama": OllamaAgent,
    "PPO": PPOAgent,
    
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
                 instruction_chain:bool=False, instruction_chain_how:str='None',
                 training_render:bool=False, training_render_save_dir:str=None):
        self.ExperimentConfig = Config
        self.LocalConfig = LocalConfig

        try:
            self.setup_info = self.ExperimentConfig['data'] | self.LocalConfig['data']
        except:
            self.setup_info = self.ExperimentConfig | self.LocalConfig

        # If in experiment config make sure it pulls from this and not local config
        if 'adapter_input_dict' in self.ExperimentConfig:
            self.setup_info['adapter_input_dict'] = self.ExperimentConfig['adapter_input_dict']
        else:
            selected_adapters = list(Adapters.keys())
            selected_agents = self.ExperimentConfig['agent_select']
            agent_adapter_dict = {agent_name: list(selected_adapters) for agent_name in selected_agents} if selected_agents else {}
            self.ExperimentConfig['adapter_input_dict'] = agent_adapter_dict
            self.setup_info['adapter_input_dict'] = agent_adapter_dict
            
        # Environment setup
        if isinstance(Engine, dict):
            print("\n Multiple Engines detected, will compare results across engines...")
            self.engine_comparison = True
            self.engine_list = Engine
        else:
            self.engine_comparison = False
            self.engine_list = {'DefaultEng':Engine}
            self.engine = Engine
        self.adapters = Adapters
        self.env = StandardInteractionLoop
        self.env_manager = EnvManager(StandardInteractionLoop, Adapters)
        
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

        # Run render results during training to show progress
        self.training_render = training_render
        self.training_render_save_dir = training_render_save_dir
        # New: Flag for if using gym agents to optimize instead
        # - Generates reward signal from instructions that is passed to gym eng translator
        # - init gym experiment if any gym agent selected
        self.is_gym_agent = {}
        for n,agent_type in enumerate(self.setup_info['agent_select']):
            # PPO handled separately via PolicyGradientExperiment
            if agent_type == "PPO":
                self.is_gym_agent[agent_type] = False
            elif agent_type.split('_')[0] == "SB3":
                self.is_gym_agent[agent_type] = True
                self.sub_goal_reward = self.setup_info['reward_signal'][0]
                self.gym_exp = GymExperiment(Config=self.ExperimentConfig, ProblemConfig=self.LocalConfig, 
                        Engine=self.engine, Adapters=self.adapters,
                        save_dir=self.save_dir, show_figures = 'No', window_size=0.1)
                # Get start position to start instr chains
                train_setup_info = self.setup_info.copy()
                agent_type = "Qlearntab" # Force agent to Qlearntab for compatibility
                # Add Qlearntab if not existed and select first adapter
                if "Qlearntab" not in train_setup_info["adapter_input_dict"]:
                    first_agent_type = list(train_setup_info["adapter_input_dict"].keys())[0]
                    first_adapter = train_setup_info["adapter_input_dict"][first_agent_type][0]
                    train_setup_info["adapter_input_dict"]["Qlearntab"] = [first_adapter]
                adapter = train_setup_info["adapter_input_dict"][agent_type][0]
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
                live_env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=train_setup_info)
                self.start_obs = live_env.start_obs
            else:
                self.is_gym_agent[agent_type] = False

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
        # - REMOVE FEEDBACK - TODO IMPROVE THIS SO INSTR AREN'T IN SAME LEVEL
        for key in list(self.instruction_path.keys()):
            if '---' not in key:
                self.instruction_path.pop(key, None)
        ignore_data = ["instr_description", "feedback_count", "feedback_plot", "user_feedback_count", "user_input"]
        # Remove non-instr data from path
        cleaned_instruction_path = {}
        for instr in self.instruction_path:
            for ignore in ignore_data:
                if ignore not in self.instruction_path[instr]:
                    cleaned_instruction_path[instr] = self.instruction_path[instr]
        self.instruction_path = cleaned_instruction_path
        self.known_instructions = list(self.instruction_path.keys())
        self.known_instructions_dict = {}
        # Extract sub-goal completion for each instruction based on search results
        agent_adapter_i = None
        for instr in self.known_instructions:
            # Extract start and end from instruction
            start = instr.split("---")[0]
            end = instr.split("---")[1]
            for n, agent_type in enumerate(self.setup_info['agent_select']):
                print("SELECTED AGENTS",self.setup_info['agent_select'])
                print("ADAPTER LIST",self.setup_info["adapter_input_dict"])
                adapter_inputs = self.setup_info["adapter_input_dict"][agent_type]
                for adapter in adapter_inputs:
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
                            if item.split("_")[0] in list(AGENT_TYPES.keys()):
                                agent_adapter_list[str(i)] = item
                                i+=1
                        
                        if (i>1)&(agent_adapter_i is None):
                            # TODO: DEFAULTING TO FIRST OPTION FOR NOW
                            agent_adapter_i = '0'
                            # print("\n Agent + Adapter not used in instruction search, please select the search agent:")
                            # print(agent_adapter_list)
                            # agent_adapter_i = input("\t - Select the id number of the default search agent+adapter you wish to use:    ")
                            # if agent_adapter_i == '':
                            #     agent_adapter_i = '0'
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

        # Store policy gradient experiments for later testing/rendering
        self.policy_gradient_experiments = {}

        for n, agent_type in enumerate(self.setup_info['agent_select']):
            # Check if this is a Policy Gradient agent (like PPO)
            # If so, use PolicyGradientExperiment instead of standard instruction following
            if agent_type == "PPO":
                print(f"\n{'='*60}")
                print(f"Detected Policy Gradient agent: {agent_type}")
                print(f"Using PolicyGradientExperiment for training...")
                print(f"{'='*60}\n")
                
                # Prepare configs for PolicyGradientExperiment
                # Need to ensure LocalConfig has all necessary parameters
                local_config_for_pg = self.LocalConfig.copy()
                if 'data' not in local_config_for_pg:
                    local_config_for_pg = {'data': local_config_for_pg}
                
                experiment_config_for_pg = self.ExperimentConfig.copy()
                if 'data' not in experiment_config_for_pg:
                    experiment_config_for_pg = {'data': experiment_config_for_pg}
                
                # Create PolicyGradientExperiment instance
                pg_exp = PolicyGradientExperiment(
                    Config=experiment_config_for_pg,
                    ProblemConfig=local_config_for_pg,
                    Engine=self.engine,
                    Adapters=self.adapters,
                    save_dir=self.save_dir,
                    show_figures=self.show_figures,
                    window_size=0.1,  # Use same window size as instruction following
                    create_subdirectory=False  # Don't create extra subdirectory
                )
                
                # Run training using policy gradient experiment
                pg_training_setups = pg_exp.train()
                
                # Store the PolicyGradientExperiment for testing/rendering
                self.policy_gradient_experiments[agent_type] = pg_exp
                
                # Store the trained agents from policy gradient experiment
                for pg_agent_key, pg_agent in pg_exp.trained_agents.items():
                    # Store in the format expected by instruction following
                    agent_adapter = agent_type + '_' + self.setup_info["adapter_input_dict"][agent_type][0]
                    if agent_adapter not in self.trained_agents:
                        self.trained_agents[agent_adapter] = {}
                    self.trained_agents[agent_adapter] = pg_agent
                
                # Store training setup
                self.training_setups[f'Training_Setup_{agent_type}'] = pg_training_setups
                
                print(f"\n{'='*60}")
                print(f"Policy Gradient training completed for {agent_type}")
                print(f"{'='*60}\n")
                
                # ====================================================================
                # FLAT EXPERIMENT for PPO (baseline comparison without instructions)
                # Train the same agent without instruction following for comparison
                # ====================================================================
                print(f"\n{'='*60}")
                print(f"Running FLAT experiment for {agent_type} (baseline comparison)")
                print(f"Training without instruction following...")
                print(f"{'='*60}\n")
                
                # Create a separate flat experiment
                # Use parent directory for compatibility with variance analysis
                # The variance analysis expects all experiment folders at the same level
                # self.save_dir is app_save_dir/instr_key/Instr_Experiment
                # We need to go up 2 levels to get to app_save_dir
                grandparent_save_dir = os.path.dirname(os.path.dirname(self.save_dir))
                flat_save_dir = os.path.join(grandparent_save_dir, 'no-instr', 'PolicyGradient_Experiment')
                if not os.path.exists(flat_save_dir):
                    os.makedirs(flat_save_dir)
                    
                pg_exp_flat = PolicyGradientExperiment(
                    Config=experiment_config_for_pg,
                    ProblemConfig=local_config_for_pg,
                    Engine=self.engine,
                    Adapters=self.adapters,
                    save_dir=flat_save_dir,
                    show_figures=self.show_figures,
                    window_size=0.1,
                    create_subdirectory=False
                )
                
                # Run flat training
                flat_training_setups = pg_exp_flat.train()
                
                # Store flat experiment for testing/rendering
                flat_agent_key = f'{agent_type}_Flat'
                self.policy_gradient_experiments[flat_agent_key] = pg_exp_flat
                
                # Store flat trained agents
                for pg_agent_key, pg_agent in pg_exp_flat.trained_agents.items():
                    agent_adapter_flat = agent_type + '_Flat_' + self.setup_info["adapter_input_dict"][agent_type][0]
                    if agent_adapter_flat not in self.trained_agents:
                        self.trained_agents[agent_adapter_flat] = {}
                    self.trained_agents[agent_adapter_flat] = pg_agent
                
                # Store flat training setup
                self.training_setups[f'Training_Setup_{agent_type}_Flat'] = flat_training_setups
                
                print(f"\n{'='*60}")
                print(f"FLAT experiment completed for {agent_type}")
                print(f"Results saved to: {flat_save_dir}")
                print(f"{'='*60}\n")
                
                # Continue to next agent
                continue
            
            # Added gym based agents as selection
            is_gym_agent = self.is_gym_agent[agent_type]
            if is_gym_agent:
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
                    agent_adapter = (self.setup_info["agent_select"][n]+'_'+self.setup_info["adapter_input_dict"][self.setup_info["agent_select"][n]][0])
                    if agent_adapter not in self.known_instructions_dict:
                        agent_adapter = "Qlearntab"+"_"+self.setup_info["adapter_input_dict"]["Qlearntab"][0]
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
                print("GYM REWARD SIGNAL: ", reward_signal)
                self.gym_exp.reward_signal = reward_signal
                # --- GYM EXPERIMENT TRAINING
                train_setup_info = self.setup_info.copy()
                for adapter in train_setup_info["adapter_input_dict"][agent_type]:
                    self.gym_exp.setup_info['agent_select'] = [agent_type] 
                    self.training_setups[agent_adapter] = self.gym_exp.train() 
            else:
                # We are adding then overriding some inputs from general configs for experimental setups
                train_setup_info = self.setup_info.copy()
                # ----- State Adapter Choice
                for adapter in train_setup_info["adapter_input_dict"][agent_type]:
                    # ----- Agent parameters
                    agent_parameters = train_setup_info["agent_parameters"][agent_type]
                    train_setup_info['agent_type'] = agent_type
                    train_setup_info['agent_name'] = str(agent_type) + '_' + str(adapter) + '_' + str(agent_parameters)
                    train_setup_info['adapter_select'] = str(adapter)
                    # ----- Neural Agent Setup
                    # Get the input dim from the adapter or the encoder's output dim
                    if agent_type == "DQN":
                        adapter_sample = self.adapters[adapter](setup_info=self.setup_info)
                        # Set input_size from adapter
                        try:
                            input_size = adapter_sample.input_dim
                            print(f"Using input_dim from adapter {adapter}: {input_size}")
                        except Exception:
                            try:
                                input_size = adapter_sample.encoder.output_dim
                                print(f"Using encoder output_dim from encoder {adapter_sample.encoder}: {input_size}")
                            except Exception:
                                try:
                                    input_size = adapter_sample.LLM_adapter.encoder.output_dim
                                    print(f"Using LLM_adapter encoder output_dim from LLM adapter {adapter_sample.LLM_adapter}: {input_size}")
                                except Exception:
                                    print(f"Adapter {adapter} does not have input_dim specified.")
                                    raise ValueError(f"No input dim size found in adapter: {adapter}")

                        engine_sample = self.engine(local_setup_info=self.setup_info)
                        try:
                            output_size = engine_sample.output_size
                        except Exception:
                            try:
                                output_size = engine_sample.output_dim
                            except Exception:
                                try:
                                    output_size = engine_sample.output_dim_size
                                except Exception:
                                    print(f"Engine {engine_sample} does not contain output dim size for DQN agent, using default 1,000.")
                                    output_size = 1000
                        # Order must match DQN input
                        temp_dict = {'input_size': input_size, 'output_size': output_size}
                        temp_dict.update(agent_parameters)
                        agent_parameters = temp_dict.copy()
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
                        # - Observed states passed over seeds but not training repeats
                        if seed_num==0:
                            train_setup_info['training_results'] = False
                            train_setup_info['observed_states'] = False
                        else:
                            train_setup_info['training_results'] = False
                            train_setup_info['observed_states'] = observed_states_stored.copy()
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
                                    live_env.number_episodes = number_instr_episodes
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
                                    live_env.number_episodes = 10
                                else:
                                    live_env.number_episodes = int(number_training_episodes*self.instruction_episode_ratio)
                            else:
                                live_env.number_episodes = number_training_episodes - total_instr_episodes
                            # Remove sub-goal
                            live_env.sub_goal = None
                            print("Goal: ", goal)
                            # Add instruction training to output chart
                            if type(instr_results)==type(pd.DataFrame()):
                                live_env.results.load(instr_results)
                            
                            training_results = live_env.episode_loop()
                            training_results['episode'] = training_results.index

                            # Render current result after all instructions have been trained
                            if self.training_render:
                                if self.training_render_save_dir is None:
                                    current_render_save_dir = agent_save_dir
                                else:
                                    current_render_save_dir = self.training_render_save_dir
                              
                                render_current_result(training_setup = train_setup_info,
                                                            current_environment = live_env, current_agent = live_env.agent,
                                                            local_save_dir = current_render_save_dir)
                              
                            # Opponent now defined in local setup.py
                            # ----- Log training setup      
                            training_results.insert(loc=0, column='Repeat', value=setup_num)                    
                            # Produce training report with Analysis.py
                            Return = self.analysis.train_report(training_results, agent_save_dir, self.show_figures)
                            if goal not in temp_agent_store:
                                temp_agent_store[goal] = {}
                            temp_agent_store[goal][setup_num] = {'Return':Return,'agent':live_env.agent.clone()}
                                                                    
                            # Extract trained agent from env and stored for re-call
                            # - Observed states from best repeat stored for next seed
                            if training_repeat == 1:
                                max_Return = Return
                                training_results_stored =  live_env.results.copy()
                                observed_states_stored = live_env.elsciRL.observed_states
                            if Return > max_Return:
                                max_Return = Return
                                training_results_stored =  live_env.results.copy()
                                observed_states_stored = live_env.elsciRL.observed_states
                            
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

                                                
                    # Store last train_setup_info as collection of observed states
                    self.training_setups['Training_Setup_'+str(agent_type) + '_' + str(adapter)] = train_setup_info.copy()

        # Generate training variance report for all trained agents
        # Wrap in try-except to handle cases where folder structure doesn't match expectations
        try:
            self.analysis.training_variance_report(self.save_dir, self.show_figures)
        except (IndexError, FileNotFoundError) as e:
            print(f"\n[Info] Could not generate training_variance_report: {e}")
            print("This may occur when using only policy gradient agents with different output structure.")
                    
        #json.dump(self.training_setups) # TODO: Won't currently serialize this output to a json file
        return self.training_setups

    # TESTING PLAY
    def test(self, training_setups:str=None):
        # Override input training setups with previously saved 
        if training_setups is None:
            training_setups = self.training_setups
            print("=== TESTING SETUPS ===")
            print(training_setups.keys())
        else:
            json.load(training_setups)

        for training_key in list(training_setups.keys()):   
            test_setup_info = training_setups[training_key].copy()
            test_setup_info['train'] = False # Testing Phase
            
            # Check if this is a policy gradient agent (PPO)
            if training_key.startswith('Training_Setup_PPO'):
                print(f"\n{'='*60}")
                print(f"Detected Policy Gradient agent for testing")
                print(f"Using PolicyGradientExperiment test method...")
                print(f"{'='*60}\n")
                
                # Determine if this is flat or instruction-based
                if 'Flat' in training_key:
                    pg_key = 'PPO_Flat'
                else:
                    pg_key = 'PPO'
                
                # Use the stored PolicyGradientExperiment instance
                if pg_key in self.policy_gradient_experiments:
                    pg_exp = self.policy_gradient_experiments[pg_key]
                    # Pass the actual PG training setups, not the modified copy
                    pg_exp.test(training_setups=None)  # Uses stored training_setups
                else:
                    print(f"Warning: No PolicyGradientExperiment found for {pg_key}")
                continue
            
            agent_type = test_setup_info['agent_type']
            print("----------")
            print(training_key) 
            print("Testing results for trained agents in saved setup configuration:")
            print("TESTING SETUP INFO")
            print(test_setup_info['agent_type'])
            print(test_setup_info['adapter_select'])
            print("----------")
            agent_adapter = agent_type + "_" + test_setup_info['adapter_select']
            print(agent_adapter)
            # Added gym based agents as selection
            # - Strip reward signal from instructions for testing
            if self.is_gym_agent[agent_type]:
                gym_test_exp = self.training_setups[agent_adapter]
                gym_test_exp.reward_signal = None
                gym_test_exp.test()                        
            else:
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
                        print(test_save_dir)
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
        self.analysis.testing_variance_report(self.save_dir, self.show_figures)

        
    def render_results(self, training_setups:str=None):
        if training_setups is None:
            training_setups = self.training_setups
        else:
            training_setups = json.load(training_setups)
        
        render_results = None
        for training_key in list(training_setups.keys()):
            test_setup_info = training_setups[training_key].copy()
            test_setup_info['train'] = False
            test_setup_info['training_results'] = False
            test_setup_info['observed_states'] = False
            test_setup_info['num_test_episodes'] = 1
            print("----------\nRendering trained agent's policy:")
            
            # Check if this is a policy gradient agent (PPO)
            if training_key.startswith('Training_Setup_PPO'):
                print(f"\n{'='*60}")
                print(f"Detected Policy Gradient agent for rendering")
                print(f"Using PolicyGradientExperiment render_results method...")
                print(f"{'='*60}\n")
                
                # Determine if this is flat or instruction-based
                if 'Flat' in training_key:
                    pg_key = 'PPO_Flat'
                    render_subdir = 'render_results_flat'
                else:
                    pg_key = 'PPO'
                    render_subdir = 'render_results'
                
                # Use the stored PolicyGradientExperiment instance
                if pg_key in self.policy_gradient_experiments:
                    pg_exp = self.policy_gradient_experiments[pg_key]
                    render_save_dir = os.path.join(self.save_dir, render_subdir)
                    # Pass None to use stored training_setups
                    render_results = pg_exp.render_results(
                        training_setups=None,
                        render_save_dir=render_save_dir
                    )
                else:
                    print(f"Warning: No PolicyGradientExperiment found for {pg_key}")
                continue
            
            for engine_name, engine in self.engine_list.items():
                env = self.env_manager.create_env(engine, self.adapters, test_setup_info)
                start_obs = env.start_obs
                goal = str(start_obs).split(".")[0] + "---GOAL"
                print("Flat agent Goal: ", goal)
                agent_key = test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']
                if goal in self.trained_agents[agent_key]:
                    print("Trained agents available for testing.")
                    all_agents = self.trained_agents[agent_key][goal]
                    if not isinstance(all_agents, list):
                        all_agents = [all_agents]
                else:
                    print("NO agent available for testing position.")
                    all_agents = []
                for agent_num, agent in enumerate(all_agents):
                    env.results.reset()
                    env.start_obs = start_obs
                    env.agent = agent
                    env.num_train_repeat = 1
                    env.num_test_repeat = 1
                    env.number_episodes = 1
                    env.agent.epsilon = 0
                    render_save_dir = os.path.join(self.save_dir, 'render_results')
                    ensure_dir(render_save_dir)
                    render_results = env.episode_loop(render=True, render_save_dir=render_save_dir)
        return render_results

