import os
import json
# ------ Interaction Protocol -----------------------------------
from elsciRL.interaction_loops.standard import StandardInteractionLoop
# ------ Experiment Import --------------------------------------
from elsciRL.evaluation.standard_report import Evaluation
# ------ Agent Imports -----------------------------------------
# Universal Agents
from elsciRL.agents.agent_abstract import Agent, QLearningAgent
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.DQN import DQNAgent
from elsciRL.agents.agent_abstract import Agent
# Stable Baselines
from elsciRL.agents.stable_baselines.SB3_DQN import SB_DQN
from elsciRL.agents.stable_baselines.SB3_PPO import SB_PPO
from elsciRL.agents.stable_baselines.SB3_A2C import SB_A2C
# ------ Gym Experiment ----------------------------------------
from elsciRL.experiments.GymExperiment import GymExperiment
from elsciRL.experiments.experiment_utils.render_current_results import render_current_result
# ------ LLM Agents ---------------------------------------------
from elsciRL.agents.LLM_agents.ollama_agent import LLMAgent as OllamaAgent
# ---------------------------------------------------------------

# This is the main run functions for elsciRL to be imported
# Defines the train/test operators and imports all the required agents and experiment functions ready to be used
# The local main.py file defines the [adapters, configs, environment] to be input

# This should be where the environment is initialised and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead
# -> DONE: This means we have multiple main.py types (e.g. with/without convergence measure) so should create a directory and finalize naming for this

class Experiment:
    """This is the standard Reinforcement Learning experiment setup for a flat agent. 
    - The agent is trained for a fixed number of episodes
    - Then learning is fixed to be applied during testing phase
    - Repeats (or seeds if environment start position changes) are used for statistical significant testing
    """
    def __init__(self, Config:dict, ProblemConfig:dict, Engine, Adapters:dict, save_dir:str, show_figures:str, window_size:float, 
                 training_render:bool=False, training_render_save_dir:str=None): 
        # Environment setup
        # - Multiple Engine support
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
        # ---
        # Configuration setup
        self.ExperimentConfig = Config
        self.LocalConfig = ProblemConfig
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir+'/Standard_Experiment'
        self.show_figures = show_figures
        if (self.show_figures.lower() != 'y')|(self.show_figures.lower() != 'yes'):
            print("Figures will not be shown and only saved.")

        # Run render results during training to show progress
        self.training_render = training_render
        self.training_render_save_dir = training_render_save_dir

        try:
            self.setup_info = self.ExperimentConfig['data'] | self.LocalConfig['data'] 
        except:
            self.setup_info = self.ExperimentConfig | self.LocalConfig 

        # Transforms adapter input to complete matching, i.e. all agents trained on all adapters
        if 'adapter_input_dict' in self.ExperimentConfig:
            self.setup_info['adapter_input_dict'] = self.ExperimentConfig['adapter_input_dict']
        else:
            selected_adapters = list(Adapters.keys())
            selected_agents = self.ExperimentConfig['agent_select']
            agent_adapter_dict = {agent_name: list(selected_adapters) for agent_name in selected_agents} if selected_agents else {}
            self.ExperimentConfig['adapter_input_dict'] = agent_adapter_dict
            self.setup_info['adapter_input_dict'] = agent_adapter_dict

        self.training_setups: dict = {}
        # new - store agents cross training repeats for completing the same start-end goal
        self.trained_agents: dict = {}
        self.num_training_seeds = self.setup_info['number_training_seeds']
        # new - config input defines the re-use of trained agents for testing: 'best' or 'all'
        if self.setup_info['test_agent_type']:
            self.test_agent_type = self.setup_info['test_agent_type']
        else:
            self.test_agent_type = 'all'
        self.analysis = Evaluation(window_size=window_size)
        # ---
        # Gym setup
        self.is_gym_agent = {}
        for n,agent_type in enumerate(self.setup_info['agent_select']):
            if agent_type.split('_')[0] == "SB3":
                self.is_gym_agent[agent_type] = True 
                self.gym_exp = GymExperiment(Config=self.ExperimentConfig, ProblemConfig=self.LocalConfig, 
                        Engine=self.engine, Adapters=self.adapters,
                        save_dir=self.save_dir, show_figures = self.show_figures, window_size=0.1)
            else:
                self.is_gym_agent[agent_type] = False 
        # ---
        # Agent setup
        self.AGENT_TYPES = {
            "Qlearntab": TableQLearningAgent,
            "DQN": DQNAgent,
            "LLM_Ollama": OllamaAgent,
        }
        

    def add_agent(self, agent_name:str, agent):
        """Add a custom agent to the experiment using the agent name as a key.
            - Parameters must be defined in the config.json file with matching name."""
        self.AGENT_TYPES[agent_name] = agent
        print("\n Agent added to experiment, all available agents: ", self.AGENT_TYPES)


    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        for engine_name, engine in self.engine_list.items():
            for n, agent_type in enumerate(self.setup_info['agent_select']):
                # Added gym based agents as selection
                is_gym_agent = self.is_gym_agent[agent_type]
                agent_adapter = (self.setup_info["agent_select"][n]+'_'+self.setup_info["adapter_input_dict"][self.setup_info["agent_select"][n]][0])
                if is_gym_agent:
                    train_setup_info = self.setup_info.copy()
                    # --- GYM EXPERIMENT TRAINING
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
                        train_setup_info['agent_name'] = str(engine_name) + str(agent_type) + '_' + str(adapter) + '_' + str(agent_parameters)
                        train_setup_info['adapter_select'] = adapter
                        # ----- Neural Agent Setup
                        # Get the input dim from the adapter or the encoder's output dim
                        # TODO: Currently has to initalize adapter to get this info and should be improved
                        if agent_type == "DQN":
                            try:
                                agent_parameters['input_size'] = self.adapters[adapter](train_setup_info).input_dim
                            except:
                                try:
                                    agent_parameters['input_size'] = self.adapters[adapter](train_setup_info).encoder.output_dim
                                except:
                                    print(f"No input dim found in the specified adapter: {adapter}. Please provide this as self.output_dim in the adapter class.")
                                    raise ValueError(f"No output dim size found in adapter: {adapter}")
                        # -----
                        # Repeat training
                        train_setup_info['train'] = True
                        number_training_repeats = self.ExperimentConfig["number_training_repeats"]
                        print("Training Agent " + str(agent_type) + " for " + str(number_training_repeats) + " repeats on " + str(engine_name) + " engine")
                        if str(engine_name) + '_' + str(agent_type) + '_' + str(adapter) not in self.trained_agents:
                            self.trained_agents[str(engine_name) + '_' + str(agent_type) + '_' + str(adapter)] = {}

                        seed_recall = {}
                        seed_results_connection = {}
                        for seed_num in range(0,self.num_training_seeds):
                            if self.num_training_seeds > 1:
                                print("------")
                                print("- Seed Num: ", seed_num)
                            # -------------------------------------------------------------------------------
                            # Initialise Environment
                            # Environment now init here and called directly in experimental setup loop
                            # - NEW: need to pass start position from live env so that experience can be sampled
                            if seed_num==0:
                                train_setup_info['training_results'] = False
                                train_setup_info['observed_states'] = False
                            else:
                                train_setup_info['training_results'] = False
                                train_setup_info['observed_states'] = observed_states_stored.copy()
                            # ---
                            setup_num:int = 0
                            temp_agent_store:dict = {}
                            for training_repeat in range(1,number_training_repeats+1):
                                if number_training_repeats > 1:
                                    print("------")
                                    print("- Repeat Num: ", training_repeat)
                                setup_num+=1
                                
                                # ----- init agent
                                player = self.AGENT_TYPES[agent_type](**agent_parameters) 
                                train_setup_info['agent'] = player 
                                
                                train_setup_info['live_env'] = True
                                live_env = self.env(Engine=engine, Adapters=self.adapters, local_setup_info=train_setup_info)
                                
                                if training_repeat > 1:
                                    live_env.start_obs = env_start

                                env_start = live_env.start_obs
                                goal = str(env_start).split(".")[0] + "---" + "GOAL"
                                print("Flat agent Goal: ", goal)
                                if goal in seed_recall:
                                    setup_num = seed_recall[goal]
                                else:
                                    seed_recall[goal] = 1
                                # - Results save dir -> will override for same goal if seen in later seed
                                if self.num_training_seeds > 1:
                                    agent_save_dir = self.save_dir+'/'+engine_name+'_'+agent_type+'_'+adapter+'__training_results_'+str(goal)+'_'+str(setup_num) 
                                else:
                                    agent_save_dir = self.save_dir+'/'+engine_name+'_'+agent_type+'_'+adapter+'__training_results_'+str(setup_num)
                                if not os.path.exists(agent_save_dir):
                                    os.mkdir(agent_save_dir)

                                # TODO: UPDATE THIS TO HANDLE REPEAT TESTING BETTER
                                # Override with trained agent if goal seen previously
                                if goal in self.trained_agents[str(engine_name) + '_' + str(agent_type) + '_' + str(adapter)]:
                                    live_env.agent = self.trained_agents[str(engine_name) + '_' + str(agent_type) + '_' + str(adapter)][goal].clone()

                                # ---
                                if goal in seed_results_connection:
                                    live_env.results.load(seed_results_connection[goal])
                                #live_env.agent.exploration_parameter_reset()
                                training_results = live_env.episode_loop()
                                training_results['episode'] = training_results.index
                                # Opponent now defined in local setup.py
                                # ----- Log training results      
                                training_results.insert(loc=0, column='Repeat', value=setup_num)
                                # Produce training report with Analysis.py
                                Return = self.analysis.train_report(training_results, agent_save_dir, self.show_figures)
                                # Extract trained agent from env and stored for re-call
                                if goal not in temp_agent_store:
                                    temp_agent_store[goal] = {}
                                temp_agent_store[goal][setup_num] = {'Return':Return,'agent':live_env.agent.clone()}
                                
                                if training_repeat == 1:
                                    max_Return = Return
                                    best_agent = live_env.agent
                                    training_results_stored =  live_env.results.copy()
                                    observed_states_stored = live_env.elsciRL.observed_states
                                if Return > max_Return:
                                    max_Return = Return
                                    best_agent = live_env.agent
                                    training_results_stored =  live_env.results.copy()
                                    observed_states_stored = live_env.elsciRL.observed_states
                                seed_recall[goal] = seed_recall[goal] + 1
                                # Save trained agent to logged output
                                train_setup_info['train_save_dir'] = agent_save_dir
                                #train_setup_info['trained_agent'] = agent
                                if self.training_render:
                                    if self.training_render_save_dir is None:
                                        current_render_save_dir = agent_save_dir
                                    else:
                                        current_render_save_dir = self.training_render_save_dir
                                    # We override inheretied variables so need new env spec to not mess with training
                                    render_current_result(training_setup = train_setup_info,
                                                        current_environment = live_env, current_agent = live_env.agent,
                                                        local_save_dir = current_render_save_dir)
                            seed_results_connection[goal] = training_results_stored

                            # ----- New: 'best' or 'all' agents saved
                            # Save trained agent to logged output for testing phase
                            if self.test_agent_type.lower() == 'best':
                                self.trained_agents[str(engine_name) + '_' + str(agent_type) + '_' + str(adapter)][goal] = best_agent.clone()
                            elif self.test_agent_type.lower() == 'all':
                                start_repeat_num = list(temp_agent_store[goal].keys())[0]
                                end_repeat_num = list(temp_agent_store[goal].keys())[-1]

                                all_agents = []
                                for repeat in range(start_repeat_num,end_repeat_num+1):
                                    agent = temp_agent_store[goal][repeat]['agent']
                                    all_agents.append(agent)
                                    
                                if goal not in self.trained_agents[str(engine_name) + '_' + str(agent_type) + '_' + str(adapter)]:
                                    self.trained_agents[str(engine_name) + '_' + str(agent_type) + '_' + str(adapter)][goal] = {}
                                self.trained_agents[str(engine_name) + '_' + str(agent_type) + '_' + str(adapter)][goal] = all_agents

                            # Collate complete setup info to full dict
                        self.training_setups['Training_Setup_'+str(engine_name) + '_' + str(agent_type)+'_'+str(adapter)] = train_setup_info.copy()
                        #if (number_training_repeats>1)|(self.num_training_seeds):
                        self.analysis.training_variance_report(self.save_dir, self.show_figures)

        return self.training_setups

    # TESTING PLAY
    def test(self, training_setups:str=None):
        # Override input training setups with previously saved 
        if training_setups is None:
            training_setups = self.training_setups
        else:
            training_setups = json.load(training_setups)

        for training_key in list(training_setups.keys()):    
            test_setup_info = training_setups[training_key]
            test_setup_info['train'] = False # Testing Phase
            test_setup_info['training_results'] = False
            test_setup_info['observed_states'] = False
            agent_type = test_setup_info['agent_type']
            print("----------")
            print(training_key) 
            print("Testing results for trained agents in saved setup configuration:")
            print("TESTING SETUP INFO")
            print(test_setup_info['agent_type'])
            print(test_setup_info['adapter_select'])
            print("----------")
            agent_adapter = agent_type + "_" + test_setup_info['adapter_select']

            if self.is_gym_agent[agent_type]:
                gym_test_exp = self.training_setups[agent_adapter]
                gym_test_exp.reward_signal = None
                gym_test_exp.test()   
            else:
                # Only use the trained agent with best return
                if self.test_agent_type.lower()=='best':
                    for engine_name, engine in self.engine_list.items():
                        for testing_repeat in range(0, test_setup_info['number_test_repeats']):  
                            # Re-init env for testing
                            env = self.env(Engine=engine, Adapters=self.adapters, local_setup_info=test_setup_info)
                            # ---
                            start_obs = env.start_obs
                            goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                            print("Flat agent Goal: ", goal)
                            # Override with trained agent if goal seen previously
                            if goal in self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                                print("Trained agent available for testing.")
                                env.agent = self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
                            else:
                                print("NO agent available for testing position.")
                            try:
                                env.agent.epsilon = 0 # Remove random actions
                            except:
                                print(self.trained_agents)
                                raise KeyError("Trained agents lookup not found for testing position.")
                            
                            # ---
                            # Testing generally is the agents replaying on the testing ENV
                            testing_results = env.episode_loop() 
                            test_save_dir = (self.save_dir+'/' + str(engine_name) + '_' + agent_adapter + '__testing_results_' + str(goal).split("/")[0]+"_"+str(testing_repeat))
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
                    for engine_name, engine in self.engine_list.items():
                        for testing_repeat in range(0, test_setup_info['number_test_repeats']):
                            env = self.env(Engine=engine, Adapters=self.adapters, local_setup_info=test_setup_info)
                            # ---
                            start_obs = env.start_obs
                            goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                            print("Flat agent Goal: ", goal)
                            # Override with trained agent if goal seen previously
                            if goal in self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                                print("Trained agents available for testing.")
                                all_agents = self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
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
                                test_save_dir = (self.save_dir+'/'+ str(engine_name) + '_' + agent_adapter + '__testing_results_' + str(goal).split("/")[0]+"_"+"agent"+str(ag)+"-repeat"+str(testing_repeat))
                                if not os.path.exists(test_save_dir):
                                    os.mkdir(test_save_dir)
                                # Produce training report with Analysis.py
                                Return = self.analysis.test_report(testing_results, test_save_dir, self.show_figures)

            # if (number_training_repeats>1)|(self.test_agent_type.lower()=='all'):
            self.analysis.testing_variance_report(self.save_dir, self.show_figures)

        
    def render_results(self, training_setups:str=None):
        """Apply fixed policy to render decision making of all trained agents for limited number of episodes."""
        # Override input training setups with previously saved 
        
        if training_setups is None:
            training_setups = self.training_setups
        else:
            json.load(training_setups)

        for training_key in list(training_setups.keys()):    
            test_setup_info = training_setups[training_key]
            test_setup_info['train'] = False # Testing Phase
            test_setup_info['training_results'] = False
            test_setup_info['observed_states'] = False
            print("----------")
            print("Rendering trained agent's policy:")
            agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']


            if self.test_agent_type.lower()=='best':
                for engine_name, engine in self.engine_list.items():
                    # Re-init env for testing
                    env = self.env(Engine=engine, Adapters=self.adapters, local_setup_info=test_setup_info)
                    # ---
                    start_obs = env.start_obs
                    goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                    print("Flat agent Goal: ", goal)
                    # Override with trained agent if goal seen previously
                    if goal in self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                        print("Trained agent available for testing.")
                        env.agent = self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
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
            else:  
                for engine_name, engine in self.engine_list.items():
                    ## Re-init env for testing
                    env = self.env(Engine=engine, Adapters=self.adapters, local_setup_info=test_setup_info)
                    # ---
                    start_obs = env.start_obs
                    goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                    print("Flat agent Goal: ", goal)
                    # Override with trained agent if goal seen previously
                    if goal in self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                        print("Trained agents available for testing.")
                        all_agents = self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
                    else:
                        print("NO agent available for testing position.")
                    #Only render first even if all selected
                    for ag,agent in enumerate(all_agents[:1]):
                        env.results.reset() # Reset results table for each agent
                        env.start_obs = start_obs
                        env.agent = agent
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