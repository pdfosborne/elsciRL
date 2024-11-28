# TODO: REMOVE OBSERVED STATES TRACKING
import os
import json

# ------ Experiment Import --------------------------------------
from elsciRL.evaluation.standard_report import Evaluation
# ------ Agent Imports -----------------------------------------
# Universal Agents
from elsciRL.agents.agent_abstract import Agent, QLearningAgent
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.neural_q_agent import NeuralQLearningAgent

AGENT_TYPES = {
    "Qlearntab": TableQLearningAgent,
    "Neural_Q": NeuralQLearningAgent,
    "Neural_Q_2": NeuralQLearningAgent,
    "Neural_Q_language": NeuralQLearningAgent
}

PLAYER_PARAMS = {
    "Qlearntab": ["alpha", "gamma", "epsilon"],
    "Neural_Q": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "Neural_Q_2": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "Neural_Q_language": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"]
}

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
    - Experience Sampling stores observed episodes into a sampled MDP model to learn from to improve training efficiency
    """
    def __init__(self, Config:dict, LocalConfig:dict, Environment, save_dir:str, show_figures:str, window_size:float):
        self.ExperimentConfig = Config
        self.LocalConfig = LocalConfig
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir+'/Standard_Experiment'
        self.show_figures = show_figures
        if (self.show_figures.lower() != 'y')|(self.show_figures.lower() != 'yes'):
            print("Figures will not be shown and only saved.")

        self.env = Environment
        try:
            self.setup_info = self.ExperimentConfig['data'] | self.LocalConfig['data'] 
        except:
            self.setup_info = self.ExperimentConfig | self.LocalConfig 
        self.training_setups: dict = {}
        # new - store agents cross training repeats for completing the same start-end goal
        self.trained_agents: dict = {}
        self.num_training_seeds = self.setup_info['number_training_seeds']
        # new - config input defines the re-use of trained agents for testing: 'best' or 'all'
        self.test_agent_type = self.setup_info['test_agent_type']
        self.analysis = Evaluation(window_size=window_size)
        
    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
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
                    train_setup_info['experience_sampling'] = False
                else:
                    train_setup_info['training_results'] = False
                    train_setup_info['observed_states'] = observed_states_stored.copy()
                    train_setup_info['experience_sampling'] = experience_sampling_stored.copy()
                # ---
                setup_num:int = 0
                temp_agent_store:dict = {}
                for training_repeat in range(1,number_training_repeats+1):
                    if number_training_repeats > 1:
                        print("------")
                        print("- Repeat Num: ", training_repeat)
                    setup_num+=1
                    
                    # ----- init agent
                    player = AGENT_TYPES[agent_type](**agent_parameters) 
                    train_setup_info['agent'] = player 
                    
                    train_setup_info['live_env'] = True
                    live_env = self.env(train_setup_info)
                    
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
                        agent_save_dir = self.save_dir+'/'+agent_type+'_'+adapter+'__training_results_'+str(goal)+'_'+str(setup_num) 
                    else:
                        agent_save_dir = self.save_dir+'/'+agent_type+'_'+adapter+'__training_results_'+str(setup_num)
                    if not os.path.exists(agent_save_dir):
                        os.mkdir(agent_save_dir)

                    # Override with trained agent if goal seen previously
                    if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                        live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()

                    if train_setup_info['experience_sample_batch_ratio']>0:
                        live_sample_batch_size = int(number_training_episodes*train_setup_info['experience_sample_batch_ratio'])
                        live_sample_batch_count = int(1/train_setup_info['experience_sample_batch_ratio'])
                        # Train on Live system for limited number of total episodes
                        live_env.num_train_episodes = live_sample_batch_size
                        print("-- Training with Simulated Batches, ", live_sample_batch_count, " total...")
                        # init simulated environment
                        train_setup_info['live_env'] = False
                        simulated_env = self.env(train_setup_info)
                        simulated_env.start_obs = env_start
                        # Connect live agent -> This should be linked so continues learning
                        simulated_env.agent = live_env.agent

                        for live_sample_batch in range(0, live_sample_batch_count-1):
                            print("--- Live Interaction Batch Num", live_sample_batch+1)                            
                            training_results = live_env.episode_loop()

                            # Train based on simulated exp  
                            simulated_env.num_train_episodes = number_training_episodes
                            simulated_env.episode_loop()
                        # Final batch doesn't require simulated exp after and we need result output
                        print("--- Final batch")
                        train_setup_info['live_env'] = True
                        train_setup_info['number_training_episodes'] = live_sample_batch_size
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
                        experience_sampling_stored = live_env.elsciRL.experience_sampling
                    if Return > max_Return:
                        max_Return = Return
                        best_agent = live_env.agent
                        training_results_stored =  live_env.results.copy()
                        observed_states_stored = live_env.elsciRL.observed_states
                        experience_sampling_stored = live_env.elsciRL.experience_sampling
                    seed_recall[goal] = seed_recall[goal] + 1
                    # Save trained agent to logged output
                    train_setup_info['train_save_dir'] = agent_save_dir
                    #train_setup_info['trained_agent'] = agent
                seed_results_connection[goal] = training_results_stored

                # ----- New: 'best' or 'all' agents saved
                # Save trained agent to logged output for testing phase
                if self.test_agent_type.lower() == 'best':
                    self.trained_agents[str(agent_type) + '_' + str(adapter)][goal] = best_agent.clone()
                elif self.test_agent_type.lower() == 'all':
                    start_repeat_num = list(temp_agent_store[goal].keys())[0]
                    end_repeat_num = list(temp_agent_store[goal].keys())[-1]

                    all_agents = []
                    for repeat in range(start_repeat_num,end_repeat_num+1):
                        agent = temp_agent_store[goal][repeat]['agent']
                        all_agents.append(agent)
                        
                    if goal not in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                        self.trained_agents[str(agent_type) + '_' + str(adapter)][goal] = {}
                    self.trained_agents[str(agent_type) + '_' + str(adapter)][goal] = all_agents

                # Collate complete setup info to full dict
            self.training_setups['Training_Setup_'+str(agent_type)+'_'+str(adapter)] = train_setup_info
        if (number_training_repeats>1)|(self.num_training_seeds):
            self.analysis.training_variance_report(self.save_dir, self.show_figures)
        #json.dump(self.training_setups) # TODO: Won't currently serialize this output to a json file
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
            test_setup_info['experience_sampling'] = False
            print("----------")
            print("Testing results for trained agents in saved setup configuration:")
            print(test_setup_info['train_save_dir'])
            number_training_repeats = test_setup_info['number_test_repeats']
            agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']

            # Only use the trained agent with best return
            if self.test_agent_type.lower()=='best':
                for testing_repeat in range(0, test_setup_info['number_test_repeats']):  
                    # Re-init env for testing
                    env = self.env(test_setup_info)
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
                    env = self.env(test_setup_info)
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
                        test_save_dir = (self.save_dir+'/'+agent_adapter+'__testing_results_'+str(goal).split("/")[0]+"_"+"agent"+str(ag)+"-repeat"+str(testing_repeat))
                        if not os.path.exists(test_save_dir):
                            os.mkdir(test_save_dir)
                        # Produce training report with Analysis.py
                        Return = self.analysis.test_report(testing_results, test_save_dir, self.show_figures)

        if (number_training_repeats>1)|(self.test_agent_type.lower()=='all'):
            self.analysis.testing_variance_report(self.save_dir, self.show_figures)