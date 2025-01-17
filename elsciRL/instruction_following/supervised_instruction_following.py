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

# TODO: This should be where the environment is initialized and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead
# -> TODO: This means we have multiple main.py types (e.g. with/without convergence measure) so should create a directory and finalise naming for this

class SupervisedExperiment:
    def __init__(self, Config:dict, LocalConfig:dict, Environment, 
                 save_dir:str, show_figures:str, window_size:float, 
                 instruction_path: dict, predicted_path: dict=None):
        self.ExperimentConfig = Config
        self.LocalConfig = LocalConfig
        
        if not predicted_path:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.save_dir = save_dir+'/Supervised_Instr_Experiment'
        else:
            save_dir_extra = save_dir.split("/")[-1]
            save_dir = '/'.join(save_dir.split("/")[:-1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.save_dir = save_dir+'/'+save_dir_extra

        self.show_figures = show_figures
        if (self.show_figures.lower() != 'y')|(self.show_figures.lower() != 'yes'):
            print("Figures will not be shown and only saved.")

        self.env = Environment
        self.setup_info = self.ExperimentConfig['data'] | vars(self.LocalConfig) # TODO: configs aren't consistent formatting 
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
        self.instruction_agents:dict = {}
    
        self.analysis = Evaluation(window_size=window_size)
        
    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        prior_sub_goal = None
        for i,instruction in enumerate(list(self.instruction_path.keys())): 
            print(" ")
            print("Instruction Sub-Goal: ", instruction)
            instr_save_dir = self.save_dir+'/'+str(i+1)+"-"+instruction.replace(" ","")
            self.instr_dir = instr_save_dir
            if not os.path.exists(instr_save_dir):
                os.mkdir(instr_save_dir)
                
            for n, agent_type in enumerate(self.setup_info['agent_select']):
                # We are adding then overriding some inputs from general configs for experimental setups
                train_setup_info = self.setup_info.copy()
                train_setup_info['instruction'] = instruction
                # Override action cap for shorter term sub-goals for faster learning
                if 'action_cap' in self.instruction_path[instruction]:
                    train_setup_info['training_action_cap'] = self.instruction_path[instruction]['action_cap'] 
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
                
                # If unsupervised/reinf search not performed then dict structure wont include agent_type/adapter
                if type(self.instruction_path[instruction]['env_code']) == type({}):
                    sub_goal = self.instruction_path[instruction]['env_code'][agent_type+'_'+adapter]['sub_goal']
                    train_setup_info['sub_goal'] = sub_goal
                else:
                    sub_goal = self.instruction_path[instruction]['env_code']
                    train_setup_info['sub_goal'] = sub_goal
                # -----
                # Repeat training
                train_setup_info['train'] = True
                number_training_episodes = train_setup_info['number_training_episodes']
                number_training_repeats = self.ExperimentConfig["number_training_repeats"]
                print("Training Agent " + str(agent_type) + " for " + str(number_training_repeats) + " repeats")
                
                temp_agent_store = {}
                setup_num:int = 0
                for training_repeat in range(1,(number_training_repeats+1)):
                    print("------")
                    print("- Repeat Num: ", training_repeat)
                    setup_num+=1
                    # ----- init agent
                    if instruction not in self.instruction_agents:
                        # Instructions are in sequence so we initialise later instructions with policy of one prior
                        if not prior_sub_goal:
                            player = AGENT_TYPES[agent_type](**agent_parameters)
                        else:
                            self.instruction_agents[prior_instruction][agent_type].clone() # We need it to adopt prior sub-goal but create a new instance otherwise repeats wont reset learning
                            #player = AGENT_TYPES[agent_type](**agent_parameters)
                            #player.load(self.instruction_agents[prior_sub_goal][agent_type])  
                    else:
                        self.instruction_agents[instruction][agent_type].clone()
                        #player = AGENT_TYPES[agent_type](**agent_parameters)
                        #player.load(self.instruction_agents[sub_goal][agent_type])
                    train_setup_info['agent'] = player
                    # -------------------------------------------------------------------------------
                    # Initialise Environment
                    # Environment now init here and called directly in experimental setup loop

                    # Initialise Environment
                    # Environment now init here and called directly in experimental setup loop
                    train_setup_info['training_results'] = False
                    train_setup_info['observed_states'] = False
                    train_setup_info['experience_sampling'] = False
                    if train_setup_info['experience_sample_batch_ratio']>0:
                        live_sample_batch_size = int(number_training_episodes*train_setup_info['experience_sample_batch_ratio'])
                        live_sample_batch_count = int(1/train_setup_info['experience_sample_batch_ratio'])
                        print("-- Training with Simulated Batches, ", live_sample_batch_count, " total...")
                        for live_sample_batch in range(0, live_sample_batch_count-1):
                            print("--- Live Interaction Batch Num", live_sample_batch+1)
                            # Train on Live system for limited number of total episodes
                            train_setup_info['live_env'] = True                        
                            train_setup_info['number_training_episodes'] = live_sample_batch_size
                            live_env = self.env(train_setup_info)
                            training_results = live_env.episode_loop()

                            # Update shared information between init environments
                            train_setup_info['training_results'] = training_results
                            train_setup_info['observed_states'] = live_env.elsciRL.observed_states
                            train_setup_info['experience_sampling'] = live_env.elsciRL.experience_sampling
                            # Train based on simulated exp
                            train_setup_info['live_env'] = False
                            train_setup_info['number_training_episodes'] = number_training_episodes

                            simulated_env = self.env(train_setup_info)
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
                        train_setup_info['live_env'] = True
                        live_env = self.env(train_setup_info)
                        training_results = live_env.episode_loop()

                    # Opponent now defined in local setup.py
                    # ----- Log training setup      
                    training_results.insert(loc=0, column='Repeat', value=setup_num)
                    # Extract trained agent from env
                    agent = live_env.agent
                    # ----- Save Directory -> Sub-directory for each setup type (parameters+opponents+repeats)
                    # -> I also moved env init into inner most loop because I'm worried the agent isn't getting reset otherwise
                    agent_save_dir = instr_save_dir+'/'+agent_type+'_'+adapter+'_'+str(setup_num) 
                    
                    if not os.path.exists(agent_save_dir):
                        os.mkdir(agent_save_dir)
                    # Produce training report with Analysis.py
                    Return = self.analysis.train_report(training_results, agent_save_dir, self.show_figures)
                                        
                    # Save trained agent to logged output
                    train_setup_info['train_save_dir'] = agent_save_dir
                    train_setup_info['trained_agent'] = agent
                    # Collate complete setup info to full dict
                    self.training_setups['Training_Setup_'+str(agent_type) + '_' + str(adapter)+'_'+str(setup_num)] = train_setup_info.copy()
                    temp_agent_store[setup_num] = {'Return':Return,'agent':agent}

                # Only save the best agent from repeated training
                best_return = float('-inf')
                best_agent = None
                for repeat in range(1, training_repeat + 1):
                    if temp_agent_store[repeat]['Return'] > best_return:
                        best_return = temp_agent_store[repeat]['Return']
                        best_agent = temp_agent_store[repeat]['agent']
                        print(best_return)

                if instruction not in self.instruction_agents:
                    self.instruction_agents[instruction] = {}
                self.instruction_agents[instruction][agent_type] = best_agent#.save()          
            # For next instruction to adopt policy from
            prior_instruction = instruction
            prior_sub_goal = sub_goal
            if number_training_repeats>1:
                self.analysis.training_variance_report(instr_save_dir, self.show_figures)

        #json.dump(self.training_setups) # TODO: Won't currently serialize this output to a json file
        print(self.instruction_agents) 
        return self.training_setups

    # TESTING PLAY
    def test(self, training_setups:str=None):
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
            print(test_setup_info['train_save_dir'])
            # Extract trained agent
            test_setup_info['agent'] = test_setup_info['trained_agent'] #Update init agent to pre-trained
            number_training_repeats = test_setup_info['number_test_repeats']

            # If we trained with predicted goal positions we now test with correct, supervised goals
            if self.testing_path:
                instruction = test_setup_info['instruction']
                test_setup_info['sub_goal'] = self.testing_path[instruction]['env_code']

            for testing_repeat in range(0, test_setup_info['number_test_repeats']):  
                # Re-init env for testing
                env = self.env(test_setup_info)
                # Testing generally is the agents replaying on the testing ENV
                testing_results = env.episode_loop() 
                test_save_dir = test_setup_info['train_save_dir']+'/testing_results_'+str(testing_repeat)
                if not os.path.exists(test_save_dir):
                    os.mkdir(test_save_dir)
                # Produce training report with Analysis.py
                Return = self.analysis.test_report(testing_results, test_save_dir, self.show_figures)

        # Path is the experiment save dir + the final instruction
        if number_training_repeats>1:
            self.analysis.testing_variance_report(self.instr_dir, self.show_figures)