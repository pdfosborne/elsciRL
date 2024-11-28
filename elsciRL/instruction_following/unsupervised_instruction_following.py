import torch

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

from elsciRL.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

# This is the main run functions for elsciRL to be imported
# Defines the train/test operators and imports all the required agents and experiment functions ready to be used
# The local main.py file defines the [adapters, configs, environment] to be input

# TODO: This should be where the environment is initialized and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead
# -> TODO: This means we have multiple main.py types (e.g. with/without convergence measure) so should create a directory and finalise naming for this

class UnsupervisedSearch():
    def __init__(self,Config:dict, LocalConfig:dict, Environment, 
                 number_exploration_episodes:int = 100, sim_threshold:float = 0.9,
                 observed_states:dict=None, instruction_results:dict=None):
        self.ExperimentConfig = Config
        self.LocalConfig = LocalConfig

        self.env = Environment
        self.setup_info = self.ExperimentConfig.data | self.LocalConfig.data # TODO: configs aren't consistent formatting 
        self.training_setups: dict = {}

        # New instruction learning
        if not observed_states:
            self.observed_states:dict = {}
        else:
            self.observed_states = observed_states   
        if not instruction_results:
            self.instruction_results:dict = {}
        else:
            self.instruction_results = instruction_results
            
        # Unsupervised search parameters
        self.enc = LanguageEncoder()
        self.number_exploration_episodes = number_exploration_episodes
        self.sim_threshold = sim_threshold
        self.cos = torch.nn.CosineSimilarity(dim=0) 


    def search(self, instruction:str='', instr_description:str='', action_cap:int=5, re_search_override:bool=False):
        # Trigger re-search
        if re_search_override:
            self.observed_states:dict = {}
        print(" ")
        print("Instruction: ", instruction)
        
        # Create tensor vector of description
        instruction_vector = self.enc.encode(' '.join(instr_description))
        for n, agent_type in enumerate(self.setup_info['agent_select']):
            # We are adding then overriding some inputs from general configs for experimental setups
            train_setup_info = self.setup_info.copy()
            # Override action cap for shorter term sub-goals for faster learning
            train_setup_info['training_action_cap'] = action_cap 
            # ----- State Adapter Choice
            adapter = train_setup_info["adapter_select"][n]
            # ----- Agent parameters
            agent_parameters = train_setup_info["agent_parameters"][agent_type]
            train_setup_info['agent_type'] = agent_type
            train_setup_info['agent_name'] = str(agent_type) + '_' + str(adapter) + '_' + str(agent_parameters)
            train_setup_info['adapter_select'] = adapter
            # ----- init agent
            player = AGENT_TYPES[agent_type](**agent_parameters)
            train_setup_info['agent'] = player
            # -----
            # Set env function to training
            train_setup_info['train'] = True
            # ---------------------------------SEARCH-----------------------------------------
            # EXPLORE TO FIND LOCATION OF SUB-GOAL
            sub_goal = None
            train_setup_info['training_results'] = False
            if not self.observed_states:
                train_setup_info['observed_states'] = False
            else:
                train_setup_info['observed_states'] = self.observed_states
            train_setup_info['experience_sampling'] = False
            train_setup_info['live_env'] = True  
            # Seen sub_goal before & sim above threshold
            if (instruction in self.instruction_results):
                if (agent_type+'_'+adapter) in self.instruction_results[instruction]['env_code']:
                    if (self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['sim_score']>=self.sim_threshold):
                        sub_goal = self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['sub_goal']
            
            if not sub_goal:
                # If no description -> no sub-goal (i.e. envs terminal goal position)
                if not instr_description: 
                    sub_goal = None
                # If no sub-goal -> find best match of description from env 
                else:
                    # Run on live env if observed states unknown
                    if not self.observed_states:
                        print("Instruction not known, searching for matching observation in environment...")
                        train_setup_info['sub_goal'] = sub_goal
                        # --- 
                        # Set exploration parameters
                        train_setup_info['number_training_episodes'] = self.number_exploration_episodes # Override 
                        # ---
                        # Explore env with limited episodes
                        live_env = self.env(train_setup_info)
                        explore_results = live_env.episode_loop() 
                        train_setup_info['training_results'] = explore_results
                        train_setup_info['observed_states'] = live_env.elsciRL.observed_states
                        train_setup_info['experience_sampling'] = live_env.elsciRL.experience_sampling
                        
                        # Extract visited states from env
                        self.observed_states = live_env.elsciRL.observed_states.copy()
                        # for obs in env.elsciRL.observed_states:
                        #     if obs not in self.observed_states:
                        #         self.observed_states[obs] = env.elsciRL.observed_states[obs]
                        #observed_states = env.elsciRL.observed_states
                    
                    # Compare to instruction vector                            
                    max_sim = -1
                    # Any states that are above threshold 
                    sub_goal_list = []
                    for obs_state in self.observed_states:
                        str_state = self.observed_states[obs_state]
                        str_state_stacked = ' '.join(str_state)
                        t_state = self.enc.encode(str_state_stacked)
                        # Average sim across each sentence in instruction vs state
                        total_sim = 0
                        for instr_sentence in instruction_vector:
                            for state_sentence in t_state:
                                total_sim+=self.cos(state_sentence, instr_sentence)
                        sim = total_sim.item()/(len(instruction_vector)*len(t_state))
                        if sim > max_sim:
                            max_sim  = sim
                        if sim > self.sim_threshold:
                            sub_goal = obs_state # Sub-Goal code
                            sub_goal_list.append(sub_goal)
                    # OR if none above threshold within (1-threshold%) of max sim
                    if max_sim < 0:
                        sub_goal = None
                        self.observed_states = {}
                    elif max_sim < self.sim_threshold:
                        print("Minimum sim for observed states to match instruction not found, using best match instead.")
                        print("Best match sim value = ", max_sim)
                        for obs_state in self.observed_states:
                            str_state = self.observed_states[obs_state]
                            str_state_stacked = ' '.join(str_state)
                            t_state = self.enc.encode(str_state_stacked)
                            total_sim = 0
                            # Average sim across each sentence in instruction vs state
                            for instr_sentence in instruction_vector:
                                for state_sentence in t_state:
                                    total_sim+=self.cos(state_sentence, instr_sentence) 
                            sim = total_sim.item()/(len(instruction_vector)*len(t_state))
                            if sim > max_sim*(self.sim_threshold):
                                sub_goal = obs_state # Temp Sub-Goal as most known similar
                                sub_goal_list.append(sub_goal)

            # Log matching sub_goal with instruction        
            if instruction not in self.instruction_results:
                self.instruction_results[instruction] = {}    
                self.instruction_results[instruction]['env_code'] = {} 
                self.instruction_results[instruction]['action_cap'] = action_cap
                
            if (agent_type+'_'+adapter) not in self.instruction_results[instruction]['env_code']:
                self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter] = {}
                
            self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['sub_goal'] = sub_goal_list
            self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['sim_score'] = max_sim
            # --------------------------------------------------------------------------------
        return self.observed_states, self.instruction_results 