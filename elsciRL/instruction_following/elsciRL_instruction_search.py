import os
from typing import List
import json
import torch
import time

import matplotlib.pyplot as plt
# ------ Interaction Protocol -----------------------------------
from elsciRL.interaction_loops.standard import StandardInteractionLoop
# ------ Experiment Import --------------------------------------
from elsciRL.evaluation.standard_report import Evaluation
# ------ Evaluation Metrics -----------------------------------------
from elsciRL.analysis.convergence_measure import Convergence_Measure
# ------ Instruction Inputs -----------------------------------------
from elsciRL.instruction_following.elsciRL_instr_input import elsciRLInput as TerminalInput
# ------ Agent Imports -----------------------------------------
# Universal Agents
from elsciRL.agents.agent_abstract import Agent, QLearningAgent
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.neural_q_agent import NeuralQLearningAgent
from elsciRL.agents.random_agent import RandomAgent

# TODO: Enable any number of the same agent types with varying parameters
AGENT_TYPES = {
    "Qlearntab": TableQLearningAgent,
    "Neural_Q": NeuralQLearningAgent,
    "Neural_Q_2": NeuralQLearningAgent,
    "Neural_Q_language": NeuralQLearningAgent,
    "Random": RandomAgent
}

PLAYER_PARAMS = {
    "Qlearntab": ["alpha", "gamma", "epsilon"],
    "Neural_Q": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "Neural_Q_2": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "Neural_Q_language": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "Random":[]
}

# TODO use encoder defined by config not manual import
from elsciRL.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

# This is the main run functions for elsciRL to be imported
# Defines the train/test operators and imports all the required agents and experiment functions ready to be used
# The local main.py file defines the [adapters, configs, environment] to be input

# This should be where the environment is initialized and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead

class elsciRLSearch:
    def __init__(self, Config:dict, LocalConfig:dict, 
                 Engine, Adapters:dict,
                 save_dir:str, 
                 num_plans:int=1, number_exploration_episodes:int = 100, sim_threshold:float = 0.9,
                 feedback_increment=0.5, feedback_repeats:int=1,
                 observed_states:dict=None, instruction_results:dict=None,
                 instruction_input_interface:str='terminal'):
        self.ExperimentConfig = Config
        self.LocalConfig = LocalConfig

        self.engine = Engine
        self.adapters = Adapters
        self.env = StandardInteractionLoop
        self.setup_info:dict = self.ExperimentConfig['data'] | self.LocalConfig['data']  
        self.training_setups: dict = {}

        save_dir_extra = save_dir.split("/")[-1]
        save_dir = '/'.join(save_dir.split("/")[:-1])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir+'/'+save_dir_extra
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

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
        self.number_exploration_episodes: int = number_exploration_episodes
        self.sim_threshold: float = sim_threshold
        self.cos = torch.nn.CosineSimilarity(dim=0)
        # elsciRL inits
        self.num_plans = num_plans
        #self.feedback_layer_form: tensor = torch.zeros(self.enc.encode(['']).size()) # Init base sizing of tensor produced by language encoder        
        self.feedback_increment: float = feedback_increment
        self.feedback_results: dict = {}
        self.feedback_repeats: int = feedback_repeats
        # Instruction input
        self.instruction_input_interface = instruction_input_interface
        if self.instruction_input_interface.lower() == 'terminal':
            self.elsciRL_input = TerminalInput()
        else:
            print("NOTE: Instruction input not defined, default set to terminal.")
            self.elsciRL_input = TerminalInput()


    def search(self, action_cap:int=5, re_search_override:bool=False, simulated_instr_goal:any=None):
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        # Trigger re-search
        if re_search_override:
            self.observed_states:dict = {}
        
        #instruction_vector = torch.rand(observed_states[list(observed_states.keys())[0]].size()) # NEED TO FIND A METHOD TO VECTORIZE OBSERVED STATES AND INSTRUCTIONS TO COMPARE
        # - TODO: Only use first given agent to do exploration - repeating search for multiple agents doesn't make sense
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
            # Set env function to training# Repeat training
            train_setup_info['train'] = True
            # --- 
            # Set exploration parameters
            train_setup_info['number_training_episodes'] = self.number_exploration_episodes # Override
            
            # ---------------------------------elsciRL-----------------------------------------
            # Train on Live system for limited number of total episodes
            train_setup_info['training_results'] = False
            if not self.observed_states:
                train_setup_info['observed_states'] = False
            else:
                train_setup_info['observed_states'] = self.observed_states
            train_setup_info['experience_sampling'] = False
            train_setup_info['live_env'] = True 
            # ---------------------------
            # Init environment to define current position
            for p in range(0, self.num_plans):
                print("---")
                print("Plan Number = ", p)
                sample_env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=train_setup_info)
                print("Environment Init Position: ", sample_env.start_obs)
                # New: user input here
                instructions, instr_descriptions = self.elsciRL_input.user_input()
                
                # ---------------------------
                feedback_count = 0
                for repeat in range(0,self.feedback_repeats): # Arbitrary repeat to further reinforce matching state to instr
                    print("===")
                    print("REINFORCEMENT Repeated search num ", repeat+1)
                    for i,instr in enumerate(instructions):
                        
                        if i == 0:
                            instruction = str(sample_env.start_obs).split(".")[0] + "---" + instr
                        else:
                            instruction = instructions[i-1] + "---" + instr
                        
                        instr_description = instr_descriptions[i]
                        
                        if type(instr_description) == type(''):
                            instr_description = instr_description.split('.')
                            instr_description = list(filter(None, instr_description))
                        # Create tensor vector of description
                        instruction_vector = self.enc.encode(instr_description)
                        print("Size of Encoded Instruction Tensor")
                        print(instruction_vector.size())
                        # EXPLORE TO FIND LOCATION OF SUB-GOAL
                        sub_goal = None
                        # ---------------------------
                        # Overrides on prior knowledge
                        # Seen sub_goal before & sim above threshold
                        if (instruction in self.instruction_results):
                            if (agent_type+'_'+adapter) in self.instruction_results[instruction]:
                                # We use feedback layer even if sub_goal not a good match
                                feedback_layer = self.instruction_results[instruction][agent_type+'_'+adapter]['feedback_layer']
                                if (self.instruction_results[instruction][agent_type+'_'+adapter]['sim_score']>=self.sim_threshold):
                                    sub_goal = self.instruction_results[instruction][agent_type+'_'+adapter]['sub_goal'][0]
                                    sub_goal_list = self.instruction_results[instruction][agent_type+'_'+adapter]['sub_goal']
                                    sim = self.instruction_results[instruction][agent_type+'_'+adapter]['sim_score']
                            else:
                                self.instruction_results[instruction][agent_type+'_'+adapter] = {}
                                feedback_layer = torch.zeros(instruction_vector.size()).to(device)
                            self.instruction_results[instruction][agent_type+'_'+adapter]['count'] = self.instruction_results[instruction][agent_type+'_'+adapter]['count']+1
                        else:
                            self.instruction_results[instruction] = {}    
                            self.instruction_results[instruction][agent_type+'_'+adapter] = {} 
                            self.instruction_results[instruction][agent_type+'_'+adapter]['count'] = 1
                            self.instruction_results[instruction][agent_type+'_'+adapter]['action_cap'] = action_cap
                            feedback_layer = torch.zeros(instruction_vector.size()).to(device)
                            self.instruction_results[instruction][agent_type+'_'+adapter]['feedback_layer'] = feedback_layer
                        # ---------------------------
                        search_count=0
                        while not sub_goal:
                            sim_delta = 1 # Parameter that stops updates when change to feedback is small
                            # If no description -> no sub-goal (i.e. envs terminal goal position)
                            if not instr_description: 
                                sub_goal = None
                                # If no sub-goal -> find best match of description from env 
                            else:
                                search_count+=1
                                print("------") 
                                print("Search: ", search_count)
                                
                                # Only run on live env if observed states empty
                                if not self.observed_states:
                                    train_setup_info['sub_goal'] = sub_goal # None
                                    # ---
                                    # Explore env with limited episodes
                                    # Environment now init here and called directly in experimental setup loop
                                    # - setup elsciRL info
                                    # Train on Live system for limited number of total episodes
                                    train_setup_info['live_env'] = True                        
                                    live_env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=train_setup_info)
                                    explore_results = live_env.episode_loop()
                                    train_setup_info['training_results'] = explore_results
                                    train_setup_info['observed_states'] = live_env.elsciRL.observed_states
                                    train_setup_info['experience_sampling'] = live_env.elsciRL.experience_sampling
                                    # Extract visited states from env
                                    self.observed_states = live_env.elsciRL.observed_states

                                # Compare to instruction vector                            
                                max_sim = -1
                                # all states that are above threshold 
                                sub_goal_list = []
                                for obs_state in self.observed_states:
                                    str_state = self.observed_states[obs_state]
                                    #str_state_stacked = ' '.join(str_state)
                                    t_state = self.enc.encode(str_state)
                                    # ---
                                    total_sim = 0
                                    dim_count = 0 # For some reason encoder here is adding extra dimension                                        
                                    for idx,instr_sentence in enumerate(instruction_vector):
                                        feedback_layer_sent = feedback_layer[idx]
                                        for state_sentence in t_state:
                                            total_sim+=self.cos(torch.add(state_sentence, feedback_layer_sent), instr_sentence)
                                            dim_count+=1
                                    #print("Observation: ", str_state)
                                    #print("Tensor check: ", t_state.size(), " --- ", instruction_vector.size(), " --- ", feedback_layer.size())
                                    #print("Sim result: ", total_sim.item(), '---', dim_count)
                                    
                                    sim = 0 if dim_count==0 else total_sim.item()/dim_count
                                    if sim > max_sim:
                                        max_sim  = sim
                                        sub_goal_max = obs_state
                                        sub_goal_max_t = t_state
                                    if sim >= self.sim_threshold:
                                        sub_goal = obs_state # Sub-Goal code
                                        sub_goal_t = t_state
                                        sub_goal_list.append(sub_goal)
                
                                # OR if none above threshold matching max sim
                                if max_sim < self.sim_threshold:
                                    sub_goal = sub_goal_max
                                    sub_goal_t = sub_goal_max_t                                    
                                    # Find all states that have same sim as max
                                    for obs_state in self.observed_states:
                                        str_state = self.observed_states[obs_state]
                                        #str_state_stacked = ' '.join(str_state)
                                        t_state = self.enc.encode(str_state)
                                        # ---
                                        total_sim = 0
                                        # Average sim across each sentence in instruction vs state
                                        dim_count = 0
                                        for idx,instr_sentence in enumerate(instruction_vector):
                                            feedback_layer_sent = feedback_layer[idx]
                                            for state_sentence in t_state:
                                                total_sim+=self.cos(torch.add(state_sentence, feedback_layer_sent), instr_sentence)
                                                dim_count+=1
                                        sim = 0 if dim_count==0 else total_sim.item()/dim_count
                                        if sim >= (max_sim):
                                            sub_goal_list.append(obs_state)

                                if max_sim < self.sim_threshold:
                                    print("Minimum sim for observed states to match instruction not found, using best match instead. Best match sim value = ", max_sim )
                            # If adapter is poor to match to instruction vector none of them observed states match
                            if (max_sim<-1)|(sub_goal is None):#|(max_sim>1):
                                print("All observed states result in similarity outside bounds (i.e. strongly opposite vectors to instruction). Re-starting Search.")
                                print("Max sim found = ", max_sim)
                                sub_goal = None
                                self.observed_states = {}
                            elif (sim_delta<0.05)|(search_count>=3):
                                print("Change in sim less than or equal to delta cap or search cap reached, assume goal-state not observed. Re-starting Search.")
                                if simulated_instr_goal:
                                    print("-- Known sub_goal position: ", simulated_instr_goal)
                                print("-- Best match: ", sub_goal_max)
                                sub_goal = None
                                self.observed_states = {}
                                search_count = 0
                            else:
                                if not simulated_instr_goal:
                                    
                                    for s_g in sub_goal_list:
                                        print(self.observed_states[s_g], ": ", s_g)
                                    print(" ")
                                    print("===========")
                                    print("Instruction: ", instr)
                                    print(sub_goal)
                                    print("Best match state for instruction \n ______ \n Adapted form: \n\t - ", self.observed_states[sub_goal], " \n Engine observation: \n\t - " , sub_goal)
                                    print("Full list of matching states...",)
                                    print("")
                                    print("Tensor check: Size of Instruction vs Best match: ", instruction_vector.size(), " --- ", sub_goal_t.size())
                                    print("")
                                    feedback = input("-- Does this match the expectation instruction outcome? (Y/N)")
                                    feedback_count+=1 
                                else:
                                    if type(simulated_instr_goal[0]) != type(sub_goal_max):
                                        print("- ERROR: Typing of simulated sub-goal check does not match typing of state from environment, please correct this.")
                                        print(type(simulated_instr_goal[0])," - ", type(sub_goal_max))
                                        #print("Observed States Examples: ", self.observed_states(list(self.observed_states.keys())[0:5]))
                                        exit()
                                    else:
                                        if sub_goal_max in simulated_instr_goal:
                                            print("- Simulated sub-goal match found.")
                                            print("-- Known sub_goal position: ", simulated_instr_goal)
                                            print("-- Unsupervised state best match: ", sub_goal_max)
                                            feedback = 'Y'
                                        else:
                                            print("- Match NOT found.")
                                            print("-- Known sub_goal position: ", simulated_instr_goal)
                                            print("-- Unsupervised state best match: ", sub_goal_max)
                                            feedback = 'N'
                                
                                if (feedback.lower() == 'y')|(feedback.lower() == 'yes'):
                                    for idx,instr_sentence in enumerate(instruction_vector):
                                        feedback_layer_sent = feedback_layer[idx]
                                        for sentence in sub_goal_t:
                                            feedback_layer[idx] = torch.add(feedback_layer_sent, self.feedback_increment*(torch.sub(instr_sentence, sentence))) 
                                    total_sim = 0
                                    # Average sim across each sentence in instruction vs state
                                    dim_count = 0
                                    for idx,instr_sentence in enumerate(instruction_vector):
                                        feedback_layer_sent = feedback_layer[idx]
                                        for state_sentence in t_state:
                                            total_sim+=self.cos(torch.add(state_sentence, feedback_layer_sent), instr_sentence)
                                            dim_count+=1
                                    sim = 0 if dim_count==0 else total_sim.item()/dim_count
                                    sim_delta = sim-max_sim
                                    print("--- Change in sim results with POSITIVE reinforcement of correct state match =", sim_delta)
                                    print("--- New Sim Value = ",  sim)
                                else:
                                    for idx,instr_sentence in enumerate(instruction_vector):
                                        feedback_layer_sent = feedback_layer[idx]
                                        for sentence in sub_goal_t:
                                            feedback_layer[idx] = torch.sub(feedback_layer_sent, self.feedback_increment*(torch.sub(instr_sentence, sentence)))
                                    total_sim = 0
                                    # Average sim across each sentence in instruction vs state
                                    dim_count = 0
                                    for idx,instr_sentence in enumerate(instruction_vector):
                                        feedback_layer_sent = feedback_layer[idx]
                                        for state_sentence in t_state:
                                            total_sim+=self.cos(torch.add(state_sentence, feedback_layer_sent), instr_sentence)
                                            dim_count+=1
                                    sim = 0 if dim_count==0 else total_sim.item()/dim_count
                                    sim_delta = sim-max_sim
                                    print("--- Change in sim results with NEGATIVE reinforcement for NO MATCH =", sim_delta)
                                    sub_goal = None
                        if (agent_type+'_'+adapter) not in self.instruction_results[instruction]:
                            self.instruction_results[instruction][agent_type+'_'+adapter] = {}   
                        self.instruction_results[instruction][agent_type+'_'+adapter]['sub_goal'] = sub_goal_list
                        self.instruction_results[instruction][agent_type+'_'+adapter]['sim_score'] = sim
                        self.instruction_results[instruction][agent_type+'_'+adapter]['feedback_layer'] = feedback_layer
                # log results of feedback loop
                if (agent_type+'_'+adapter) not in self.feedback_results:
                    self.feedback_results[agent_type+'_'+adapter] = {}
            
                if p not in self.feedback_results[(agent_type+'_'+adapter)]:
                    self.feedback_results[(agent_type+'_'+adapter)][p] = {}
                    self.feedback_results[(agent_type+'_'+adapter)][p]['instr_count'] = i+1
                    self.feedback_results[(agent_type+'_'+adapter)][p]['feedback_count'] = feedback_count
                else:
                    self.feedback_results[(agent_type+'_'+adapter)][p]['feedback_count'] = feedback_count + self.feedback_results[(agent_type+'_'+adapter)][p]['feedback_count']
        # --------------------------------------------------------------------------------
        # Quick Visual Analysis of Feedback results
        goal = str(sample_env.start_obs).split(".")[0] + '_GOAL'
         
        plan_numbers = []
        feedback_required = []
        for plan_num in self.feedback_results[agent_type+'_'+adapter]:
            plan_numbers.append(int(plan_num))
            instr_count = self.feedback_results[agent_type+'_'+adapter][plan_num]['instr_count']
            feedback_count = self.feedback_results[agent_type+'_'+adapter][plan_num]['feedback_count']
            feedback_avg = feedback_count/instr_count
            feedback_required.append(feedback_avg)
        plt.plot(plan_numbers,feedback_required, label='Num Searches')
        plt.title('Amount of Feedback for each Instruction by Plan')
        plt.ylabel('Feedback Needed per Instr')
        plt.xlabel('Plan Number')
        plt.tight_layout()
        plt.savefig(self.save_dir+'/reinforcement_results.png', dpi=100)
        plt.show()
        plt.close()
        
        
        # Save instruction inputs
        instruction_predictions = {}
        for instr in self.instruction_results:
            
            if instr not in instruction_predictions:
                instruction_predictions[instr] = {}
            for n, agent_type in enumerate(self.setup_info['agent_select']):
                adapter = self.setup_info["adapter_select"][n]
                
                if (agent_type+'_'+adapter) not in instruction_predictions:
                    instruction_predictions[instr][(agent_type+'_'+adapter)] = {}

                instruction_predictions[instr][(agent_type+'_'+adapter)]['count'] = self.instruction_results[instr][(agent_type+'_'+adapter)]['count']
                instruction_predictions[instr][(agent_type+'_'+adapter)]['sim_score'] = self.instruction_results[instr][(agent_type+'_'+adapter)]['sim_score']
                instruction_predictions[instr][(agent_type+'_'+adapter)]['sub_goal'] = self.instruction_results[instr][(agent_type+'_'+adapter)]['sub_goal']



        with open(self.save_dir+'/instruction_predictions.json', 'w') as f:
            json.dump(instruction_predictions, f)
        
        print("-- Complete Instruction Results --")
        print(self.instruction_results)

        return self.observed_states, self.instruction_results