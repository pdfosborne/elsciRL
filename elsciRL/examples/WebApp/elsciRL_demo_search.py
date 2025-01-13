import os
import torch
# ------ Interaction Protocol -----------------------------------
from elsciRL.interaction_loops.standard import StandardInteractionLoop
# ------ Agent Imports -----------------------------------------
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.neural_q_agent import NeuralQLearningAgent
from elsciRL.agents.random_agent import RandomAgent
# --------------------------------------------------------------
# TODO use encoder defined by config not manual import
from elsciRL.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

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

class elsciRLSearch:
    def __init__(self, Config:dict, LocalConfig:dict, 
                 Engine, Adapters:dict,
                 save_dir:str, 
                 number_exploration_episodes:int = 10000,
                 match_sim_threshold:float=0.9,
                 observed_states:dict = {}):
        
        # ----- Configs
        self.observed_states = observed_states
        # Meta parameters
        self.ExperimentConfig = Config
        # Local Parameters
        self.ProblemConfig = LocalConfig
        
        self.engine = Engine
        self.adapters = Adapters
        self.env = StandardInteractionLoop
        self.setup_info:dict = self.ExperimentConfig | self.ProblemConfig  
        self.training_setups: dict = {}
        self.instruction_results:dict = {}

        save_dir_extra = save_dir.split("/")[-1]
        save_dir = '/'.join(save_dir.split("/")[:-1])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir+'/'+save_dir_extra
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir) 

        # Unsupervised search parameters
        self.agent_type: str = "Qlearntab" # fix to one type for fast search
        self.epsilon: float = 1 # fully random search agent
        self.number_exploration_episodes: int = number_exploration_episodes

        # Unsupervised search parameters
        self.enc = LanguageEncoder()
        self.sim_threshold: float = match_sim_threshold
        self.cos = torch.nn.CosineSimilarity(dim=0)  
        

    def search(self, action_cap:int=100):
        # Fixed to Tabular Q learning agent for now
        agent_type = self.agent_type
        # We are adding then overriding some inputs for exploration setups
        train_setup_info = self.setup_info.copy()
        # Override action cap for shorter term sub-goals for faster learning
        train_setup_info['training_action_cap'] = action_cap 
        # ----- State Adapter Choice
        for adapter in train_setup_info["adapter_select"]:
            if (("language" in adapter.lower()) |
                ("lang" in adapter.lower()) |
                ("_l" in adapter.lower())|
                ("l_" in adapter.lower())):
                train_setup_info["adapter_select"] = adapter
                break

        #adapter = train_setup_info["adapter_select"][0]
        # ----- Agent parameters
        agent_parameters = train_setup_info["agent_parameters"][agent_type]
        train_setup_info['agent_type'] = agent_type
        train_setup_info['agent_name'] = (str(agent_type) + '_' +
                                        str(adapter) + '_' +
                                        str(agent_parameters))
        train_setup_info['adapter_select'] = adapter
        # ----- init agent
        player = AGENT_TYPES[agent_type](**agent_parameters)
        train_setup_info['agent'] = player
        # -----
        # Set env function to training
        train_setup_info['train'] = True
        # --- 
        # Set exploration parameters
        train_setup_info['number_training_episodes'] = self.number_exploration_episodes 
        train_setup_info['epsilon'] = self.epsilon 
        # ---------------------------------elsciRL-----------------------------------------
        # Train on Live system for limited number of total episodes
        train_setup_info['training_results'] = False
        train_setup_info['observed_states'] = False
        train_setup_info['experience_sampling'] = False
        train_setup_info['live_env'] = True 
        # ---------------------------
        # Init environment to define current position
        train_setup_info['sub_goal'] = None 
        sample_env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=train_setup_info)
        print("Environment Init Position: ", sample_env.start_obs)
        # ---
        # Explore env with limited episodes
        # Environment now init here and called directly in experimental setup loop
        # - setup elsciRL info
        # Train on Live system for limited number of total episodes
        explore_results = sample_env.episode_loop()
        train_setup_info['training_results'] = explore_results
        train_setup_info['observed_states'] = sample_env.elsciRL.observed_states
        train_setup_info['experience_sampling'] = sample_env.elsciRL.experience_sampling
        # Extract visited states from env
        self.observed_states = sample_env.elsciRL.observed_states
        # ---------------------------
        
        return self.observed_states
    
    def match(self, action_cap:int=5, instructions:list=[''], instr_descriptions:list=['']):
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        # Fixed to Tabular Q learning agent for now
        agent_type = self.agent_type
        # We are adding then overriding some inputs for exploration setups
        train_setup_info = self.setup_info.copy()
        # Override action cap for shorter term sub-goals for faster learning
        train_setup_info['training_action_cap'] = action_cap 
        # ----- State Adapter Choice
        adapter = train_setup_info["adapter_select"][0]
        # ----- Agent parameters
        agent_parameters = train_setup_info["agent_parameters"][agent_type]
        train_setup_info['agent_type'] = agent_type
        train_setup_info['agent_name'] = (str(agent_type) + '_' +
                                        str(adapter) + '_' +
                                        str(agent_parameters))
        train_setup_info['adapter_select'] = adapter
        # ----- init agent
        player = AGENT_TYPES[agent_type](**agent_parameters)
        train_setup_info['agent'] = player
        # -----
        # Set env function to training
        train_setup_info['train'] = True
        # ---------------------------------elsciRL-----------------------------------------
        # Train on Live system for limited number of total episodes
        train_setup_info['training_results'] = False
        train_setup_info['observed_states'] = False
        train_setup_info['experience_sampling'] = False
        train_setup_info['live_env'] = True 
        # ---------------------------
        # Init environment to define current position
        train_setup_info['sub_goal'] = None 
        sample_env = self.env(Engine=self.engine, Adapters=self.adapters, local_setup_info=train_setup_info)
            
        # New: user input here
        #instructions, instr_descriptions = self.elsciRL_input.user_input()
        # DEMO SETS THIS AS FUNCTION INPUT
        # ---------------------------
        best_match_results = {}
        for i,instr in enumerate(instructions):
            if i == 0:  
                # env start is '0' so force rest to start from '1' instead otherwise 
                # breaks uniqueness requirement for instructions
                instruction = str(sample_env.start_obs).split(".")[0] + "---" + str(int(instr)+1)
            else:
                instruction = str(int(instructions[i-1])+1) + "---" + str(int(instr)+1)
            instr_description = instr_descriptions[i]
            
            if type(instr_description) == type(''):
                instr_description = instr_description.split('.')
                instr_description = list(filter(None, instr_description))
            # Create tensor vector of description
            instruction_vector = self.enc.encode(instr_description)
            # Default fedeback layer - DEMO wont currently updated this 
            feedback_layer = torch.zeros(instruction_vector.size()).to(device)
            # EXPLORE TO FIND LOCATION OF SUB-GOAL
            sub_goal = None
            # ---------------------------
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
            while not sub_goal:
                # If no description -> no sub-goal (i.e. envs terminal goal position)
                if not instr_description: 
                    sub_goal = None
                    # If no sub-goal -> find best match of description from env 
                else:
                    # Compare to instruction vector                            
                    max_sim = -1
                    # all states that are above threshold 
                    sub_goal_list = []
                    for obs_state in self.observed_states:
                        str_state = self.observed_states[obs_state]
                        t_state = self.enc.encode(str_state)
                        # ---
                        total_sim = 0
                        dim_count = 0 # For some reason encoder here is adding extra dimension                                        
                        for idx,instr_sentence in enumerate(instruction_vector):
                            feedback_layer_sent = feedback_layer[idx]
                            for state_sentence in t_state:
                                total_sim+=self.cos(torch.add(state_sentence, feedback_layer_sent), instr_sentence)
                                dim_count+=1
                       
                        sim = 0 if dim_count==0 else total_sim.item()/dim_count
                        if sim > max_sim:
                            max_sim  = sim
                            sub_goal_max = obs_state
                            #sub_goal_max_t = t_state
                        if sim >= self.sim_threshold:
                            sub_goal = obs_state # Sub-Goal code
                            sub_goal_max = obs_state
                            #sub_goal_t = t_state
                            sub_goal_list.append(sub_goal)
    
                    # OR if none above threshold matching max sim
                    if max_sim < self.sim_threshold:
                        sub_goal = sub_goal_max
                        #sub_goal_t = sub_goal_max_t                                    
                        # Find all states that have same sim as max
                        for obs_state in self.observed_states:
                            str_state = self.observed_states[obs_state]
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
                    print("All observed states result in similarity outside bounds (i.e. not found).")
                    print("Max sim found = ", max_sim)
                    sub_goal_list = ['']
                    break

            if (agent_type+'_'+adapter) in self.instruction_results[instruction]:
                self.instruction_results[instruction][agent_type+'_'+adapter]['sub_goal'] = sub_goal_list
                self.instruction_results[instruction][agent_type+'_'+adapter]['sim_score'] = sim
                self.instruction_results[instruction][agent_type+'_'+adapter]['feedback_layer'] = feedback_layer
            else:
                self.instruction_results[instruction][agent_type+'_'+adapter] = {}
                self.instruction_results[instruction][agent_type+'_'+adapter]['count'] = 1
                self.instruction_results[instruction][agent_type+'_'+adapter]['action_cap'] = action_cap
                self.instruction_results[instruction][agent_type+'_'+adapter]['sub_goal'] = sub_goal_list
                self.instruction_results[instruction][agent_type+'_'+adapter]['sim_score'] = sim
                self.instruction_results[instruction][agent_type+'_'+adapter]['feedback_layer'] = feedback_layer

            best_match_results[instruction] = {}
            best_match_results[instruction]['sub_goal'] = self.observed_states[sub_goal]
            best_match_results[instruction]['best_match'] = sub_goal_max
                       
        return best_match_results, self.instruction_results