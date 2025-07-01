import os
import torch
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
import string
# ------ Interaction Protocol -----------------------------------
from elsciRL.interaction_loops.standard import StandardInteractionLoop
from elsciRL.interaction_loops.state_search import episode_loop
from joblib import Parallel, delayed, cpu_count
# ------ Agent Imports -----------------------------------------
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.DQN import DQNAgent
from elsciRL.agents.random_agent import RandomAgent
# --------------------------------------------------------------
# TODO use encoder defined by config not manual import
from elsciRL.encoders.language_transformers.MiniLM_L6v2 import LanguageEncoder

# Only use Qlearntab for for search agent for speeeeeed and randomn exploration
AGENT_TYPES = {
    "Qlearntab": TableQLearningAgent
}



class elsciRLSearch:
    def __init__(self, Config:dict, LocalConfig:dict, 
                 Engine, Adapters:dict,
                 save_dir:str, 
                 number_exploration_episodes:int = 10000,
                 match_sim_threshold:float=0.9,
                 observed_states:dict = None, observed_states_encoded:Tensor = None,
                 context_length:int=1000):
        
        # ----- Configs
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
        
        self.save_dir = save_dir
        # Create save directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Unsupervised search parameters
        self.agent_type: str = "Qlearntab" # fix to one type for fast search
        self.epsilon: float = 1 # fully random search agent
        self.number_exploration_episodes: int = number_exploration_episodes

        # Unsupervised search parameters
        self.context_length = context_length # Limit length of observed states
        self.enc = LanguageEncoder()
        self.sim_threshold: float = match_sim_threshold
        self.cos = torch.nn.CosineSimilarity(dim=1)  

        # Initialize observed states
        if (observed_states is not None) and (len(observed_states) > 0):
            self.observed_states = observed_states
        else:
            self.observed_states = {}
        # Encode all observed states 
        if (observed_states_encoded is not None) and (len(observed_states_encoded) > 0):
            self.str_states_encoded = observed_states_encoded
        else:
            if self.observed_states is None:
                self.str_states_encoded = None
            else:
                if len(self.observed_states) > 0:
                    print("\n -------------------------------\n")
                    str_states = [str_state[:self.context_length] for str_state in self.observed_states.values()]
                    print(f"Encoding {len(str_states)} observed states in {int(len(str_states)/256)} batches of size {256}.")
                    self.str_states_encoded = self.enc.encode(state=str_states, progress_bar=True)
        

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
        # Parallel processing for faster episode runs
        parallel = Parallel(n_jobs=cpu_count(True), prefer="processes", verbose=0)
        number_parallel_batches = 10
        if self.number_exploration_episodes > number_parallel_batches:
            number_episodes_per_parallel = int(self.number_exploration_episodes / number_parallel_batches)
        else:
            number_episodes_per_parallel
        observed_state_output = parallel(delayed(episode_loop)(Engine=self.engine, Adapters=self.adapters, 
                                                                local_setup_info=train_setup_info, 
                                                                number_episodes=number_episodes_per_parallel,
                                                                batch_number=i) for i in tqdm(range(number_parallel_batches)))
        for batch in observed_state_output:
            self.observed_states.update(batch)

        train_setup_info['training_results'] = None
        train_setup_info['observed_states'] = self.observed_states
        train_setup_info['experience_sampling'] = None
        # Extract visited states from env
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
        self.agent_adapter = agent_type+'_'+adapter
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
            
        if self.str_states_encoded is None:
            str_states = [str_state[:self.context_length] for str_state in self.observed_states.values()]
            self.str_states_encoded = self.enc.encode(str_states)
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
            print(f"\nFinding match for instruction: {instruction}")
            
            if (type(instr_description)) == type(''):
                if (len(instr_description.split('.')) > 1):
                    instr_description = instr_description.split('.')
                    # Remove punctuation from description
                    instr_description = [s.translate(str.maketrans('', '', string.punctuation)) for s in instr_description]
                    # Remove empty strings from description
                    instr_description = [s.strip() for s in instr_description if s.strip()]
                    instr_description = ' '.join(instr_description)
                else:
                    instr_description = [instr_description.translate(str.maketrans('', '', string.punctuation)).strip()]
            # Create tensor vector of description
            instruction_vector = self.enc.encode(instr_description)
            if instruction_vector.size()[0] > 1:
                print(f"Instruction vector size: {instruction_vector.size()}")
                print(f"\n Instruction vector: {instr_description} \n")
            # Default fedeback layer - DEMO wont currently updated this 
            feedback_layer = torch.zeros(instruction_vector.size()).to(device)
            # EXPLORE TO FIND LOCATION OF SUB-GOAL
            sub_goal = None
            # ---------------------------
            if (instruction in self.instruction_results):
                if (self.agent_adapter) in self.instruction_results[instruction]:
                    # We use feedback layer even if sub_goal not a good match
                    feedback_layer = self.instruction_results[instruction][self.agent_adapter]['feedback_layer']
                    if (self.instruction_results[instruction][self.agent_adapter]['sim_score']>=self.sim_threshold):
                        sub_goal = self.instruction_results[instruction][self.agent_adapter]['sub_goal'][0]
                        sub_goal_list = self.instruction_results[instruction][self.agent_adapter]['sub_goal']
                        sim = self.instruction_results[instruction][self.agent_adapter]['sim_score']
                else:
                    self.instruction_results[instruction][self.agent_adapter] = {}
                self.instruction_results[instruction][self.agent_adapter]['count'] = self.instruction_results[instruction][self.agent_adapter]['count']+1
            else:
                self.instruction_results[instruction] = {}    
                self.instruction_results[instruction]['instr_description'] = instr_description
                self.instruction_results[instruction][self.agent_adapter] = {} 
                self.instruction_results[instruction][self.agent_adapter]['count'] = 1
                self.instruction_results[instruction][self.agent_adapter]['action_cap'] = action_cap
                self.instruction_results[instruction][self.agent_adapter]['feedback_layer'] = feedback_layer
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
                    for obs_state_idx,obs_state in tqdm(enumerate(self.observed_states), total=len(self.observed_states)):
                        #str_state = self.observed_states[obs_state][:self.context_length]
                        t_state = self.str_states_encoded[obs_state_idx] #self.enc.encode(str_state)
                        # ---
                        sim = self.cos(torch.add(t_state, feedback_layer), instruction_vector).mean()
                       
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
                    # TODO: IMPROVE RUNTIME HERE BY NOT RUNNING SEARCH TWICE
                    if max_sim < self.sim_threshold:
                        print(" --- Best match not above threshold, finding highest sim instead...")
                        sub_goal = sub_goal_max
                        #sub_goal_t = sub_goal_max_t                                    
                        # Find all states that have same sim as max
                        for obs_state_idx,obs_state in tqdm(enumerate(self.observed_states), total=len(self.observed_states)):
                            #str_state = self.observed_states[obs_state][:self.context_length]
                            t_state = self.str_states_encoded[obs_state_idx] #self.enc.encode(str_state)
                            # ---
                            sim = self.cos(torch.add(t_state, feedback_layer), instruction_vector).mean()
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

            if (self.agent_adapter) in self.instruction_results[instruction]:
                self.instruction_results[instruction][self.agent_adapter]['sub_goal'] = sub_goal_list
                self.instruction_results[instruction][self.agent_adapter]['sim_score'] = sim
                self.instruction_results[instruction][self.agent_adapter]['feedback_layer'] = feedback_layer
            else:
                self.instruction_results[instruction][self.agent_adapter] = {}
                self.instruction_results[instruction][self.agent_adapter]['count'] = 1
                self.instruction_results[instruction][self.agent_adapter]['action_cap'] = action_cap
                self.instruction_results[instruction][self.agent_adapter]['sub_goal'] = sub_goal_list
                self.instruction_results[instruction][self.agent_adapter]['sim_score'] = sim
                self.instruction_results[instruction][self.agent_adapter]['feedback_layer'] = feedback_layer

            best_match_results[instruction] = {}
            best_match_results[instruction]['sub_goal'] = self.observed_states[sub_goal]
            best_match_results[instruction]['best_match'] = sub_goal_max
        
        # Store data for feedback and plot
        self.instructions = instructions
        self.instr_descriptions = instr_descriptions
                       
        return best_match_results, self.instruction_results
    
    def feedback(self, feedback_type:str='positive', feedback_increment:float=0.5,
                 plot:bool=False, plot_save_dir:str=None):
        # Update feedback layer to based on validation
        # Get instruction vector
        instructions = self.instructions
        instr_descriptions = self.instr_descriptions

        sim_total_delta = 0
        for i,instr_description in enumerate(instr_descriptions):
            instruction = list(self.instruction_results.keys())[i]
            print(f"\nFinding match for instruction: {instruction}")
            
            if (type(instr_description)) == type(''):
                if (len(instr_description.split('.')) > 1):
                    instr_description = instr_description.split('.')
                    # Remove punctuation from description
                    instr_description = [s.translate(str.maketrans('', '', string.punctuation)) for s in instr_description]
                    # Remove empty strings from description
                    instr_description = [s.strip() for s in instr_description if s.strip()]
                    instr_description = ' '.join(instr_description)
                else:
                    instr_description = [instr_description.translate(str.maketrans('', '', string.punctuation)).strip()]
            # Create tensor vector of description
            instruction_vector = self.enc.encode(instr_description)
            # Get sub-goal vector
            sub_goal = self.instruction_results[instruction][self.agent_adapter]['sub_goal'][0]
            sub_goal_t = self.enc.encode(sub_goal.replace(".", "")) # We cannot parse multiple sentences from sub-goal for feedback layer update
            # Get feedback layer vector
            feedback_layer = self.instruction_results[instruction][self.agent_adapter]['feedback_layer']
            try:
                if feedback_type == 'positive':
                    # Positive feedback - add to feedback layer
                    feedback_layer = torch.add(feedback_layer, feedback_increment*(torch.sub(instruction_vector, sub_goal_t))) 
                elif feedback_type == 'negative':
                    feedback_layer = torch.sub(feedback_layer, feedback_increment*(torch.sub(instruction_vector, sub_goal_t)))
                else:
                    raise ValueError(f"Invalid feedback type: {feedback_type}")
            except:
                print(f"Error updating feedback layer for instruction: {instruction}")
                print(f"Feedback type: {feedback_type}")
                print(f"Feedback increment: {feedback_increment}")
                print(f"Instruction vector: {instruction_vector.shape} - {instruction_vector.size()}")
                print(f"Sub-goal vector: {sub_goal_t.shape} - {sub_goal_t.size()}")
                print(f"Feedback layer: {feedback_layer.shape} - {feedback_layer.size()}")
                
            self.instruction_results[instruction][self.agent_adapter]['feedback_layer'] = feedback_layer
            # Average sim across each sentence in instruction vs state
            sim = self.cos(torch.add(sub_goal_t, feedback_layer), instruction_vector).mean() # bounds: [0,1]       
            sim_delta = sim-self.instruction_results[instruction][self.agent_adapter]['sim_score']
            print(f"--- Change in sim results with {feedback_type} reinforcement of correct state match = {sim_delta:.2f}")
            print(f"--- New Sim Value = {sim:.2f}") 

            if plot:
                if plot_save_dir is not None:
                    self.file_name = self.save_dir + '/' + plot_save_dir + '/feedbackplot__' + instruction[:4] + '.png'
                else:
                    self.file_name = self.save_dir + '/feedbackplot__'  + instruction[:4] + '.png'

                print(f"--- Plotting feedback for instruction: {instruction} to {self.file_name}")
                return self.sim_plot(sim_delta, self.instruction_results[instruction][self.agent_adapter]['sim_score'], feedback_type)

            self.instruction_results[instruction][self.agent_adapter]['sim_score'] = sim
            sim_total_delta += sim_delta

        sim_delta_avg = sim_total_delta / len(instructions)
        return sim_delta_avg
    
    def sim_plot(self, sim_delta:float, sim:float, feedback_type:str='positive'):
        # Plot sim delta and sim value
        if type(sim) == torch.Tensor:
            sim = sim.item()

        # Create arbitrary 2D vectors for visualization
        instruction_vec = [1, 1]  # Unit vector along x-axis
        
        # Calculate predicted vector based on similarity (sim)
        angle = torch.arccos(torch.tensor(sim)) # Convert sim to angle
        pred_x = torch.cos(angle)
        pred_y = torch.sin(angle)
        pred_vec = [pred_x.item(), pred_y.item()]
        
        # Calculate updated vector after feedback
        if pred_x > instruction_vec[0]:
            new_angle = torch.arccos(torch.tensor(sim - sim_delta))
        else:
            new_angle = torch.arccos(torch.tensor(sim + sim_delta))
      
        new_x = torch.cos(new_angle)
        new_y = torch.sin(new_angle)
        new_vec = [new_x.item(), new_y.item()]

        # Plot vectors
        fig = plt.figure(figsize=(8,8))
        plt.quiver(0, 0, instruction_vec[0], instruction_vec[1], angles='xy', scale_units='xy', scale=1, color='black', label='Instruction')
        plt.quiver(0, 0, pred_vec[0], pred_vec[1], angles='xy', scale_units='xy', scale=1, color='grey', linestyle='dashed', facecolor='none', linewidth=2,
          width=0.0001, headwidth=300, headlength=500, label='Initial Prediction')
        if sim_delta >= 0:
            plt.quiver(0, 0, new_vec[0], new_vec[1], angles='xy', scale_units='xy', scale=1, color='blue', label='After Feedback')
        else:
            # For negative feedback, use a different color
            plt.quiver(0, 0, new_vec[0], new_vec[1], angles='xy', scale_units='xy', scale=1, color='red', label='After Feedback')
        
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Vector Similarity After {feedback_type.title()} Feedback\nUpdated Sim: {abs(sim+sim_delta):.2f}, Delta: {sim_delta:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.file_name, dpi=100)
        plt.close()

        return self.file_name