import random
import numpy as np
import pandas as pd
import json
import pickle
import torch
# LLM libraries
import ollama

import logging
from elsciRL.agents.agent_abstract import LLMAgentAbstract
from elsciRL.encoders.language_transformers.MiniLM_L6v2 import LanguageEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LLMAgent(LLMAgentAbstract):
    def __init__(self, epsilon:float=0.2, epsilon_step:float=0.01, model_name: str = "llama3.2", 
                 system_prompt: str = None, context_length: int = 1000, previous_state_action_history_length: int = 10,
                 action_language_mapping: bool = False):
        """
        Initialize the Ollama LLM model for policy-based action selection.
        
        Args:
            model_name (str): Name of the Ollama model to use
            system_prompt (str, optional): System prompt to guide the model's behavior
        """
        # Check if model exists locally, if not pull it
        # List available models
        models = ollama.list()
        model_exists = any(model['model'] == model_name for model in models['models'])

        self.encoder = LanguageEncoder()
        self.cos = torch.nn.CosineSimilarity(dim=0) 
        
        # Set of actions that result in null outcome and so random is taken instead (all lowercase)
        self.null_actions = ['none', 'non', 'null', 'no', 'n', 'na', 'n/a']
        if not model_exists:
            logger.info(f"Model {model_name} not found locally. Please check your model name and try again.")
            print(f"\n ---- \n Available models: {[model['model'] for model in models['models']]}")
        
        # TODO: REMOVED FOR NOW AND INSTEAD JUST CUTTING OF INPUT LENGTH TO SPECIFIED PARAMETER
        # Import all modelfiles from modelfiles directory
        # try:
        #     # TODO: PULLING FROM GITHUB AS LOCAL REFERENCE WOULD REQUIRE USER TO HAVE MODELFILE DEFINED IN THEIR DIRECTORY
        #     # Create llama3.2 model from modelfile
        #     model_name_modelfile = model_name.replace('.', '-')
        #     # ollama.create() doesn't support remote URLs directly
        #     # Need to download the modelfile first
        #     try:
        #         modelfile_url = 'https://raw.githubusercontent.com/pdfosborne/elsciRL/main/elsciRL/agents/LLM_agents/agent_modelfiles/'+model_name_modelfile+'.modelfile'
        #         modelfile_content = urllib.request.urlopen(modelfile_url).read().decode('utf-8')
        #         ollama.create(model=model_name_modelfile, modelfile=modelfile_content)
        #     except Exception as e:
        #         logger.error(f"Error downloading or creating model from modelfile: {e}")
        #     logger.info("Successfully created "+model_name_modelfile+" model from modelfile")
        # except Exception as e:
        #     logger.error(f"Error getting modelfile for model {model_name} from https://github.com/pdfosborne/elsciRL/tree/main/elsciRL/agents/LLM_agents/agent_modelfiles {e}")
        #     logger.info("Using default model instead")
        
        self.manual_context_length = context_length  # Limit context length for LLM to avoid exceeding token limits

        self.model_name = model_name
        self.system_prompt = (
            "You are an AI agent that takes actions based on the current state. "
            "Your task is to analyze the state and select the most appropriate action from the available actions. "
            "Respond with a JSON object containing the selected action and a brief explanation."
        )
        
        # No need to instantiate ollama.Client; use ollama.chat directly
        
        # Store the model for save/load functionality
        self.model = {
            'model_name': model_name,
            'system_prompt': system_prompt
        }
        # Epsilon-greedy exploration parameter
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.epsilon_reset = epsilon
        self.previous_state_action_history_length = previous_state_action_history_length

        # Diary is used to improve the LLM's decision making based on previous states, actions and rewards
        # TODO: RENAME THE TERM 'DIARY' TO SOMETHING BETTER BECAUSE LLM THINKS ITS WRITING AN ACTUAL DIARY
        self.diary_system_prompt = """
            You are a expected to track the experience of the agent over many episodes.
            You are given a state, action and reward.
            You need to write a summary of the experience based on the state, action and reward that allows the LLM to learn which actions are good and which are bad.
            ONLY select the most appropriate action, do not provide any reasoning, do not provide any explanation, do not provide any additional text, just the action.
            DO NOT RETURN THE ACTION 'THINK' OR 'THINKING' OR 'THINK ABOUT IT' OR 'THINKING ABOUT IT' OR 'THINKING ABOUT THE NEXT ACTION' OR 'THINKING ABOUT THE NEXT MOVE' OR 'THINKING ABOUT THE NEXT STEP' OR 'THINKING ABOUT THE NEXT DECISION' OR 'THINKING ABOUT THE NEXT CHOICE' OR 'THINKING ABOUT THE NEXT OPTION'.

            Respond in JSON format with the following structure:
            {{
                "action": "selected_action",
                "explanation": "brief explanation of the action taken"
            }}
        """
        self.diary = ''
        self.diary_model_name = 'llama3.2'

        # Add system prompt to the LLM prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt + self.system_prompt
            self.diary_system_prompt = system_prompt + self.diary_system_prompt

        # Initial empty value for trajectory history used by LLM to create diary
        self.state_action_history_current = ''

        self.action_language_mapping = action_language_mapping
        self.action_language_mapping_encoder = {}
        self.action_language_mapping_decoder = {}
        self.action_language_mapping_reset = True
        self.action_language_mapping_system_prompt = """
            You are a language mapping agent that maps a single action to a language that the LLM can understand.
            You are given a list of actions and you need to map each action to a language that the LLM can understand.
            The language should be a single phrase that is a common language that the LLM can understand.
            You need to return a JSON object with the following structure:
            {{
                "action": "action",
                "language": "language"
            }}
        """
        self.action_language_mapping_system_prompt += f"Here is some information on the problem: {self.system_prompt}"

    def _LLM_prompt_adjustment(self, state_action_history: str) -> str:
        """
        Adjust the prompt to be more suitable for the LLM based on states, actions and outcomes.
        """
    
        diary = f"""
                Previous log: {self.diary}

                State action history with rewards: {str(state_action_history.split('.')[-self.previous_state_action_history_length:])}

                Please write or update the log entry based on the knowledge obtained from the states, actions and rewards to maximise the reward obtained.
            """
        # Use ollama.chat with the correct message format
        messages = []
        messages.append({'role': 'system', 'content': self.diary_system_prompt})
        messages.append({'role': 'user', 'content': diary})

        response = ollama.chat(
            model=self.diary_model_name,
            messages=messages[:self.manual_context_length]  # Limit context length
        )

        self.diary = response['message']['content']

        return self.diary
    
    def _action_language_mapping(self, state:str, legal_actions:list[str], state_action_history:str) -> str:
        """
        Map the action to a language that the LLM can understand.
        """
        
        # Use previous state action history if available and long enough
        if (state_action_history) and (len(state_action_history.split('.')) >= self.previous_state_action_history_length):
            action_mapping_system_prompt = self.action_language_mapping_system_prompt + f"""Here is the a recent history of states, actions and resulting outcome states, use this to understand the problem:
                {str(state_action_history.split('.')[-self.previous_state_action_history_length:])}"""
            # Reset knowledge after first time it is available
            if self.action_language_mapping_reset:
                print(f"----> Using previous state action history, action language mapping knowledge reset.")
                self.action_language_mapping_reset = False
                self.action_language_mapping_encoder = {}
                self.action_language_mapping_decoder = {}
        else:
            action_mapping_system_prompt = self.action_language_mapping_system_prompt
        # ------------------------------------------------------------
        legal_actions_mapped = []
        for action in legal_actions:
            if action not in self.action_language_mapping_encoder:
                # Convert action to language
                action_mapping = ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'system', 'content': action_mapping_system_prompt},
                            {'role': 'user', 'content': str(action)}]
                )['message']['content']
                try:
                    # Convert to JSON object
                    action_mapping = json.loads(action_mapping)
                    action_mapping = action_mapping['language']
                except:
                    pass
                action_mapping = action_mapping.split('\n')[-1].strip()  # Take last line of response
                # Use original action in mapping as fallback incase mapping is poor
                action_mapping = 'Original action code was [' + str(action) + '], mapped to language is [' + action_mapping + ']'
                # Store action mapping
                self.action_language_mapping_encoder[action] = action_mapping
                # Decoder used to convert language back to action
                self.action_language_mapping_decoder[action_mapping] = action
                print(f"----> Mapped action {action} to language {action_mapping}")
                legal_actions_mapped.append(action_mapping)
            else:
                # If action already mapped, use existing mapping
                legal_actions_mapped.append(self.action_language_mapping_encoder[action])

        return legal_actions_mapped

    def save(self) -> dict:
        return self.model
    
    def load(self, saved_agent: dict = {}):
        if saved_agent:
            self.model = saved_agent
            self.model_name = saved_agent.get('model_name', 'llama2')
            self.system_prompt = saved_agent.get('system_prompt')
            # No need to re-instantiate a client

    def exploration_parameter_reset(self):
        self.epsilon = self.epsilon_reset
        

    def clone(self):
        clone = pickle.loads(pickle.dumps(self))
        clone.epsilon = self.epsilon_reset
        return clone

    # Fixed order of variables
    def policy(self, state: str, legal_actions: list[str]) -> str:
        """Agent's decision making for next action based on current knowledge and policy type"""

        # Epsilon-greedy action selection to encourage exploration
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
            logger.info(f"Epsilon-greedy: Random action selected: {action}")   
            if self.epsilon > 0:
                self.epsilon = self.epsilon - (self.epsilon*self.epsilon_step) # Added epsilon step reduction to smooth exploration to greedy
                if self.epsilon < 0:
                    self.epsilon = 0         
        else:
            if self.action_language_mapping:
                print(f"----> Action language mapping enabled, mapping actions to language.")
                legal_actions = self._action_language_mapping(state, legal_actions, self.state_action_history_current)

            prompt = f"""
                        Current state: {state}
                        
                        You must select an action from the following list: {str(legal_actions)}

                        You have access to information on previous states, actions and rewards. Use this information to help you make the best decision: {self.diary}
"""

            # Use ollama.chat with the correct message format
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})
            messages.append({'role': 'user', 'content': prompt[:self.manual_context_length]})

            if len(prompt) > self.manual_context_length:
                print("\n WARNING: Context length exceeded, cutting off input to fit within model limits.")
                print(" Context Length:", len(prompt))
                print(" Model Limit:", self.manual_context_length)
            response = ollama.chat(
                model=self.model_name,
                messages=messages  # Limit context length
            )
            
            # Result not always ending content with brackets
            # - limit output length to save time, use end of content as action is usually given at the end
            action = response['message']['content'].split('\n')[-1].strip()
            print("\n ----------------------------------------------------")
            # Validate Action against Legal Actions
            if (action in legal_actions) and (action is not None):
                print(f"LLM response content: {action} in LEGAL ACTIONS")
                action = action
            else:
                print(f"Action {action} not in legal actions, trying to find best match.")
                # 1. Check if action is a cross-reference to legal actions
                cross_reference = False
                for term in action.split(' '):
                    if term.strip() in legal_actions:
                        action = term.strip()
                        cross_reference = True
                        break

                if cross_reference is False:
                    # 2. CHECK IF LEGAL ACTION IN ACTION TEXT
                    action_match_found = False
                    for legal_a in legal_actions:
                        if str(legal_a) in action:
                            print(f"Action {action} found in legal moves as {legal_a}.")
                            action = legal_a
                            action_match_found = True
                            break

                # 3. IF NOT EITHER, FIND BEST MATCH OVER THRESHOLD
                if (cross_reference is False) and (action_match_found is False):
                    # Encode the proposed action and legal actions using miniLMv6
                    if (len(action)>0) and (action.lower() not in self.null_actions):
                        action_embedding = self.encoder.encode(action)[0]
                        legal_action_embeddings = [self.encoder.encode(str(a))[0] for a in legal_actions]
                        
                        # Calculate cosine similarity between proposed action and legal actions
                        best_match_sim = 0
                        best_match_idx = 0
                        for idx, legal_emb in enumerate(legal_action_embeddings):
                            sim = self.cos(action_embedding,legal_emb).item()
                            if sim > best_match_sim:
                                best_match_idx = idx
                                best_match_sim = sim
                                            
                        if best_match_sim > 0.6:
                            # Accept the closest matching legal action
                            action = legal_actions[best_match_idx]
                            logger.info(f"Found similar legal action: {action} with similarity {best_match_sim}")
                        else:
                            print(f"Best match {legal_actions[best_match_idx]} not similar enough to action {action} with sim score {best_match_sim}.")
                            # No close match found, use random action
                            action = random.choice(legal_actions)
                            logger.warning(f"No similar legal actions found, using random choice: {action}")
                    else:
                        print(f"Null action taken: {action} - using random instead.")
                        action = random.choice(legal_actions)
            
            # Convert action back to language if action language mapping is enabled
            if self.action_language_mapping:
                action = self.action_language_mapping_decoder[action]

        return action

    # We now break agent into a policy choice, action is taken in game_env then next state is used in learning function
    def learn(self, state: str, next_state: str, r_p: float, action_code: str) -> str:
        """Given action is taken, agent learns from outcome (i.e. next state), states and actions must be text strings for LLM input."""

        # Collect outcomes in current path until a significant reward found, then use trajectory to this reward to update the knowledge
        # - Much lower costs compared to calling LLM after every step

        # Track reward history in a circular buffer
        if not hasattr(self, '_reward_array'):
            self._reward_array = np.zeros(1000)  # Pre-allocate fixed size array
            self._reward_idx = 0
        self._reward_array[self._reward_idx] = r_p
        self._reward_idx = (self._reward_idx + 1) % 1000  # Circular buffer
        self.reward_history = self._reward_array[:self._reward_idx] if self._reward_idx > 0 else self._reward_array  # View into array
    
        # Set fixed threshold for reward importance
        if abs(r_p)<=0.1:
            reward_threshold = np.max(self.reward_history)+100
        elif len(self.reward_history) > 100:
            # Calculate reward statistics over recent history
            reward_mean = np.mean(self.reward_history)
            reward_std = np.std(self.reward_history)
            
            # Define significant rewards as those outside 1 standard deviation from mean
            reward_threshold = reward_mean + reward_std
        else:
            # Dont allow it to learn until we know reward is significant
            # This will only be for the first 10 actions in the entire experiment
            reward_threshold = np.max(self.reward_history)+100

        if abs(r_p)>reward_threshold:
            print("\n SIGNIFICANT REWARD OBTAINED", r_p)
            current_diary = self._LLM_prompt_adjustment(state_action_history=self.state_action_history_current)
            self.state_action_history_current = ''
        else:
            current_outcome = 'You were positioned at' + str(state) + ', after taking action ' + str(action_code) + ' the outcome position was ' + str(next_state) + ' with reward: ' + str(r_p) +'. '
            self.state_action_history_current+=current_outcome
            # Output current knowledge from trajectory until LLM is used
            current_diary = current_outcome 
        return current_diary
