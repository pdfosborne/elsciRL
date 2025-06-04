import random
import numpy as np
import pandas as pd
import json
import pickle
import string

import torch
from torch import Tensor

import ollama
import urllib.request

import logging
from elsciRL.agents.agent_abstract import LLMAgentAbstract

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LLMAgent(LLMAgentAbstract):
    def __init__(self, epsilon:float=0.2, model_name: str = "llama3.2", system_prompt: str = None, context_length: int = 1000):
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
            "Your task is to analyze the state and select the most appropriate action "
            "from the available actions. Respond with a JSON object containing the "
            "selected action and a brief explanation."
        )
        
        # No need to instantiate ollama.Client; use ollama.chat directly
        
        # Store the model for save/load functionality
        self.model = {
            'model_name': model_name,
            'system_prompt': system_prompt
        }
        # Epsilon-greedy exploration parameter
        self.epsilon = epsilon
        self.epsilon_reset = epsilon

        # Diary is used to improve the LLM's decision making based on previous states, actions and rewards
        # TODO: RENAME THE TERM 'DIARY' TO SOMETHING BETTER BECAUSE LLM THINKS ITS WRITING AN ACTUAL DIARY
        self.diary_system_prompt = """
            You are a expected to track the experience of the agent over many episodes.
            You are given a state, action and reward.
            You need to write a summary of the experience based on the state, action and reward that allows the LLM to learn which actions are good and which are bad.
        """
        self.diary = ''
        self.diary_model_name = 'llama3.2'

        # Add system prompt to the LLM prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt + self.system_prompt
            self.diary_system_prompt = system_prompt + self.diary_system_prompt

        # Initial empty value for trajectory history used by LLM to create diary
        self.state_action_history_current = ''

    def _LLM_prompt_adjustment(self, state_action_history: str) -> str:
        """
        Adjust the prompt to be more suitable for the LLM based on states, actions and outcomes.
        """
    
        diary = f"""
                Previous log: {self.diary}

                State action history with rewards: {state_action_history}

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
        else:
            try:
                prompt = f"""Current state: {state}

                            You must select an action from the following list: {', '.join(legal_actions)}

                            Please select the most appropriate action and explain your reasoning.

                            You have access to a diary of previous states, actions and rewards. Use the diary to help you make the best decision: {self.diary}

                            Respond in JSON format with the following structure:
                            {{
                                "action": "selected_action",
                                "explanation": "brief explanation of why this action was chosen"
                            }}"""

                # Use ollama.chat with the correct message format
                messages = []
                if self.system_prompt:
                    messages.append({'role': 'system', 'content': self.system_prompt})
                messages.append({'role': 'user', 'content': prompt})

                response = ollama.chat(
                    model=self.model_name,
                    messages=messages[:self.manual_context_length]  # Limit context length
                )
                
                # Result not always ending content with brackets
                content = response['message']['content'].strip().replace('\n', '').replace('```', '').translate(str.maketrans('', '', string.punctuation))
                # Try to parse as JSON object
                content_split = content.split(',')
                for i in range(len(content_split)):
                    if 'action' in content_split[i]:
                        content_action = (content_split[i])
                        break
                action = content_action.split(':')[1].strip().replace('"', '')
                if (action in legal_actions) and (action is not None):
                    action = action
                else:
                    print("++++++++++++++++++++++++++++++++++++++++")
                    print("INVALID ACTION")
                    print("State:", state)
                    print("Action:", action)
                    print("Legal actions:", legal_actions)
                    print("++++++++++++++++++++++++++++++++++++++++")
                    logger.error(f"Action {action} not in legal actions: {legal_actions}")
                    action = random.choice(legal_actions)
            except Exception as e:
                action = random.choice(legal_actions)
                logger.error(f"Error getting LLM action: {e}, using random choice: {action}")
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
    
        if len(self.reward_history) > 100:
            # Calculate reward statistics over recent history
            reward_mean = np.mean(self.reward_history)
            reward_std = np.std(self.reward_history)
            
            # Define significant rewards as those outside 1 standard deviation from mean
            reward_threshold = reward_mean + reward_std
        else:
            # Dont allow it to learn until we know reward is significant
            # This will only be for the first 10 actions in the entire experiment
            reward_threshold = np.max(self.reward_history)

        if abs(r_p)>reward_threshold:
            print("\n SIGNIFICANT REWARD OBTAINED", r_p)
            current_diary = self._LLM_prompt_adjustment(state_action_history=self.state_action_history_current)
            self.state_action_history_current = ''
        else:
            current_outcome = 'You were positioned at' + state + ', after taking action ' + action_code + ' the outcome position was ' + next_state + ' with reward: ' + str(r_p) +'. '
            self.state_action_history_current+=current_outcome
            # Output current knowledge from trajectory until LLM is used
            current_diary = current_outcome 
        return current_diary
