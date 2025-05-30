import random
import numpy as np
import pandas as pd
import json

import torch
from torch import Tensor

import ollama

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LLMAgent:
    def __init__(self, epsilon:float=0.2, model_name: str = "llama2", system_prompt: str = None):
        """
        Initialize the Ollama LLM model for policy-based action selection.
        
        Args:
            model_name (str): Name of the Ollama model to use
            system_prompt (str, optional): System prompt to guide the model's behavior
        """
        self.model_name = model_name
        self.system_prompt = system_prompt or (
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


    def save(self) -> dict:
        return self.model
    
    def load(self, saved_agent: dict = {}):
        if saved_agent:
            self.model = saved_agent
            self.model_name = saved_agent.get('model_name', 'llama2')
            self.system_prompt = saved_agent.get('system_prompt')
            # No need to re-instantiate a client

    def exploration_parameter_reset(self):
        # No client to update
        pass

    def clone(self):
        return self.model

    # Fixed order of variables
    def policy(self, state: str, legal_actions: list[str]) -> str:
        """Agent's decision making for next action based on current knowledge and policy type"""

        # Epsilon-greedy action selection to encourage exploration
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
            logger.info(f"Epsilon-greedy: Random action selected: {action}")
            return action
        else:
            try:
                prompt = f"""Current state: {state}

                            Available actions: {', '.join(legal_actions)}

                            Please select the most appropriate action and explain your reasoning.
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
                    messages=messages
                )
                
                try:
                    # Result not always ending content with brackets
                    content = response['message']['content'].strip().replace('\n', '').replace('```', '')
                    # Try to parse as JSON object
                    content_split = content.split(',')
                    for i in range(len(content_split)):
                        if 'action' in content_split[i]:
                            content_action = (content_split[i])
                            break
                    action = content_action.split(':')[1].strip().replace('"', '')
                    print("Action:", action)
                    return action
                
                except Exception as e:
                    logger.error(f"Error parsing LLM response: {e}, response content does no contain action in required for 'action:action,...': {response['message']['content']}")
                    return random.choice(legal_actions),
        
            except Exception as e:
                action = random.choice(legal_actions)
                logger.error(f"Error getting LLM action: {e}, using random choice: {action}")
                return action

    # We now break agent into a policy choice, action is taken in game_env then next state is used in learning function
    def learn(self, state: Tensor, next_state: Tensor, r_p: float, action_code: str) -> float:
        """Given action is taken, agent learns from outcome (i.e. next state)"""
        
        return None
