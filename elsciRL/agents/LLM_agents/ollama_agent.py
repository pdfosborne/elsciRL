import random
import numpy as np
import pandas as pd
import json
import pickle

import torch
from torch import Tensor

import ollama

import logging
from elsciRL.agents.agent_abstract import LLMAgentAbstract

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LLMAgent(LLMAgentAbstract):
    def __init__(self, epsilon:float=0.2, model_name: str = "llama2", system_prompt: str = None):
        """
        Initialize the Ollama LLM model for policy-based action selection.
        
        Args:
            model_name (str): Name of the Ollama model to use
            system_prompt (str, optional): System prompt to guide the model's behavior
        """
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
        self.diary = ""
        self.diary_model_name = 'llama3.2'

        # Add system prompt to the LLM prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt + self.system_prompt
            self.diary_system_prompt = system_prompt + self.diary_system_prompt
        

    def _LLM_prompt_adjustment(self, state: str, action: str, next_state: str, reward: str) -> str:
        """
        Adjust the prompt to be more suitable for the LLM based on states, actions and outcomes.
        """
    
        diary = f"""
                Previous diary: {self.diary}

                State: {state}
                Action: {action}
                Next state: {next_state}
                Reward: {reward}

                Please write a diary entry based on the prior diary knowledge, state, action and reward.
            """
        # Use ollama.chat with the correct message format
        messages = []
        messages.append({'role': 'system', 'content': self.diary_system_prompt})
        messages.append({'role': 'user', 'content': diary})

        response = ollama.chat(
            model=self.diary_model_name,
            messages=messages
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
                    messages=messages
                )
                
                # Result not always ending content with brackets
                content = response['message']['content'].strip().replace('\n', '').replace('```', '')
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
                    action = random.choice(legal_actions)
                    logger.error(f"Action {action} not in legal actions: {legal_actions}")
            except Exception as e:
                action = random.choice(legal_actions)
                logger.error(f"Error getting LLM action: {e}, using random choice: {action}")
        return action

    # We now break agent into a policy choice, action is taken in game_env then next state is used in learning function
    def learn(self, state: str, next_state: str, r_p: float, action_code: str) -> str:
        """Given action is taken, agent learns from outcome (i.e. next state), states and actions must be text strings for LLM input."""

        current_diary = self._LLM_prompt_adjustment(state, action_code, next_state, r_p)
        return current_diary
