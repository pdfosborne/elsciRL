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
    def __init__(self, model_name: str = "llama2", temperature: float = 0.7, system_prompt: str = None):
        """
        Initialize the Ollama LLM model for policy-based action selection.
        
        Args:
            model_name (str): Name of the Ollama model to use
            temperature (float): Temperature for sampling (higher = more random)
            system_prompt (str, optional): System prompt to guide the model's behavior
        """
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are an AI agent that takes actions based on the current state. "
            "Your task is to analyze the state and select the most appropriate action "
            "from the available actions. Respond with a JSON object containing the "
            "selected action and a brief explanation."
        )
        
        self.llm_model = ollama.Client(
            model_name=model_name,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        # Store the model for save/load functionality
        self.model = {
            'model_name': model_name,
            'temperature': temperature,
            'system_prompt': system_prompt
        }


    def save(self) -> dict:
        return self.model
    
    def load(self, saved_agent: dict = {}):
        if saved_agent:
            self.model = saved_agent
            self.model_name = saved_agent.get('model_name', 'llama2')
            self.temperature = saved_agent.get('temperature', 0.7)
            system_prompt = saved_agent.get('system_prompt')
            
            # Reinitialize the LLM model with saved parameters
            # TODO: Have this save and load the actual model weights
            self.llm_model = ollama.Client(
                model_name=self.model_name,
                temperature=self.temperature,
                system_prompt=system_prompt
            )

    def exploration_parameter_reset(self):
        # For LLM agents, we could reset temperature to original value
        self.temperature = self.model.get('temperature', 0.7)
        self.llm_model.temperature = self.temperature

    def clone(self):
        return self.model

    # Fixed order of variables
    def policy(self, state: str, legal_actions: list[str]) -> str:
        """Agent's decision making for next action based on current knowledge and policy type"""
        # Initialise the agent_id entry if not seen
        
        try:
            # Use the LLM model to get action recommendation
            """
            Get the next action based on the current state and available actions.
            
            Args:
                state (str): Text description of the current state
                available_actions (List[str]): List of possible actions to choose from
                
            Returns:
                Dict[str, Any]: Dictionary containing the selected action and explanation
            """
            # Construct the prompt
            prompt = f"""Current state: {state}

                        Available actions: {', '.join(legal_actions)}

                        Please select the most appropriate action and explain your reasoning.
                        Respond in JSON format with the following structure:
                        {{
                            "action": "selected_action",
                            "explanation": "brief explanation of why this action was chosen"
                        }}"""

            # Get response from Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                system=self.system_prompt,
                temperature=self.temperature
            )

            try:
                # Parse the response as JSON
                result = json.loads(response['response'])
                return result
            except json.JSONDecodeError:
                # Fallback in case the response isn't valid JSON
                return {
                    "action": legal_actions[0],  # Default to first action
                    "explanation": "Failed to parse model response as JSON"
                }
            
            # Extract the action from the response
            if isinstance(llm_response, dict) and 'action' in llm_response:
                suggested_action = llm_response['action']
                
                # Validate that the suggested action is in legal_actions
                if suggested_action in legal_actions:
                    action = suggested_action
                    logger.info(f"LLM selected action: {action}")
                    if 'explanation' in llm_response:
                        logger.info(f"LLM reasoning: {llm_response['explanation']}")
                else:
                    # Fallback to random choice if LLM suggests invalid action
                    action = random.choice(legal_actions)
                    logger.warning(f"LLM suggested invalid action '{suggested_action}', using random choice: {action}")
            else:
                # Fallback to random choice if response format is unexpected
                action = random.choice(legal_actions)
                logger.warning(f"Unexpected LLM response format, using random choice: {action}")
                
        except Exception as e:
            # Fallback to random choice in case of any errors
            action = random.choice(legal_actions)
            logger.error(f"Error getting LLM action: {e}, using random choice: {action}")

        return action

    # We now break agent into a policy choice, action is taken in game_env then next state is used in learning function
    def learn(self, state: Tensor, next_state: Tensor, r_p: float, action_code: str) -> float:
        """Given action is taken, agent learns from outcome (i.e. next state)"""
        
        # For future implementation: we could use the reward signal to update the LLM policy
        # This could involve techniques like RLHF (Reinforcement Learning from Human Feedback)
        # or storing experiences for later fine-tuning
        
        # For now, we can log the experience for potential future use
        logger.debug(f"LLM Agent experience: state={state}, action={action_code}, reward={r_p}")
        
        return None
