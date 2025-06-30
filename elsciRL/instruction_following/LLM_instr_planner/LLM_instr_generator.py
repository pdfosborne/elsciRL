import ollama
import json
import re
from typing import List, Dict, Optional, Union
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaTaskBreakdown:
    """
    Ollama LLM chat class for breaking down complex tasks into manageable sub-goals.
    """
    
    def __init__(self, model_name: str = "llama3.2", context_length: int = 4000, host: str = "localhost:11434",
                 input_prompt: str = None, observed_states: dict = None):
        """
        Initialize the Ollama chat client.
        
        Args:
            model_name: Name of the Ollama model to use
            context_length: Maximum number of messages to keep in context
            host: Ollama server host and port
        """
        self.model_name = model_name.strip().lower()
        self.host = host
        self.client = ollama.Client(host=host)
        self.conversation_history = []
        self.context_length = context_length
        
        # Verify model availability
        try:
            self._verify_model()
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise

        # Set input prompt if provided
        if input_prompt is not None:
            logger.info(f"Using input prompt: {input_prompt[:self.context_length]}...")  # Log first 100 chars
            self.input_prompt = input_prompt

        # Get randomly sampled observed states if provided to use for prompt output structure
        if observed_states is not None:
            print("Using observed states for prompt output structure.")
            keys = random.sample(list(observed_states), min(10, len(observed_states)))
            self.observed_states_prompt = str([observed_states[k] for k in keys])
    
    def _verify_model(self):
        """Verify that the specified model is available."""
        try:
            models = self.client.list()
            available_models = [model['model'] for model in models['models']]
            available_models_short = [model['model'].split(':')[0] for model in models['models']]
            if (self.model_name.strip not in available_models) and (self.model_name not in available_models_short):
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
            else:
                logger.info(f"Model {self.model_name} is available.")
        except Exception as e:
            logger.error(f"Error verifying model: {e}")
            raise
    
    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """
        Send a chat message to the Ollama model.
        
        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt to set context
            
        Returns:
            Model response as string
        """
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add current user message
            messages.append({
                'role': 'user',
                'content': prompt
            })
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages[:self.context_length]
            )
            
            assistant_response = response['message']['content']
            
            # Update conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': prompt
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': assistant_response
            })
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    def break_down_task(self, task_prompt: str, max_subgoals: int = 5) -> List[str]:
        """
        Break down a complex task into sub-goals using the LLM.
        
        Args:
            task_prompt: The main task to break down
            max_subgoals: Maximum number of sub-goals to generate
            
        Returns:
            List of sub-goal strings
        """
        system_prompt = f"""You are an expert task planner. Your job is to break down complex tasks into smaller, manageable sub-goals.
Your instructions will be used to help an agent that interacts with the environment.
You do not interact with the environment directly, but your output will be used to plan actions.
DO NOT return any steps that are not actionable, exclude any 'gather information' 'generate' 'think', 'assess', 'look around', 'determine', 'research', 'analyze', 'review', 'explore', 'investigate', 'identify', 'understand', 'examine', 'evaluate', 'plan', 'decide'.
You must generate actionable steps based on the task description provided that an agent can complete.
INSTRUCTIONS:
1. Analyze the given task and break it down into {max_subgoals} or fewer concrete sub-goals
2. Each sub-goal should be specific, actionable, and measurable
3. List each sub-goal on a separate line, numbered (1., 2., 3., etc.)
4. Focus on creating logical, sequential sub-goals that when completed will achieve the main objective
5. Do not include any additional commentary or explanations, just the sub-goals

Example format:
1. First sub-goal description
2. Second sub-goal description
3. Third sub-goal description"""
        
        # If input prompt is provided, include it in the system prompt
        if hasattr(self, 'input_prompt'):
            system_prompt += f"\n\nThe problem's data source are provided by the following code, use this as guidance on how to act: {self.input_prompt}"

        # If observed states are provided, include them in the system prompt
        if hasattr(self, 'observed_states_prompt'):
            system_prompt += f"\n\nMatch your output language form to the observed language from the environment, here are some examples of the language structure:\n{self.observed_states_prompt}"

        user_prompt = f"""Task to break down: {task_prompt}

Please analyze this task and break it down into manageable sub-goals. Consider:
- What are the key components needed?
- What is the logical sequence of steps?
- Sub-goals must be defined sequentially?

Provide your response as a numbered list of sub-goals."""

        try:
            response = self.chat(user_prompt, system_prompt)
            return self._parse_subgoals_response(response, max_subgoals)
            
        except Exception as e:
            logger.error(f"Error breaking down task: {e}")
            raise
    
    def _parse_subgoals_response(self, response: str, max_subgoal: int = None) -> List[str]:
        """
        Parse the LLM response and extract sub-goals.
        
        Args:
            response: Raw LLM response containing numbered list
            
        Returns:
            List of sub-goal strings
        """
        try:
            sub_goals = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for numbered items (1., 2., etc.) or bullet points
                if re.match(r'^\d+\.', line) or line.startswith('- ') or line.startswith('• '):
                    # Remove the numbering/bullet and clean up
                    clean_goal = re.sub(r'^\d+\.\s*', '', line)
                    clean_goal = re.sub(r'^[-•]\s*', '', clean_goal)
                    if clean_goal:
                        sub_goals.append(clean_goal.strip())
            
            # If no numbered items found, try to split by sentences or lines
            if not sub_goals:
                potential_goals = [line.strip() for line in lines if line.strip()]
                sub_goals = [goal for goal in potential_goals if len(goal) > 10]  # Filter out very short lines

            # Filter out non-actionable sub-goals
            actionable_sub_goals = []
            for goal in sub_goals:
                # Lowercase for easier matching
                goal_lower = goal.lower()
                # Filter out goals that are not actionable
                if any(word in goal_lower for word in ['think', 'consider', 'assess', 'look around', 'determine', 'research', 'analyze', 'review', 'explore', 'investigate', 'identify', 'understand', 'examine', 'evaluate', 'plan', 'decide']):
                    continue
                actionable_sub_goals.append(goal)

            # If we have more or fewer than max_subgoal, use LLM to summarise into exactly max_subgoal actionable steps
            summarise_prompt = (
                f"Given the following actionable sub-goals:\n"
                + "\n".join([f"- {g}" for g in actionable_sub_goals])
                + f"\n\nSummarise and combine them into exactly {max_subgoal} clear, actionable, and non-overlapping steps. "
"The steps are to interact with the environment and the final output used to plan, not to think, refine or plan or as inputs for a model so do not include text like 'generate' or 'produce'."
"Do not include any steps about thinking, assessing, looking around, creating a plan, or planning. "
"Return only a numbered list of actionable steps."
"Do not include any additional commentary or explanations, just the sub-goals. "
"Do not include any text that is not a sub-goal, such as 'Here are the sub-goals:' or 'The sub-goals are:' or 'These are the steps to take:' or 'Write an instruction:'. "
            )
            try:
                summary_response = self.chat(summarise_prompt)
                actionable_sub_goals = []
                lines = summary_response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if re.match(r'^\d+\.', line):
                        clean_goal = re.sub(r'^\d+\.\s*', '', line)
                        if clean_goal:
                            actionable_sub_goals.append(clean_goal.strip())
                # If still not exactly max_subgoal, trim or pad
                if len(actionable_sub_goals) > max_subgoal:
                    actionable_sub_goals = actionable_sub_goals[:max_subgoal]
                elif len(actionable_sub_goals) < max_subgoal:
                    # Pad with generic actionable step if needed
                    actionable_sub_goals += [f"Complete the task step {i+1}" for i in range(len(actionable_sub_goals), max_subgoal)]
            except Exception as e:
                logger.error(f"Error summarising sub-goals with LLM: {e}")
                # Fallback: trim or pad
                if len(actionable_sub_goals) > max_subgoal:
                    actionable_sub_goals = actionable_sub_goals[:max_subgoal]
                elif len(actionable_sub_goals) < max_subgoal:
                    actionable_sub_goals += [f"Complete the task step {i+1}" for i in range(len(actionable_sub_goals), max_subgoal)]

            sub_goals = actionable_sub_goals

            # NOTE: HARD LIMIT ON NUMBER OF SUB-GOALS
            if max_subgoal is None or max_subgoal > len(sub_goals):
                max_subgoal = len(sub_goals)
            sub_goals = sub_goals[:max_subgoal]
            
            return sub_goals if sub_goals else [f"Complete the task: {response[:self.context_length]}..."]
            
        except Exception as e:
            logger.error(f"Error parsing sub-goals response: {e}")
            logger.error(f"Raw response: {response}")
            # Return a fallback sub-goal
            return [f"Complete the task: {response[:self.context_length]}..."]
    
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Initialize the task breakdown system
        task_breaker = OllamaTaskBreakdown(model_name="llama3.2")
        
        # Example task
        example_task = "Build a machine learning model to predict customer churn for an e-commerce website"
        
        print(f"Breaking down task: {example_task}")
        print("-" * 50)
        
        # Break down the task
        sub_goals = task_breaker.break_down_task(example_task, max_subgoals=5)
        
        # Display the sub-goals
        print("Sub-goals:")
        for i, goal in enumerate(sub_goals, 1):
            print(f"{i}. {goal}")
        
    except Exception as e:
        print(f"Error in example: {e}")
        print("Make sure Ollama is running and the specified model is available.")
