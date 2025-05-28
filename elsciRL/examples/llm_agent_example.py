#!/usr/bin/env python3
"""
Example usage of the LLM Agent with Ollama model.

This script demonstrates how to initialize and use the LLM agent
for decision making in a simple environment.
"""

import sys
import os

# Add the parent directory to the path to import elsciRL modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from elsciRL.agents.LLM_Agent import LLMAgent


def main():
    print("=== LLM Agent Example ===")
    
    # Initialize the LLM agent
    # Make sure you have an Ollama model installed (e.g., 'llama2', 'mistral', 'codellama')
    agent = LLMAgent(
        parameter=0.1,  # This parameter can be used for future customization
        model_name="llama2",  # Change this to your preferred Ollama model
        temperature=0.7,  # Controls randomness (0.0 = deterministic, 1.0 = very random)
        system_prompt="You are a strategic decision maker. Analyze the situation carefully and choose the best action."
    )
    
    print(f"Initialized LLM Agent with model: {agent.model_name}")
    print(f"Temperature: {agent.temperature}")
    
    # Example scenarios
    scenarios = [
        {
            "state": "You are in a maze. There is a wall to the north, an open path to the east, and a treasure room to the south.",
            "legal_actions": ["move_east", "move_south", "stay"]
        },
        {
            "state": "You are in a game. Your health is low (20%), you have a healing potion, and there's an enemy nearby.",
            "legal_actions": ["attack", "use_potion", "flee", "defend"]
        },
        {
            "state": "You are trading stocks. The market is volatile, you own 100 shares, and the price just dropped 5%.",
            "legal_actions": ["buy_more", "sell_all", "sell_half", "hold"]
        }
    ]
    
    # Test the agent on different scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        print(f"State: {scenario['state']}")
        print(f"Available actions: {scenario['legal_actions']}")
        
        try:
            # Get action from the agent
            chosen_action = agent.policy(scenario['state'], scenario['legal_actions'])
            print(f"Agent chose: {chosen_action}")
            
            # Simulate learning (in a real environment, you'd get actual rewards)
            simulated_reward = 1.0 if chosen_action != "stay" else 0.0
            agent.learn(
                state=scenario['state'],  # In real use, this would be a Tensor
                next_state="next_state",  # In real use, this would be a Tensor  
                r_p=simulated_reward,
                action_code=chosen_action
            )
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is installed and running, and the specified model is available.")
            print("Install Ollama: https://ollama.ai/")
            print("Pull a model: ollama pull llama2")
    

    # Demonstrate save/load functionality
    print(f"\n--- Save/Load Example ---")
    saved_model = agent.save()
    print(f"Saved agent configuration: {saved_model}")
    
    # Create a new agent and load the saved configuration
    new_agent = LLMAgent(parameter=0.1)
    new_agent.load(saved_model)
    print(f"Loaded agent with model: {new_agent.model_name}")


if __name__ == "__main__":
    main() 