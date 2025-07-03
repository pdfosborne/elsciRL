from abc import ABC, abstractmethod
from typing import List, Any
import os
import numpy as np
from torch import Tensor

from elsciRL.adapters.LLM_state_generators.base_prompt import elsciRL_base_prompt

from elsciRL.encoders.language_transformers.MiniLM_L6v2 import LanguageEncoder as MiniLM_L6v2


try:
    import ollama
except ImportError:
    print("Warning: ollama library not found. Please install it with: pip install ollama")
    ollama = None

ENCODERS = {
    "MiniLM_L6v2": MiniLM_L6v2
}

class LLMAdapter(ABC):
    """Convert a general prompt and raw text state into a description of the state."""
    def __init__(self, base_prompt:str=None):
        super().__init__()
        # Define the fields that describe the state features:
        if base_prompt:
            self.base_prompt = elsciRL_base_prompt + base_prompt
        else:
            self.base_prompt = elsciRL_base_prompt

    @abstractmethod
    def _read(raw_state) -> list:
        # Read the data.
        # fill in the feature fields
        raise NotImplementedError


class OllamaAdapter(LLMAdapter):
    """Adapter for local Ollama LLM models."""
    
    def __init__(self, base_prompt:str=None, model_name:str="llama3.2", 
                 context_length:int = 1000,
                 action_history_length:int=10, encoder:str='MiniLM_L6v2'):
        print("Using OllamaAdapter with model:", model_name)
        super().__init__(base_prompt)
        self.model_name = model_name
        self.manual_context_length = context_length

        if ollama is None:
            raise ImportError("ollama library is required. Install it with: pip install ollama")
        
        # Initialize the language encoder for encoding functionality
        self.encoder = ENCODERS[encoder]()
        
        # Set the action history length, lower reduces context size and computation time
        self.action_history_length = action_history_length
        self.state_history = []
        self.prior_action_count = -1  # Track the number of actions taken in the current episode

        self.cache = {}  # Cache for storing previous responses
        self.encoder_cache = {}  # Cache for storing encoded states
        
    
    def _read(self, raw_state) -> list:
        """Read the data and fill in the feature fields."""
        # This method should be implemented based on specific requirements
        # For now, returning the raw state as a list
        return [raw_state] if isinstance(raw_state, str) else raw_state
    
    def call_ollama_api(self, prompt: str):
        """Call the local Ollama API with the given prompt."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': self.base_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt[:self.manual_context_length]
                    }
                ]
            )
            return response
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return None

    def process_ollama_response(self, response):
        """Process the response from Ollama API."""
        if response and 'message' in response:
            output_response = response['message']['content']
            return output_response
        return None

    def adapter(self, state: any, legal_moves: list = [], episode_action_history: list = [], encode: bool = True, indexed: bool = False) -> Tensor:
        """Returns the adapted form, may require input flag for encoded or non-encoded output.
        
        Args:
            state: The current state of the environment.
            legal_moves: A list of legal moves for the agent.
            episode_action_history: A list of actions taken in the current episode.
            encode: A flag indicating whether to encode the response.
            indexed: A flag indicating whether to index the response.

        States and actions are typically numeric and so need the LLM to describe them in a way that is useful for the agent.

        In order to convert numeric information to text we track recent states and actions as a means to establish a context for the current state.
        """
        # Check cache for previously processed state
        if str(state) in self.cache:
            processed_response = self.cache[str(state)]
            if encode:
                encoded_response = self.encoder_cache.get(str(state), None)
                if encoded_response is not None:
                    # If we have a cached response but not an encoded one, encode it
                    encoded_response = self.encoder.encode(
                        state=processed_response,
                        legal_actions=legal_moves,
                        episode_action_history=episode_action_history,
                        indexed=indexed
                    )
                    self.encoder_cache[str(state)] = encoded_response
            return encoded_response if encode else processed_response
        # -----------------------------------------------
        else:
            # Build the full context prompt including legal moves and action history
            context_parts = []
            # Add state information
            if state:
                # We can call the adapter multiple times for the same state so we only add the state
                # to history if it is new
                if len(episode_action_history) > self.prior_action_count:
                    self.state_history.append(str(state))
                    self.prior_action_count = len(episode_action_history)
            
                context_parts.append(f"The current state to describe is: {str(state)}")
            # Add legal moves if provided
            # TODO: ADD ACTION MAPPER TO ALL ADAPTERS AS STANDARD FUNCTIONALITY
            if len(legal_moves) > 0:
                context_parts.append(f"Legal moves: {str(legal_moves)}")
            
            # Add action history if provided
            if len(episode_action_history) > 0:
                context_parts.append("The following is a history of the states and actions taken in the current episode.")
                recent_actions = episode_action_history[-self.action_history_length:]  # Last N actions
                for n,prior_action in enumerate(recent_actions):
                    try:
                        context_parts.append(f"Prior state {n}: {str(self.state_history[len(self.state_history)-len(recent_actions)-1+n])}")
                        context_parts.append(f"Action {n}: {str(prior_action)}")
                    except:
                        print("\nError accessing state history or action history.\n")
                        print(episode_action_history)
                        print("----------------------\n")
                        print(recent_actions)
            
            # Combine all context into a single prompt
            full_prompt = " | ".join(context_parts)
            
            # Get LLM response
            adapted_state = self.call_ollama_api(full_prompt)
            processed_response = self.process_ollama_response(adapted_state)
            
            if processed_response is None:
                processed_response = str(state) if state is not None else "No state available"
        
            # Cache the processed response
            if str(state) not in self.cache:
                self.cache[str(state)] = processed_response

            # Handle encoding
            if encode:
                if self.encoder is not None:
                    # Use the LanguageEncoder to encode the response
                    state_encoded = self.encoder.encode(
                        state=processed_response,
                        legal_actions=legal_moves,
                        episode_action_history=episode_action_history,
                        indexed=indexed
                    )
                    # Cache the encoded state
                    if str(state) not in self.encoder_cache:
                        self.encoder_cache[str(state)] = state_encoded
                    return state_encoded
                else:
                    print("Warning: Encoder not available, returning processed response as string")
                    return processed_response
            else:
                return processed_response
        
    def sample(self, state: any=None):
        """Returns a sample of an adapted state form (typically initial position of the environment).
           A generic blackjack exmaple is used if no state is provided."""
        if not state:
            state_player = np.random.randint(0, 11, size=(1, 3))[0]
            if sum(state_player) > 21:
                state_player += ' you are bust'
            state_dealer = np.random.randint(0, 11, size=(1, 1))[0]
            state_blackjack = [state_player, state_dealer]
            state_blackjack = [state_player, state_dealer]

            legal_moves = [0, 1, 2]

            if len(state_player) > 2:
                previous_actions = [0]
            else:
                previous_actions = []

            state = 'You are playing blackjack. State='+str(state_blackjack)+', legal moves='+str(legal_moves)+', previous actions='+str(previous_actions)+'.'

        print("\n --- ADAPTED STATE --- \n")
        print(self.adapter(state, encode=False))
        print("\n --- ADAPTED STATE ENCODED --- \n")
        print(self.adapter(state))