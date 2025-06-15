from abc import ABC, abstractmethod
from typing import List, Any
import os

from openai import OpenAI

try:
    from torch import Tensor
    from elsciRL.encoders.language_transformers.MiniLM_L6v2 import LanguageEncoder
except ImportError:
    print("Warning: torch or LanguageEncoder not found. Please ensure elsciRL is properly installed.")
    Tensor = None
    LanguageEncoder = None

class LLMAdapter(ABC):
    """Convert a general prompt and raw text state into a description of the state."""
    def __init__(self, base_prompt: str):
        super().__init__()
        # Define the fields that describe the state features:
        self.base_prompt = base_prompt

    @abstractmethod
    def _read(raw_state) -> list:
        # Read the data.
        # fill in the feature fields
        raise NotImplementedError


class GPTAdapter(LLMAdapter):
    """Adapter for OpenAI GPT models."""
    
    def __init__(self, base_prompt: str, model_name: str = "gpt-4"):
        super().__init__(base_prompt)
        self.model_name = model_name
        
        # Initialize the language encoder for encoding functionality
        if LanguageEncoder is not None:
            self.encoder = LanguageEncoder()
        else:
            print("Warning: LanguageEncoder not available. Encoding will not work.")
            self.encoder = None
    
    def _read(self, raw_state) -> list:
        """Read the data and fill in the feature fields."""
        # This method should be implemented based on specific requirements
        # For now, returning the raw state as a list
        return [raw_state] if isinstance(raw_state, str) else raw_state
    
    def call_gpt_api(self, prompt: str):
        """Call the OpenAI GPT API with the given prompt."""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.base_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5000
            )
            return response.to_dict() if hasattr(response, 'to_dict') else response
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None

    def process_gpt_response(self, response):
        """Process the response from OpenAI API."""
        if response and 'choices' in response:
            return response['choices'][0]['message']['content']
        return None

    def adapter(self, state: any, legal_moves: list = None, episode_action_history: list = None, encode: bool = True, indexed: bool = False) -> Tensor:
        """Returns the adapted form, may require input flag for encoded or non-encoded output."""
        # Build the full context prompt including legal moves and action history
        context_parts = []
        
        # Add state information
        if state is not None:
            context_parts.append(f"Current state: {state}")
        
        # Add legal moves if provided
        if legal_moves is not None and len(legal_moves) > 0:
            context_parts.append(f"Legal moves: {legal_moves}")
        
        # Add action history if provided
        if episode_action_history is not None and len(episode_action_history) > 0:
            recent_actions = episode_action_history[-5:]  # Last 5 actions
            context_parts.append(f"Recent actions: {recent_actions}")
        
        # Combine all context into a single prompt
        full_prompt = " | ".join(context_parts)
        
        # Get GPT response
        adapted_state = self.call_gpt_api(full_prompt)
        processed_response = self.process_gpt_response(adapted_state)
        
        if processed_response is None:
            processed_response = str(state) if state is not None else "No state available"
        
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
                return state_encoded
            else:
                print("Warning: Encoder not available, returning processed response as string")
                return processed_response
        else:
            return processed_response
        
    def sample(self, state: any):
        """Returns a sample of an adapted state form (typically initial position of the environment)."""
        if not state:
            state = 'The current state is empty.'
        return self.adapter(state, encode=True)

        
