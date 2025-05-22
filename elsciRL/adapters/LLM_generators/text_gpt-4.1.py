from abc import ABC, abstractmethod
from typing import List, Any
import os

from openai import OpenAI

class LLMAdapter(ABC):
    """Convert a general prompt and raw text state into a description of the state."""
    def __init__(self, base_prompt:str):
        super().__init__()
        # Define the fields that describe the state features:
        self.base_prompt = base_prompt

    @abstractmethod
    def _read(raw_state) -> list:
        # Read the data.
        # fill in the feature fields
        raise NotImplementedError
    
    def call_gpt_api(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        self.response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": self.base_prompt}],
            max_tokens=5000
        )
        return self.response.to_dict() if hasattr(self.response, 'to_dict') else self.response

    def process_gpt_response(self, response):
        if response and 'choices' in response:
            return response['choices'][0]['message']['content']
        return None

    def adapter(self, state:str):
        "Returns the adapted form, may require input flag for encoded or non-encoded output."
        adapted_state = self.call_gpt_api(self.base_prompt + "\nUser: " + state)
        return self.process_gpt_response(adapted_state)
        
    def sample(self, state:str):
        """Returns a sample of an adapted state form (typically initial position of the environment)."""
        if not state:
            state = 'The current state is empty.'
        return self.adapter(state)

        
