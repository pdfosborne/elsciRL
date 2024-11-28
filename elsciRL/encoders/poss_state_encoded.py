import torch
from typing import List, Any
from torch import Tensor

from elsciRL.encoders.encoder_abstract import StateEncoder

class StateEncoder():
    def __init__(self, possible_states):
        """Encoder for default state representation produced by the environment/engine."""
        self.all_possible_states = possible_states
        device = "cuda" if torch.cuda.is_available() else "cpu" # Make this optional choice with parameter
        self.vectors: Tensor = torch.cat([torch.eye(len(self.all_possible_states)), torch.zeros(1,len(self.all_possible_states))]).to(device)         # tensor needs to be defined to len(local_object)
    
    def encode(self, state:Any = None, legal_actions:list = None, episode_action_history:list = None,
               indexed: bool = False) -> Tensor:
        """ NO CHANGE - Board itself is used as state as is and simply converted to a vector"""
        # Goes through every possible state and labels if occurance of state matches
        # Binary vector
        # NOT RECOMMENDED FOR LARGE STATE SPACES
        # TODO: improve runtime with dict instead of list itearations
        state_encoded: List[int] = []
        for possible_state in self.all_possible_states:
            if possible_state == state:
                state_encoded.append(1)
            else:
                state_encoded.append(0)
        state_encoded = torch.tensor(state_encoded)
        if (not indexed):
            state_encoded = self.vectors[state_encoded].flatten()

        return state_encoded    