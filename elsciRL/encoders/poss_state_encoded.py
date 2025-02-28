import torch
from typing import List, Any
from torch import Tensor

from elsciRL.encoders.encoder_abstract import StateEncoder

class StateEncoder():
    def __init__(self, possible_states):
        """Encoder for default state representation produced by the environment/engine."""
        self.all_possible_states = possible_states
        # Create dict lookup
        # - get binary list that indexes the state e.g. 0_0 -> [1,0,0,0] or 0_3 -> [0,0,0,1]
        self.enc = {}
        for i,s in enumerate(possible_states):
            self.enc[s] = [1 if i == j else 0 for j in range(len(possible_states))]

        device = "cuda" if torch.cuda.is_available() else "cpu" # Make this optional choice with parameter
        self.vectors: Tensor = torch.cat([torch.eye(len(self.all_possible_states)), torch.zeros(1,len(self.all_possible_states))]).to(device)         # tensor needs to be defined to len(local_object)
    
    def encode(self, state:Any = None, legal_actions:list = None, episode_action_history:list = None,
               indexed: bool = False) -> Tensor:
        """ NO CHANGE - Board itself is used as state as is and simply converted to a vector"""
        # Goes through every possible state and labels if occurance of state matches
        # Binary vector
        # NOT RECOMMENDED FOR LARGE STATE SPACES
        state_encoded = self.enc[state]
        state_encoded = torch.tensor(state_encoded)
        if (not indexed):
            state_encoded = self.vectors[state_encoded].flatten()

        return state_encoded    