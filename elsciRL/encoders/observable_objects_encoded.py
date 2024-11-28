import torch
from typing import List
from torch import Tensor

from elsciRL.encoders.encoder_abstract import StateEncoder

class ObjectEncoder():
    def __init__(self, local_objects):
        """Encoder for default state representation produced by the environment/engine."""
        self.local_objects = {obj: i for i, obj in enumerate(local_objects)}
        device = "cuda" if torch.cuda.is_available() else "cpu" # Make this optional choice with parameter
        self.vectors: Tensor = torch.cat([torch.eye(len(self.local_objects)), torch.zeros(1, len(self.local_objects))]).to(device)         # tensor needs to be defined to len(local_object)
    
    def encode(self, state:list = None, legal_actions:list = None, episode_action_history:list = None,
               indexed: bool = False) -> Tensor:
        """ NO CHANGE - Board itself is used as state as is and simply converted to a vector"""
        # Goes through every item in state and labels based on the known objects available in the environment
        # New vector encoded form, for Chess: 64x12 flattened into 768x1 int vector to denote object occurance
        # NOT BINARY vector, value is the occurance of each object type. 
        #  -> In chess this happens to be [1 or 0] because you cant have more than one piece in each position.
        state_encoded: Tensor = torch.tensor([self.local_objects.get(state_pos, len(self.local_objects)) for state_pos in state], 
                                             device=self.vectors.device)
        
        if (not indexed):
            state_encoded = self.vectors[state_encoded].flatten()

        return state_encoded    