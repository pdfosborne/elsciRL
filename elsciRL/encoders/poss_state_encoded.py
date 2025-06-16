import torch
from typing import List, Any
from torch import Tensor
from tqdm import tqdm
from elsciRL.encoders.encoder_abstract import StateEncoder

class StateEncoder(StateEncoder):
    def __init__(self, num_states):
        """Encoder for default state representation produced by the environment/engine."""
        # Create dict lookup
        # - get binary list that indexes the state e.g. 0_0 -> [1,0,0,0] or 0_3 -> [0,0,0,1]
        # UPDATED - Now uses torch.nn.functional.one_hot for one-hot encoding
        # Using one-hot encoder is incredibly inefficient for large state spaces
        # Instead, we consider using an index-based encoding where each unique state is assigned a unique index.
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Make this optional choice with parameter
        self.vectors: Tensor = torch.cat([torch.eye(num_states), torch.zeros(1,num_states)]).to(self.device)         # tensor needs to be defined to len(local_object)
        self.name = "StateEncoder"
        self.input_type = "list"
        self.output_type = "tensor"
        self.output_dim = num_states

        self.encoder = {}
        self.encoder_idx = 0
        self.num_states = num_states

    def encode(self, state:Any = None, legal_actions:list = None, episode_action_history:list = None,
               indexed: bool = False) -> Tensor:
        """ Set of all possible states are simply converted to a vector"""
        # One hot encode the state if it is not already indexed
        if state not in self.encoder:
            state_encoded = self.encoder_idx  # Use the index as the state encoded value
            # Store the encoded state in the encoder dictionary
            self.encoder[state] = state_encoded
            # Increment the encoder index for the next unique state
            self.encoder_idx += 1
        else:
            state_encoded = self.encoder[state]

        # If indexed, use one-hot encoding
        # If not indexed, use the unique index to retrieve the vector
        if indexed:
            state_encoded = torch.nn.functional.one_hot(torch.tensor(state_encoded), num_classes=self.num_states).float().to(self.device)
        else:
            state_encoded = self.vectors[int(state_encoded)].flatten()
        
        return state_encoded    