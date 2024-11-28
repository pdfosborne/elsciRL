import torch
from typing import List
from torch import Tensor
import numpy as np

#from elsciRL.encoders.encoder_abstract import StateEncoder
class PriorActionsEncoder():
    def __init__(self, all_possible_actions):
        self.all_possible_actions = all_possible_actions
        device = "cuda" if torch.cuda.is_available() else "cpu" # Make this optional choice with parameter
        self.vectors: Tensor = torch.cat([torch.eye(len(self.all_possible_actions)), torch.zeros(1, len(self.all_possible_actions))]).to(device) 

        self.all_possible_actions_dict_init = {}
        for action in self.all_possible_actions:
            self.all_possible_actions_dict_init[action] = int(0)

    def encode(self, state: List[str] = None, legal_actions:list = None, episode_action_history:list = None, 
               indexed: bool = False) -> Tensor:
        """Vector of prio actions in game so far, similar to blindfold chess."""
        # STATE ENCODER
        # - Updated to use all possible actions for consistency with poss action encoder and generally more suitable
        # - Chess has loads of possible actions which is somewhat unique to the problem
        # - BUT order must be preserved in the prior action encoder
        all_possible_actions = self.all_possible_actions_dict_init.copy()
        for a,action in enumerate(episode_action_history): 
            all_possible_actions[action] = int(a)

        state_encoded = torch.tensor(list(all_possible_actions.values()))
        if (not indexed):
            state_encoded = self.vectors[state_encoded].flatten()
        
        return state_encoded