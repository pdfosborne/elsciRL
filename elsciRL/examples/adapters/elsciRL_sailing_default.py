from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from elsciRL.encoders.poss_state_encoded import StateEncoder
from gymnasium.spaces import Text, Discrete

class DefaultAdapter:

    # ------ Static Methods ---------------------------------------
    # - Defined by simulator source https://github.com/PPierzc/ai-learns-to-sail/blob/master/tasks/channel.py
    @staticmethod
    def angle_to_state(angle):
        return int(30 * ((angle + np.pi) / (2 * np.pi) % 1))  # Discretization of the angle space
    
    @staticmethod
    def x_to_state(x):
        return int(40 * ((x + -10) / 20))  # Discretization of the x space
    
    @staticmethod
    def state_discretizer(state):
        x = float(state.split('_')[0])
        x_state = DefaultAdapter.x_to_state(x)

        angle = float(state.split('_')[1])
        angle_state = DefaultAdapter.angle_to_state(angle)

        state_out = str(x_state)+'_'+str(angle_state)
        return state_out
    # -------------------------------------------------------------

    _cached_state_idx: Dict[str, int] = dict()
    def __init__(self, setup_info:dict={}) -> None:
        # ------ State Encoder ---------------------------------------
        # Initialise encoder based on all possible env states
        all_possible_x = [i*-1 for i in range(40)]
        all_possible_angle = [i for i in range(30)]
        # Need an index that preserves the identity of both the x and angle values
        all_possible_states = []
        for x_ind in all_possible_x:
            for angle_ind in all_possible_angle:
                index = str(x_ind)+'_'+str(angle_ind)
                all_possible_states.append(index)
        # Input to pre-built possible state encoder
        #self.encoder = StateEncoder(all_possible_states)
        self.encoder = {}
        # Observartion is string: "x_angle"
        # -> Then discretized and returned as string: "x_state_angle_state"
        # -> Before being numeritized to a unique id (x:-10-10*2dp * angle:0-2pi*1dp)
        self.observation_space = Discrete(2000*30)
    
    
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every piece name for current board position """
        
        state = DefaultAdapter.state_discretizer(state)

        # Encode to Tensor for agents
        if encode:
            #state_encoded = self.encoder.encode(state=state)
            # elsciRL state encoder is large and not needed for tabular agents
            # - Wont work for neural agents
            if (state not in self.encoder):
                state_encoded = torch.tensor(len(self.encoder))
                self.encoder[state] = state_encoded
            else:
                state_encoded = self.encoder[state]
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in DefaultAdapter._cached_state_idx):
                    DefaultAdapter._cached_state_idx[sent] = len(DefaultAdapter._cached_state_idx)
                state_indexed.append(DefaultAdapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded