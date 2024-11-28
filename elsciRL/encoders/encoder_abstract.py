import numpy as np
import pandas as pd
from typing import List, Dict
from abc import ABC, abstractmethod
from torch import Tensor

class Encoder(ABC):
    @abstractmethod
    def encode(self, *args, **kwargs) -> Tensor:
        pass

class StateEncoder(Encoder):
    tensor_cache: Dict[int, Tensor] = dict()
    tensor_cache_index: int = 0

    @staticmethod
    def cache_insert(t: Tensor):
        StateEncoder.tensor_cache[StateEncoder.tensor_cache_index] = t
        StateEncoder.tensor_cache_index += 1
    
    @staticmethod
    def cache_retrieve(offset: int, index: int):
        return StateEncoder.tensor_cache[offset + index]

        
    def encode(self, state:list = None, legal_actions:list = None, episode_action_history:str = None) -> Tensor:
        pass
    