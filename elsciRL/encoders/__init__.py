import time
import numpy as np
import pandas as pd
from typing import List, Dict, Iterable
from abc import ABC, abstractmethod
from elsciRL.adapters import StateAdapter
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
    
    # index_objects are the complete list of adapter specific elements used to define the encoder's index
    def encode(self, index_objects:list=None, state:list = None, legal_actions:list = None, prior_action:str = None,
                opponent_action:str = None, indexed: bool = False) -> Tensor:
        pass
    
    
class EncodedState(ABC):
    @abstractmethod
    def data() -> Iterable:
        raise NotImplementedError


class StateConverter(ABC):
    def __init__(self, adapter: StateAdapter):
        super().__init__()
        # Calls the conversion procedure
        self.data: EncodedState = self.convert(adapter.s)


    def convert(state: list) -> EncodedState:
        pass