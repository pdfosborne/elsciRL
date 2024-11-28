from abc import ABC, abstractmethod
from typing import Iterable, Hashable, Any
from torch import Tensor

class Agent(ABC):
    @abstractmethod
    def policy(self, **kwargs) -> str:
        pass

    def learn(self, **kwargs) -> str:
        pass

class QLearningAgent(Agent):
    def policy(self, state:Tensor, game_over:bool, legal_actions:list, **kwargs) -> Hashable:
        pass
    
    def learn(self, state:Tensor, action:Hashable, next_state:Iterable[Any], immediate_reward:float, **kwargs):
        pass


