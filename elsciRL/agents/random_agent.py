import random
from elsciRL.agents.agent_abstract import Agent
import torch
from torch import Tensor

class RandomAgent(Agent):
    """This is simply a random decision maker, does not learn."""
    def __init__(self):
        super().__init__()

    def policy(self, state: Tensor, legal_actions: list) -> str:
        action = random.choice(legal_actions)
        return action
    
    def learn(self, state: Tensor, next_state: Tensor, r_p: float, 
              action_code: str) -> float:
        # Do nothing.
        return None
    
    def q_result(self):
        """Random agent has no knowledge."""
        total_q = 0
        mean_q = 0
        return total_q, mean_q
