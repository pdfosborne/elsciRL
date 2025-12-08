from typing import Dict, Type

from elsciRL.agents.stable_baselines.SB3_PPO import SB_PPO
from elsciRL.agents.stable_baselines.SB3_A2C import SB_A2C
from elsciRL.agents.stable_baselines.SB3_DQN import SB_DQN
from elsciRL.agents.clean_rl.ppo import CleanRLPPO


class PolicyAgentFactory:
    """Factory for Gym/PyTorch policy-gradient agents (SB3-backed)."""

    def __init__(self):
        self.agent_types: Dict[str, Type] = {
            "SB3_PPO": SB_PPO,
            "SB3_A2C": SB_A2C,
            "SB3_DQN": SB_DQN,
            "PPO": CleanRLPPO,
        }

    def register_agent(self, name: str, agent_cls: Type):
        self.agent_types[name] = agent_cls

    def create(self, agent_type: str, agent_parameters: Dict, env):
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown policy agent type: {agent_type}")
        agent_cls = self.agent_types[agent_type]
        # Most SB3 wrappers accept the env kwarg directly.
        return agent_cls(env=env, **agent_parameters)
