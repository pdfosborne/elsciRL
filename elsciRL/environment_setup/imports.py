from typing import List
from elsciRL.agents.agent_abstract import Agent, QLearningAgent


class ImportHelper:
    def __init__(self, local_setup_info:dict={}) -> None:
        self.setup_info = local_setup_info

    def agent_info(self, STATE_ADAPTER_TYPES:dict={}):
        agent: Agent | QLearningAgent = self.setup_info['agent']
        agent_type: str = self.setup_info['agent_type']
        agent_name: str = self.setup_info['agent_name']
        if self.setup_info['adapter_select'] in STATE_ADAPTER_TYPES:
            agent_state_adapter = STATE_ADAPTER_TYPES[self.setup_info['adapter_select']](setup_info=self.setup_info)
        else:
            agent_state_adapter = ''
        return agent, agent_type, agent_name, agent_state_adapter

    def parameter_info(self):
        num_train_episodes: int = self.setup_info['number_training_episodes']
        num_test_episodes: int = self.setup_info['number_test_episodes']
        training_action_cap: int = self.setup_info['training_action_cap']
        testing_action_cap: int = self.setup_info['testing_action_cap']
        reward_signal: List[int] = self.setup_info['reward_signal'] 

        return num_train_episodes, num_test_episodes, training_action_cap, testing_action_cap, reward_signal
    
    def training_flag(self):
        train: bool = self.setup_info['train']
        return train

    def live_env_flag(self):
        live_env: bool = self.setup_info['live_env']
        observed_states: bool = self.setup_info['observed_states']
        experience_sampling: bool = self.setup_info['experience_sampling']
        return live_env, observed_states, experience_sampling