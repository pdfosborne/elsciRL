class AgentFactory:
    """Factory for creating agent instances based on type name and parameters."""
    def __init__(self, adapters, setup_info):
        from elsciRL.agents.table_q_agent import TableQLearningAgent
        from elsciRL.agents.DQN import DQNAgent
        from elsciRL.agents.LLM_agents.ollama_agent import LLMAgent as OllamaAgent
        self.adapters = adapters
        self.agent_types = {
            "Qlearntab": TableQLearningAgent,
            "DQN": DQNAgent,
            "LLM_Ollama": OllamaAgent,
        }
        self.setup_info = setup_info

    def register_agent(self, name, agent_class):
        self.agent_types[name] = agent_class

    def create(self, agent_type, agent_parameters, adapter=None):
        if agent_type == "DQN" and adapter:
            # Set input_size from adapter
            try:
                agent_parameters['input_size'] = self.adapters[adapter](setup_info=self.setup_info).input_dim
            except Exception:
                print(f"Adapter {adapter} does not have input_dim specified.")
                try:
                    agent_parameters['input_size'] = self.adapters[adapter](setup_info=self.setup_info).encoder.output_dim
                except Exception:
                    raise ValueError(f"No input dim size found in adapter: {adapter}")
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return self.agent_types[agent_type](**agent_parameters)
