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

    def create(self, agent_type, agent_parameters, engine=None, adapter=None):
        if agent_type == "DQN":
            if adapter:
                adapter_sample = self.adapters[adapter](setup_info=self.setup_info)
                # Set input_size from adapter
                try:
                    input_size = adapter_sample.input_dim
                    print(f"Using input_dim from adapter {adapter}: {input_size}")
                except Exception:
                    try:
                        input_size = adapter_sample.encoder.output_dim
                        print(f"Using encoder output_dim from encoder {adapter_sample.encoder}: {input_size}")
                    except Exception:
                        try:
                            input_size = adapter_sample.LLM_adapter.encoder.output_dim
                            print(f"Using LLM_adapter encoder output_dim from LLM adapter {adapter_sample.LLM_adapter}: {input_size}")
                        except Exception:
                            print(f"Adapter {adapter} does not have input_dim specified.")
                            raise ValueError(f"No input dim size found in adapter: {adapter}")

            if engine:
                print(engine)
                engine_sample = engine(local_setup_info=self.setup_info)
                try:
                    output_size = engine_sample.output_size
                except Exception:
                    try:
                        output_size = engine_sample.output_dim
                    except Exception:
                        try:
                            output_size = engine_sample.output_dim_size
                        except Exception:
                            print(f"Engine {engine} does not contain output dim size for DQN agent, using default 1,000.")
                            output_size = 1000
            # Order must match DQN input
            temp_dict = {'input_size': input_size, 'output_size': output_size}
            temp_dict.update(agent_parameters)
        else:
            # For other agents, we assume the parameters are already in the correct format
            temp_dict = agent_parameters
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return self.agent_types[agent_type](**temp_dict)
