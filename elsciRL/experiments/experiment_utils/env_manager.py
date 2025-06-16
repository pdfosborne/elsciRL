from elsciRL.environment_setup.gym_translator import EngineToGym

class EnvManager:
    """Handles environment setup and management."""
    def __init__(self, interaction_loop_class, adapters):
        self.interaction_loop_class = interaction_loop_class
        self.adapters = adapters

    def create_env(self, Engine, Adapters, local_setup_info):
        return self.interaction_loop_class(Engine=Engine, Adapters=Adapters, local_setup_info=local_setup_info)

    def create_gym_env(self, Engine, Adapter, setup_info):
        """Create a Gym environment from an elsciRL Engine and Adapter using gym_translator."""
        gym_env = EngineToGym()
        gym_env.load(Engine, Adapter=Adapter, setup_info=setup_info)
        return gym_env
