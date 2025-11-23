from elsciRL.environment_setup.gym_translator import EngineToGym

class EnvManager:
    """Handles environment setup and management."""
    def __init__(self, interaction_loop_class, adapters):
        self.interaction_loop_class = interaction_loop_class
        self.adapters = adapters

    def create_env(self, Engine, Adapters, local_setup_info):
        return self.interaction_loop_class(Engine=Engine, Adapters=Adapters, local_setup_info=local_setup_info)

    def create_gym_env(self, Engine, Adapter, setup_info, wrappers=None):
        """Create a Gym environment from an elsciRL Engine and Adapter using gym_translator.

        Adapter can be either the adapter class itself or the lookup key registered in
        ``self.adapters``. Optional wrappers can be provided to post-process the created
        environment (e.g., to add reward shaping).
        """
        adapter_cls = Adapter
        if not callable(Adapter):
            adapter_cls = self.adapters.get(Adapter)
        if adapter_cls is None:
            raise ValueError(f"Adapter '{Adapter}' not found when creating Gym environment.")

        gym_env = EngineToGym()
        gym_env.load(Engine, Adapter=adapter_cls, setup_info=setup_info)

        if wrappers:
            for wrapper in wrappers:
                gym_env = wrapper(gym_env)
        return gym_env
