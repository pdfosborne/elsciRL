class EnvManager:
    """Handles environment setup and management."""
    def __init__(self, interaction_loop_class, adapters):
        self.interaction_loop_class = interaction_loop_class
        self.adapters = adapters

    def create_env(self, Engine, Adapters, local_setup_info):
        return self.interaction_loop_class(Engine=Engine, Adapters=Adapters, local_setup_info=local_setup_info)
