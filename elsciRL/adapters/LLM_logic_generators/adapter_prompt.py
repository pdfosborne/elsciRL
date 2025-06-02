adapter_prompt = """
    Your role is to generate pseudocode for an adapter function that will be used to transform the state of an environment into a form that can be used by an agent.
    
    Adapters unify problems into a standard form so any agent in the elsciRL library can be used.

    In short, it transforms the state to a new form, optionally adding more context and then outputting a tensor.

        inputs: state, legal moves, action history for episode
        outputs: tensor for the encoded form of the adapted state

    # numeric adapter (numeric.py)
    class DefaultAdapter(setup_info):
    def __init__():
        # Determine discrete environment size: e.g. "4x4" => 16 positions
        # Initialize a StateEncoder for these positions
        # Optionally define an observation space (e.g., Discrete) needed for Gym agents

    def adapter(state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
        # If encode=True, convert the numeric state to a tensor (StateEncoder)
        # If indexed=True, map states to integer IDs

        return tensor(state_encoded)

    # language adapter (language.py)
    class LanguageAdapter(setup_info):
    def __init__():
        # Build obs_mapping dictionary describing each state as text
        # Initialize LanguageEncoder

    def adapter(state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
        # Convert numeric state ID to a text description (obs_mapping)
        # Optionally encode the text into a tensor (LanguageEncoder)
        # Optionally map each unique description to an indexed ID

        return tensor(state_encoded)

"""