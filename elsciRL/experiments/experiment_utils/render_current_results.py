import os

def render_current_result(training_setup, current_environment, current_agent, local_save_dir):
    """Apply fixed policy to render current decision making for limited number of episodes."""
    # Override input training setups with previously saved 
    
    test_setup_info = training_setup.copy()

    test_setup_info['train'] = False # Testing Phase
    test_setup_info['training_results'] = False
    test_setup_info['observed_states'] = False
    test_setup_info['experience_sampling'] = False
    print("----------")
    print("Rendering trained agent's policy:")

    env = current_environment
    # ---
    env.number_episodes = 1 # Only render 1 episode
    env.agent = current_agent
    env.agent.epsilon = 0 # Remove random actions
    # ---
    # Render results
    if not os.path.exists(local_save_dir):
        os.mkdir(local_save_dir)
    env.episode_loop(render=True, render_save_dir=local_save_dir) 