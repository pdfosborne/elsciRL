import gymnasium as gym

class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self, local_setup_info:dict={}) -> None:
        """Initialize Engine"""
        self.Environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4", 
                                    is_slippery=True,
                                    render_mode='rgb_array')
        
    def reset(self, start_obs:str=None):
        """Fully reset the environment."""
        obs, info = self.Environment.reset()
        return obs

    
    def step(self, state:any, action:any):
        """Enact an action."""
        # In problems where the agent can choose to reset the env
        if (state=="ENV_RESET")|(action=="ENV_RESET"):
            self.reset()
            
        obs, reward, terminated, truncated, info = self.Environment.step(action)
        return obs, reward, terminated, info

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = [0,1,2,3]
        return legal_moves

    def render(self):
        """Render an image or text of the environment."""
        return self.Environment.render()
        
    def close(self):
        """Close/Exit the environment."""
        self.Environment.close()
        print("Environment Closed")
