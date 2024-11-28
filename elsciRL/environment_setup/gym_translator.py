from typing import Callable, Any
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces

class EngineToGym(gym.Env):
    def __init__(self):
        print("elsciRL Env transformed to Gym Env.")

    def load(self, Engine, engine_name:str=None, Adapter:Callable[[Any], Any]=None, setup_info:dict={}):
        self.engine = Engine(setup_info)
        self.Adapter = Adapter()
        self.reward_signal = None
        self.reward_signal_tracker = []
        #self.action_history = []
        #self.obs_history = []
        # Use name if given directly, otherwise check engine ledger
        if engine_name is not None:
            self.name = engine_name
        elif (self.engine.ledger['id'] != 'Unique Problem ID')&(self.engine.ledger['id'] != ''):
            self.name = self.engine.ledger['id']
        else:
            print("\n WARNING: Engine name not set, using default name --> set inside ledger [id] field.")
            self.name = "elsciRLEnv-v0"
            
        # --------------------------
        # Define observation and action spaces
        # - Observations are dictionaries with the agent's and the target's location.
        # - Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        try:
            # First check if observation space is defined by the adapter
            self.observation_space = self.Adapter.observation_space
        except:
            # Then check if observation space is defined by the engine
            try:
                self.observation_space = self.engine.observation_space
            except AttributeError:
                # Otherwise, use default observation space
                print("WARNING: Observation space not defined in either adapter of engine.")

        # - A single dimension of N number of discrete actions 
        self.action_space = spaces.Discrete(self.engine.ledger['action_space_size'])
        # --------------------------
        self.render_mode = self.engine.ledger['render_data']['render_mode']

    def reset(self, seed=None, options=None):
        observation = self.engine.reset()
        obs_enc = self.Adapter.adapter(observation, self.engine.legal_move_generator(), 
                               self.engine.action_history, encode = True )
        obs_enc = obs_enc.cpu().numpy() # Need to get all on CPU before registering with GYM
        self.reward_signal_tracker = [] # Only give agent reward for first time it sees a sub-goal
        self.action_history = [] # Reset action history
        self.episode_reward = 0
        #self.obs_history = []
        return obs_enc, {}

    def step(self, action=None):
        # Gym step function combines elsciRL Engine step and Adapter
        observation, reward, terminated, info = self.engine.step(action=int(action))
        self.action_history.append(action)
        # if observation not in self.obs_history:
        #     reward += 0.05 # Give small reward to encourage exploration
        # self.obs_history.append(observation)
        info['obs'] = observation

        # Apply custom reward signal if defined
        # - Defined as dict:= {obs:reward, obs:reward, ...}
        if self.engine.reward_signal:
            if observation in self.engine.reward_signal:
                if observation not in self.reward_signal_tracker:
                    # Only override if new reward is higher
                    if self.engine.reward_signal[observation] > reward:
                        reward=self.engine.reward_signal[observation]
                    self.reward_signal_tracker.append(observation)
                    

        # If a language problem then we also want processed observation
        if self.engine.ledger['type'] =='Language':
            obs_adapted = self.Adapter.adapter(observation, self.engine.legal_move_generator(), 
                                    self.engine.action_history, encode = False)
            info['obs_adapted'] = obs_adapted
        obs_enc = self.Adapter.adapter(observation, self.engine.legal_move_generator(), 
                               self.engine.action_history, encode = True )
        obs_enc = obs_enc.cpu().numpy() # Need to get all on CPU before registering with GYM
        truncated = False
        self.episode_reward += reward
        return obs_enc, reward, terminated, truncated, info

    def render(self):
        return self.engine.render() 

    def close(self):
        self.engine.close()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self

    
@staticmethod
def GymRegistration(Engine, Adapter, setup_info:dict={}):
    """This provides a function for converting elsciRL engines into OpenAI Gym environments. \n
    elsciRL engines include a conditional action space which is not inherently supported by OpenAI Gym. \n
    Outputs Engine in the OpenAI Gym format with a wrapper for the elsciRL adapter.
    """
    # Translate Engine to OpenAI Gym class structure
    environment = EngineToGym()
    environment.load(Engine, 'Test-1', Adapter, setup_info)
    # Register and make the environment
    register(id=environment.name, entry_point=environment)
    gym_env = gym.make(environment.name)
    
    return gym_env
