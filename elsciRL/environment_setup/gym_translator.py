from typing import Callable, Any
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _figure_to_rgb_array(fig):
    """Convert a Matplotlib figure into an RGB numpy array."""
    canvas = fig.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    # Prefer buffer RGBA for compatibility, then drop alpha channel.
    if hasattr(canvas, "buffer_rgba"):
        buffer = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        array = np.array(buffer).reshape((height, width, 4))[..., :3]
    else:
        buffer = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        array = buffer.reshape((height, width, 3))
    plt.close(fig)
    return array

class EngineToGym(gym.Env):
    def __init__(self):
        print("elsciRL Env transformed to Gym Env.")

    def load(self, Engine, engine_name:str=None, Adapter:Callable[[Any], Any]=None, setup_info:dict={}):
        self.engine = Engine(setup_info)
        self.Adapter = Adapter(setup_info=setup_info)
        self.reward_signal = None
        self.reward_signal_tracker = []
        # Use name if given directly, otherwise check engine ledger
        if engine_name is not None:
            self.name = engine_name
        elif (self.engine.ledger['id'] != 'Unique Problem ID')&(self.engine.ledger['id'] != ''):
            self.name = self.engine.ledger['id']
        else:
            print("\n WARNING: Engine name not set, using default name --> set inside ledger [id] field.")
            self.name = "elsciRLGymEnv-v0"
            
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
        self.last_obs = observation
        self.last_info = {}
        obs_enc = self.Adapter.adapter(
            observation,
            self.engine.legal_move_generator(),
            self.engine.action_history,
            encode=True,
        )
        obs_enc = self._format_observation(obs_enc)
        self.reward_signal_tracker = [] # Only give agent reward for first time it sees a sub-goal
        self.action_history = [] # Reset action history
        self.episode_reward = 0
        #self.obs_history = []
        return obs_enc, {}

    def step(self, state=[], action=0):
        # Gym step function combines elsciRL Engine step and Adapter
        base_state = getattr(self, "last_obs", None)
        step_result = self.engine.step(state=base_state, action=action)
        if not isinstance(step_result, tuple) or len(step_result) != 4:
            print(
                "[EngineToGym] Invalid engine step output:",
                {
                    "engine": type(self.engine).__name__,
                    "adapter": type(self.Adapter).__name__,
                    "result": step_result,
                },
            )
            raise ValueError(
                "Engine.step must return a tuple of (observation, reward, terminated, info). "
                f"Received: {step_result!r}"
            )
        observation, reward, terminated, info = step_result
        if isinstance(action, np.int64):
            self.action_history.append(action.item())
        else:
            self.action_history.append(action)
        # if observation not in self.obs_history:
        #     reward += 0.05 # Give small reward to encourage exploration
        # self.obs_history.append(observation)
        if info:
            info['obs'] = observation
        else:
            info = {'obs': observation}

        # Apply custom reward signal if defined
        # - Defined as dict:= {obs:reward, obs:reward, ...}
        engine_reward_signal = getattr(self.engine, "reward_signal", None)
        if engine_reward_signal:
            if observation in engine_reward_signal:
                if observation not in self.reward_signal_tracker:
                    # Only override if new reward is higher
                    if engine_reward_signal[observation] > reward:
                        reward = engine_reward_signal[observation]
                    self.reward_signal_tracker.append(observation)
                    

        # If a language problem then we also want processed observation
        # TODO: Need better method for checking if language problem
        if 'lang' in self.engine.ledger['type'].lower():
            obs_adapted = self.Adapter.adapter(observation, self.engine.legal_move_generator(), 
                                    self.engine.action_history, encode = False)
            info['obs_adapted'] = obs_adapted
        obs_enc = self.Adapter.adapter(
            observation,
            self.engine.legal_move_generator(),
            self.engine.action_history,
            encode=True,
        )
        obs_enc = self._format_observation(obs_enc)
        truncated = False
        self.episode_reward += reward
        self.last_obs = observation
        self.last_info = info
        return obs_enc, reward, terminated, truncated, info

    def _format_observation(self, obs_enc):
        """Ensure adapter outputs match the declared Gym observation space."""

        # Handle PyTorch tensors - move to CPU before converting to numpy
        if hasattr(obs_enc, "detach"):
            obs_enc = obs_enc.detach()
            if hasattr(obs_enc, "cpu"):
                obs_enc = obs_enc.cpu()
        obs_array = np.asarray(obs_enc, dtype=np.float32)

        if isinstance(self.observation_space, spaces.Discrete):
            # Convert one-hot or vector encodings to scalar indices
            if obs_array.ndim == 0:
                return np.int64(obs_array.item())
            if obs_array.ndim == 1 and obs_array.size > 1:
                return np.int64(np.argmax(obs_array))
            return np.int64(obs_array.flatten()[0])

        # Default: ensure numpy array on CPU with correct dtype
        return obs_array

    def render(self):
        render_output = self.engine.render()
        if isinstance(render_output, np.ndarray):
            return render_output
        if hasattr(render_output, "canvas"):
            return _figure_to_rgb_array(render_output)
        return np.asarray(render_output)

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
