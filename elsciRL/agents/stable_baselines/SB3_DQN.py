import pickle
import torch
import numpy as np
from elsciRL.agents.agent_abstract import QLearningAgent
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from PIL import Image # Used to generate GIF

class SB_DQN(QLearningAgent):
    def __init__(self, policy:str='MlpPolicy', env:gym.Env = None, learning_rate:float=0.0001, buffer_size:int=1000000):
        self.epsilon: int = 0 # Not used currently but required for compatibility
        self.device = "auto" if torch.cuda.is_available() else "cpu" 
        self.dqn = DQN(policy, env, verbose=0, device=self.device, 
                       learning_rate=learning_rate, buffer_size=buffer_size)
        if torch.cuda.is_available():
            print("---- Using GPU ----")
            print("Device:", self.dqn.device)

    def policy(self, state: any) -> str:
        return self.dqn.predict(state)

    def learn(self, total_steps:int=100) -> float:
        self.dqn.learn(total_timesteps=total_steps)
    
    def test(self, env, render:bool=False):
        #mean_reward, std_reward = evaluate_policy(self.a2c, env, n_eval_episodes=1)
        vec_env = self.dqn.get_env()
        obs = vec_env.reset()
        
        actions = []
        states = []

        done = False
        render_stack = []
        if render:
            render_stack.append(
                Image.fromarray(vec_env.render().astype('uint8'))
                )
        while not done: 
            action, _state = self.dqn.predict(obs, deterministic=True)
            if isinstance(action, np.int64):
                actions.append(action.item())
            else:
                actions.append(action[0])
            #actions.append(action[0])
            
            obs, r, done, info = vec_env.step(action)
            states.append(info[0]['obs'])
            if render:
                render_stack.append(
                    Image.fromarray(vec_env.render().astype('uint8'))
                    )
            
            #vec_env.render("human")
        episode_reward = info[0]['episode']['r']
        if episode_reward > 0.5:
            print("----> ", episode_reward)

        return episode_reward, actions, states, render_stack
    
    def q_result(self):
        results = [0,0]
        total_q = results[0]
        mean_q = results[1]
        return total_q, mean_q
    
    def clone(self):
        clone = pickle.loads(pickle.dumps(self))
        return clone