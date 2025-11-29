import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pickle
from typing import List, Tuple, Dict, Any, Hashable, Iterable

from elsciRL.agents.agent_abstract import QLearningAgent

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DQNAgent(QLearningAgent):
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 hidden_size: int = None,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 10,
                 device: str = None,
                 ):
        
        self.input_size = input_size
        self.output_size = output_size
        
        
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_reset = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.update_counter = 0
        
        # Create main and target networks with optional device specification
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.policy_net = DQN(input_size, output_size, hidden_size).to(self.device)
        print(f"DQN Agent initialized on device: {self.device}")
        print(self.policy_net)
        self.target_net = DQN(input_size, output_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def save(self) -> List[Dict]:
        """Save the agent's state"""
        return [
            self.policy_net.state_dict(),
            self.target_net.state_dict(),
            self.optimizer.state_dict(),
            {
                'epsilon': self.epsilon,
                'update_counter': self.update_counter,
                'memory': list(self.memory)  # Convert deque to list for serialization
            }
        ]
    
    def load(self, saved_agent: List[Dict]):
        """Load the agent's state"""
        if len(saved_agent) >= 4:
            self.policy_net.load_state_dict(saved_agent[0])
            self.target_net.load_state_dict(saved_agent[1])
            self.optimizer.load_state_dict(saved_agent[2])
            state_dict = saved_agent[3]
            self.epsilon = state_dict['epsilon']
            self.update_counter = state_dict['update_counter']
            if 'memory' in state_dict:
                self.memory = deque(state_dict['memory'], maxlen=self.memory.maxlen)

    def exploration_parameter_reset(self):
        """Reset the exploration parameter to its initial value"""
        self.epsilon = self.epsilon_reset

    def clone(self):
        """Create a deep copy of the agent"""
        clone = pickle.loads(pickle.dumps(self))
        clone.epsilon = self.epsilon_reset
        return clone

    def policy(self, state: torch.Tensor, legal_actions: list, **kwargs) -> Hashable:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return random.choice(legal_actions)
        
        with torch.no_grad():
            state = state.to(self.device)
            # Ensure state has correct shape [batch_size, input_size]
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            q_values = self.policy_net(state)
            
            # Mask illegal actions with large negative values
            mask = torch.ones_like(q_values) * float('-inf')
            for action in legal_actions:
                mask[0][action] = 0
            q_values = q_values + mask
            
            return q_values.argmax().item()

    def learn(self, state: torch.Tensor, next_state: torch.Tensor, 
              immediate_reward: float, action: Hashable, **kwargs) -> None:
        """Store experience and train the network"""
        # Detach tensors and move to CPU to save GPU memory
        # This prevents keeping computational graphs in replay buffer
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu()
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu()
            if len(next_state.shape) == 1:
                next_state = next_state.unsqueeze(0)
            
        # Store experience in replay memory (on CPU to save GPU memory)
        self.memory.append((state, action, next_state, immediate_reward))
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self._train()
            
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            

    def _train(self):
        """Train the network using experience replay"""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards = zip(*batch)
        
        # Convert to tensors and move to GPU only for training
        states = torch.cat(states).to(self.device)  # [batch_size, input_size]
        next_states = torch.cat(next_states).to(self.device)  # [batch_size, input_size]
        actions = torch.tensor(actions, device=self.device).long()  # [batch_size]
        rewards = torch.tensor(rewards, device=self.device).float()  # [batch_size]
        
        # Clear optimizer gradients
        self.optimizer.zero_grad()
        
        # Get current Q values
        current_q_values = self.policy_net(states)  # [batch_size, output_size]
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0][:self.batch_size]  # [batch_size]
        
        # Compute target Q values (detach to prevent gradient flow)
        target_q_values = (rewards + (self.gamma * next_q_values)).detach()
        
        # Compute loss and update
        loss = self.criterion(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        
        # Clear intermediate tensors from GPU memory
        del states, next_states, actions, rewards, current_q_values, next_q_values, target_q_values, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None 