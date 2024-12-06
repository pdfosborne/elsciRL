import numpy as np
import random
import pickle
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from enum import Enum
from collections import deque
from functools import lru_cache

from torch import Tensor
from torch.autograd import Variable
from sentence_transformers import SentenceTransformer

from elsciRL.agents.agent_abstract import QLearningAgent


class StateEncNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sequence_size: int = 1,
        seq_hidden_dim: int = 10,
        hidden_dim: int = 128,
        num_hidden: int = 1,
    ):
        super(StateEncNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.seq_size: int = sequence_size
        self.sent_hidden_dim: int = seq_hidden_dim
        self.hidden_dim: int = hidden_dim
        self.num_hidden: int = num_hidden
        self.seq: nn.LSTM = None
        self.lstm_h0c0: Tuple[float] = (0.0, 0.0)

        if sequence_size > 1:
            self.seq: nn.LSTM = nn.LSTM(
                input_size, seq_hidden_dim, 1, batch_first=True
            ).to(self.device)
            self.hidden: nn.Sequential = nn.Sequential(
                nn.Linear(seq_hidden_dim, hidden_dim, device=self.device),
                *[nn.Linear(hidden_dim, hidden_dim) for i in range(num_hidden)]
            ).to(self.device)
            h0 = torch.randn(1, 10).to(self.device)
            c0 = torch.randn(1, 10).to(self.device)
            self.lstm_h0c0 = (h0, c0)
        else:
            self.hidden: nn.Sequential = nn.Sequential(
                nn.Linear(input_size, hidden_dim, device=self.device),
                *[nn.Linear(hidden_dim, hidden_dim) for i in range(num_hidden)]
            ).to(self.device)

        self.output:nn.Linear=nn.Linear(hidden_dim, output_size).to(self.device)

    def forward(self, x):
        x = x.to(self.device)

        if self.seq_size > 1:
            seq_h = self.seq(x)[0][-1]
            # seq_h = self.seq(x, self.lstm_h0c0)[0][-1]
            h = self.hidden(seq_h)
        else:
            try:
                h = self.hidden(x)
            except AttributeError:
                print(
                    'Data Type Error: Encoder must be a tensor(float32) \n For example use np.array([0,0,...], dtype="float32")'
                )

        out = torch.softmax(self.output(h), -1)

        return out


class Transition:
    def __init__(self, state: Tensor, action: int,
                 next_state: Tensor, reward: float):
        self.state: Tensor = state
        self.action: int = action
        self.next_state: Tensor = next_state
        self.reward: float = reward


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)


class DQN:
    def __init__(
        self,
        sequence_size: int,
        input_size: int,
        output_size: int,
        action_space_index: Dict[str, int] = None,
        seq_hidden_dim: int = 10,
        hidden_dim: int = 128,
        num_hidden: int = 1,
        learn_step_counter: int = 0,
        memory_size: int = 2000,
        epsilon: float = 0.2,
        epsilon_step: float = 0.01,
        target_replace_iter: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 1,
    ):

        self.policy_net = StateEncNet(
            input_size,
            output_size,
            sequence_size,
            seq_hidden_dim,
            hidden_dim,
            num_hidden,
        )
        self.target_net = StateEncNet(
            input_size,
            output_size,
            sequence_size,
            seq_hidden_dim,
            hidden_dim,
            num_hidden,
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_step_counter = learn_step_counter  # for target updating
        self.memory_size: int = memory_size
        self.sequence_size: int = sequence_size
        self.epsilon_reset: float = epsilon
        self.epsilon: float = epsilon
        self.epsilon_step: float = epsilon_step
        self.action_space_index = action_space_index if action_space_index else dict()

        self.memory = ReplayMemory(memory_size)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.loss_func = nn.BCELoss()
        self.q_eval_list = []

        # Hyperparameters and parameters now defined by config system
        self.TARGET_REPLACE_ITER: int = target_replace_iter
        self.batch_size = batch_size
        self.batch_size_enabled = False

    def choose_action(self, state_enc: Tensor, legal_actions: List[str]) -> str:
        x = torch.as_tensor(
            state_enc, device=self.policy_net.device, dtype=torch.float32
        )

        for act in legal_actions:
            if act not in self.action_space_index:
                self.action_space_index[act] = np.random.randint(
                    0, self.policy_net.output_size
                )

        # Epsilon greedy action selection
        rng = np.random.rand()
        if rng < self.epsilon:
            action = legal_actions[np.random.randint(0, len(legal_actions))]
        else:
            if self.epsilon > 0:
                self.epsilon = self.epsilon - (
                    self.epsilon * self.epsilon_step
                )  # Added epsilon step reduction to smooth exploration to greedy
                if self.epsilon < 0:
                    self.epsilon = 0
            with torch.no_grad():
                actions_value = self.policy_net(x)
                list_value_legal_moves = [
                    (act, actions_value[self.action_space_index[act]])
                    for act in legal_actions
                ]
                action = max(list_value_legal_moves, key=lambda e: e[1])[0]
        return action

    def store_transition(
        self, state: Tensor, action: int, reward: float, next_state: Tensor
    ):
        transition = Transition(state, action, next_state, reward)
        self.memory.push(transition)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        try:
            b_memory = self.memory.sample(self.batch_size)
            if self.batch_size_enabled is False:
                print(
                    "Sufficient experience to use batch_size="
                    + str(self.batch_size)
                    + " given by config inputs"
                )
                self.batch_size_enabled = True
        except:
            if self.learn_step_counter == 1:
                print(
                    "Not enough experience for batch size, defaulting to batch_size=1 for now..."
                )
            b_memory = self.memory.sample(1)

        b_state = torch.stack([mem.state for mem in b_memory])
        b_action = torch.tensor(
            [mem.action for mem in b_memory],
            device=self.policy_net.device,
            dtype=torch.int64,
        )
        b_reward = torch.tensor(
            [mem.reward for mem in b_memory],
            device=self.policy_net.device,
            dtype=torch.float32,
        )
        b_next_state = torch.stack([mem.next_state for mem in b_memory])

        # q_eval w.r.t the action in experience
        q_eval = self.policy_net(b_state)[range(b_action.shape[0]), b_action]
        q_next = self.target_net(b_next_state).max(dim=1)[0][
            0
        ]  # .detach()  # detach from graph, don't backpropagate
        q_target = b_reward + 0.8 * q_next
        loss = self.loss_func(q_eval, q_target)

        self.q_eval_list.extend(q_eval.tolist())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def summary_results(self):
        total_q = np.sum(self.q_eval_list)
        mean_q = total_q / (len(self.q_eval_list) - self.q_eval_list.count(0))

        return total_q, mean_q


class NeuralQLearningAgent(QLearningAgent):
    def __init__(
        self,
        sequence_size: int,
        input_size: int,
        output_size: int,
        learn_step_counter: int = 0,
        memory_size: int = 2000,
        epsilon: float = 0.05,
        epsilon_step: float = 0.05,
        seq_hidden_dim: int = 10,
        hidden_dim: int = 128,
        num_hidden: int = 1,
        action_space_index: Dict[str, int] = None,
        target_replace_iter: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 1,
    ):
        self.dqn: DQN = DQN(
            sequence_size,
            input_size,
            output_size,
            action_space_index,
            seq_hidden_dim,
            hidden_dim,
            num_hidden,
            learn_step_counter,
            memory_size,
            epsilon,
            epsilon_step,
            target_replace_iter,
            learning_rate,
            batch_size,
        )
        self.sequence_size = sequence_size

    def policy(self, state: Tensor, legal_actions: list) -> str:
        return self.dqn.choose_action(state, legal_actions)

    def learn(
        self, state: Tensor, next_state: Tensor, r_p: float, action_code: str
    ) -> float:
        action_index = self.dqn.action_space_index[action_code]
        self.dqn.store_transition(state, action_index, r_p, next_state)
        dqn_loss = self.dqn.learn()

        return dqn_loss

    def exploration_parameter_reset(self):
        self.dqn.epsilon = self.dqn.epsilon_reset

    def q_result(self):
        results = self.dqn.summary_results()
        total_q = results[0]
        mean_q = results[1]

        return total_q, mean_q

    def clone(self):
        clone = pickle.loads(pickle.dumps(self))
        clone.dqn.epsilon = self.dqn.epsilon_reset
        return clone
