import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

from elsciRL.agents.agent_abstract import QLearningAgent


def _flatten_obs(obs) -> np.ndarray:
    """Convert arbitrary Gym observation structures to a flat float32 array."""

    if isinstance(obs, dict):
        parts = [np.asarray(value, dtype=np.float32).ravel() for value in obs.values()]
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    obs_array = np.asarray(obs, dtype=np.float32)
    if obs_array.ndim == 0:
        obs_array = np.expand_dims(obs_array, axis=0)
    return obs_array.flatten()


class _ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.shared(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        features = self.get_features(obs)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value, dist


class CleanRLPPO(QLearningAgent):
    """Minimal CleanRL-style PPO agent compatible with elsciRL experiments."""

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        update_epochs: int = 4,
        batch_size: int = 2048,
        minibatch_size: int = 256,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_size: int = 128,
    ) -> None:
        if not hasattr(env.action_space, "n"):
            raise ValueError("CleanRLPPO currently supports discrete action spaces only.")

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        sample_obs, _ = self.env.reset()
        self.obs_dim = _flatten_obs(sample_obs).shape[0]
        self.action_dim = self.env.action_space.n
        self.policy_net = _ActorCritic(self.obs_dim, self.action_dim, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, eps=1e-5)

    def _prep_obs(self, obs) -> torch.Tensor:
        flat = _flatten_obs(obs)
        return torch.tensor(flat, dtype=torch.float32, device=self.device)

    def learn(self, total_steps: int = 1000) -> float:
        if total_steps <= 0:
            return 0.0

        total_updates = max((total_steps + self.batch_size - 1) // self.batch_size, 1)
        remaining_steps = total_steps
        obs, _ = self.env.reset()
        obs = self._prep_obs(obs)
        cumulative_reward = 0.0

        progress_bar = tqdm(total=total_updates, desc="PPO Updates", leave=False) if tqdm else None
        for _ in range(total_updates):
            if remaining_steps <= 0:
                break
            steps_to_sample = min(self.batch_size, remaining_steps)
            storage = {
                "obs": [],
                "actions": [],
                "logprobs": [],
                "rewards": [],
                "dones": [],
                "values": [],
            }

            for _step in range(steps_to_sample):
                action, log_prob, _, value, _ = self.policy_net.get_action_and_value(obs.unsqueeze(0))
                action_item = action.item()

                next_obs, reward, terminated, truncated, _ = self.env.step(action_item)
                done = terminated or truncated

                storage["obs"].append(obs)
                storage["actions"].append(action_item)
                storage["logprobs"].append(log_prob.detach())
                storage["rewards"].append(reward)
                storage["dones"].append(done)
                storage["values"].append(value.detach().squeeze(-1))

                cumulative_reward += reward
                obs = self._prep_obs(next_obs)

                if done:
                    next_obs, _ = self.env.reset()
                    obs = self._prep_obs(next_obs)

            with torch.no_grad():
                _, _, _, next_value, _ = self.policy_net.get_action_and_value(obs.unsqueeze(0))

            remaining_steps -= steps_to_sample
            if progress_bar:
                progress_bar.update(1)

            b_obs = torch.stack(storage["obs"])
            b_actions = torch.tensor(storage["actions"], device=self.device)
            b_logprobs = torch.stack(storage["logprobs"])
            b_rewards = torch.tensor(storage["rewards"], dtype=torch.float32, device=self.device)
            b_dones = torch.tensor(storage["dones"], dtype=torch.float32, device=self.device)
            b_values = torch.stack(storage["values"])

            advantages = torch.zeros(steps_to_sample, device=self.device)
            last_adv = 0.0
            for t in reversed(range(steps_to_sample)):
                if t == steps_to_sample - 1:
                    next_non_terminal = 1.0 - b_dones[t]
                    next_value = next_value.squeeze(0)
                else:
                    next_non_terminal = 1.0 - b_dones[t + 1]
                    next_value = b_values[t + 1]
                delta = b_rewards[t] + self.gamma * next_value * next_non_terminal - b_values[t]
                advantages[t] = last_adv = delta + self.gamma * self.gae_lambda * next_non_terminal * last_adv

            returns = advantages + b_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            batch_indices = np.arange(steps_to_sample)
            effective_minibatch = min(self.minibatch_size, steps_to_sample)
            for _ in range(self.update_epochs):
                np.random.shuffle(batch_indices)
                for start in range(0, steps_to_sample, effective_minibatch):
                    idx = batch_indices[start : start + effective_minibatch]
                    obs_mb = b_obs[idx]
                    actions_mb = b_actions[idx]
                    logprobs_mb = b_logprobs[idx]
                    advantages_mb = advantages[idx]
                    returns_mb = returns[idx]
                    values_mb = b_values[idx]

                    _, new_logprob, entropy, new_value, _ = self.policy_net.get_action_and_value(obs_mb, actions_mb)
                    ratio = (new_logprob - logprobs_mb).exp()
                    surrogate1 = ratio * advantages_mb
                    surrogate2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages_mb
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                    value_clipped = values_mb + torch.clamp(new_value - values_mb, -self.clip_coef, self.clip_coef)
                    value_losses = (new_value - returns_mb) ** 2
                    value_losses_clipped = (value_clipped - returns_mb) ** 2
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                    self.optimizer.step()

        if progress_bar:
            progress_bar.close()
        return cumulative_reward

    def policy(self, state: np.ndarray) -> int:  # type: ignore[override]
        obs_tensor = self._prep_obs(state).unsqueeze(0)
        with torch.no_grad():
            _, _, _, _, dist = self.policy_net.get_action_and_value(obs_tensor)
            action = torch.argmax(dist.probs, dim=-1)
        return int(action.item())

    def test(self, env, render: bool = False):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        actions = []
        states = []
        render_stack = []

        if render:
            frame = env.render()
            if frame is not None:
                render_stack.append(frame)

        while not done:
            action = self.policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            actions.append(action)
            states.append(info.get('obs', obs))
            if render:
                frame = env.render()
                if frame is not None:
                    render_stack.append(frame)

        return episode_reward, actions, states, render_stack

    def q_result(self):
        return 0, 0

    def clone(self):
        clone_agent = CleanRLPPO(
            self.env,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_coef=self.clip_coef,
            update_epochs=self.update_epochs,
            batch_size=self.batch_size,
            minibatch_size=self.minibatch_size,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
        )
        clone_agent.policy_net.load_state_dict(self.policy_net.state_dict())
        return clone_agent
