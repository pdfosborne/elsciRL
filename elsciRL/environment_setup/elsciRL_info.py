import torch
from torch import Tensor
import numpy.typing as npt
import random


class elsciRLInfo:
    def __init__(
        self,
        observed_states: dict | None = None,
        experience_sampling: dict | None = None,
        # tensor_index: dict | None = None,
    ) -> None:
        if not experience_sampling:
            self.experience_sampling = {}
        else:
            self.experience_sampling = experience_sampling

        if not observed_states:
            self.observed_states = {}
        else:
            self.observed_states = observed_states

    def observed_state_tracker(
        self,
        engine_observation: Tensor | npt.ArrayLike | list | None = None,
        language_state: Tensor | npt.ArrayLike | list | None = None,
    ):
        """Tracks adapted form of state from engine observation for unsupervised approaches."""
        if engine_observation not in self.observed_states:
            self.observed_states[engine_observation] = language_state

    def experience_sampling_add(
        self,
        engine_observation: Tensor | npt.ArrayLike | list | None = None,
        action: int | str | bool | None = None,
        next_observation: Tensor | npt.ArrayLike | list | None = None,
        reward: float = 0,
        terminated: bool = False,
    ):
        """Adds experience from interaction with the Live environment to sample from."""
        # --------------------------------------------------------------------------
        # Required if input observation is tensor as this cant be used for dict keys
        # - create tuple store transitions
        if type(engine_observation) is Tensor:
            engine_observation = tuple(engine_observation.cpu().numpy().flatten())
        if type(next_observation) is Tensor:
            next_observation = tuple(next_observation.cpu().numpy().flatten())
        # --------------------------------------------------------------------------
        # Get occurrence of current observation+action
        if engine_observation not in self.experience_sampling:
            self.experience_sampling[engine_observation] = {}
        if action not in self.experience_sampling[engine_observation]:
            self.experience_sampling[engine_observation][action] = {}
            self.experience_sampling[engine_observation][action]["obs_a_count"] = 1
        obs_a_count = (
            self.experience_sampling[engine_observation][action]["obs_a_count"] + 1
        )
        self.experience_sampling[engine_observation][action][
            "obs_a_count"
        ] = obs_a_count

        # Get occurrence of next obs given obs+action
        # - Compute prob, reward is static and set on first occurrence
        if next_observation in self.experience_sampling[engine_observation][action]:
            next_obs_count = (
                self.experience_sampling[engine_observation][action][next_observation][
                    "next_obs_count"
                ]
                + 1
            )
            prob = next_obs_count / obs_a_count
            self.experience_sampling[engine_observation][action][next_observation][
                "next_obs_count"
            ] = next_obs_count
            self.experience_sampling[engine_observation][action][next_observation][
                "prob"
            ] = prob
        else:
            self.experience_sampling[engine_observation][action][next_observation] = {}
            self.experience_sampling[engine_observation][action][next_observation][
                "next_obs_count"
            ] = 1
            self.experience_sampling[engine_observation][action][next_observation][
                "prob"
            ] = (1 / obs_a_count)
            self.experience_sampling[engine_observation][action][next_observation][
                "reward"
            ] = reward
            self.experience_sampling[engine_observation][action][next_observation][
                "terminated"
            ] = terminated

    def experience_sampling_legal_actions(
        self, engine_observation: Tensor | npt.ArrayLike | list | None = None
    ):
        """Returns a list of known actions from the experience."""
        if type(engine_observation) is Tensor:
            engine_observation = tuple(engine_observation.cpu().numpy().flatten())
            # state_tuple = tuple(engine_observation)
            # engine_observation = self.tensor_index.index(state_tuple)
        if engine_observation in self.experience_sampling:
            legal_actions = list(self.experience_sampling[engine_observation].keys())
        else:
            legal_actions = None
        return legal_actions

    def experience_sampling_step(
        self,
        engine_observation: Tensor | npt.ArrayLike | list | None = None,
        action: int | str | bool | None = None,
    ):
        """Outcome of action given current observation from sampled experience."""
        # If state-action has not been seen from live system
        engine_observation_shape = None
        next_obs = None
        if type(engine_observation) is Tensor:
            engine_observation_shape = engine_observation.shape
            engine_observation = tuple(engine_observation.cpu().numpy().flatten())

        if action not in self.experience_sampling[engine_observation]:
            next_obs = engine_observation
            reward = 0
            terminated = False
        # Select action from distribution of probabilities
        else:
            cumulative = 0
            rng = random.random()
            for next_obs in self.experience_sampling[engine_observation][action]:
                # first key is just the count of obs+action so skip over this
                if next_obs != "obs_a_count":
                    if (
                        self.experience_sampling[engine_observation][action][next_obs][
                            "prob"
                        ]
                        <= rng
                    ):
                        break
                    else:
                        cumulative += self.experience_sampling[engine_observation][
                            action
                        ][next_obs]["prob"]

            reward = self.experience_sampling[engine_observation][action][next_obs][
                "reward"
            ]
            terminated = self.experience_sampling[engine_observation][action][next_obs][
                "terminated"
            ]

        # --------------------------------------------------------------------------
        # Converts stored obs back from int to tensor to match env
        if (type(next_obs) is tuple) and (
            engine_observation_shape is not None
        ):  # Phil: had to fix AND if statements by separating fully
            next_obs = torch.tensor(next_obs).reshape(engine_observation_shape)
        # --------------------------------------------------------------------------
        return next_obs, reward, terminated
