import random
import pickle
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict

import torch
from torch import Tensor

from elsciRL.agents.agent_abstract import QLearningAgent


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TableQLearningAgent(QLearningAgent):
    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 epsilon_step:float):
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon_reset = epsilon
        self.epsilon: float = epsilon
        self.epsilon_step: float = epsilon_step
        self.q_table: dict = {}
        self.q_table_max: dict = {}
        self.total_q: float = 0
        self.q_size: int = 0
        self.q_zeros: int = 0
        self.debugging: bool = False

    def save(self) -> List[dict]:
        return [self.q_table, self.q_table_max]
    
    def load(self, saved_agent:List[dict]=[None,None]):
        # Override init q table with pre saved info
        self.q_table:dict = saved_agent[0]
        self.q_table_max:dict = saved_agent[1]
        if not self.q_table:
            self.q_table = {}
        if not self.q_table_max:
            self.q_table_max = {}

    def exploration_parameter_reset(self):
        self.epsilon = self.epsilon_reset

    def clone(self):
        clone = pickle.loads(pickle.dumps(self))
        clone.epsilon = self.epsilon_reset
        return clone

    # Fixed order of variables
    def policy(self, state: Tensor, legal_actions: list) -> str:
        """Agent's decision making for next action based on current knowledge and policy type"""
        # Initialise the agent_id entry if not seen
        state = state.cpu().numpy().flatten()
        state_tuple = tuple(state)
        # Epsilon greedy action selection
        # - Early exploration
        rng = np.random.rand()
        # - State not seen before)
        if state_tuple not in self.q_table:
            if self.debugging:
                print("\n --- State not seen before - ", state_tuple)
            self.q_table[state_tuple] = {}
            action_code = random.choice(legal_actions)
        # - Epsilon greedy random selection
        elif rng < self.epsilon: 
            action_code = random.choice(legal_actions)
            if self.epsilon > 0:
                self.epsilon = self.epsilon - (self.epsilon*self.epsilon_step) # Added epsilon step reduction to smooth exploration to greedy
                if self.epsilon < 0:
                    self.epsilon = 0
        # - No known actions
        elif not self.q_table[state_tuple]:
            if self.debugging:
                print("\n --- No known actions for state - ", state_tuple)
            action_code = random.choice(legal_actions)
        # - Current state not in q_table_max
        elif not self.q_table_max[state_tuple]:
            if self.debugging:
                print("\n --- State not in q_table_max - ", state_tuple)
            action_code = random.choice(legal_actions)
        # --------------------------------------
        # - If best action has zero value, pick random action from unexplored actions
        elif (list(self.q_table_max[state_tuple].items())[0][1]==0):
            # Start with actions not seen before
            legal_actions_filtered = [action for action in legal_actions if action not in self.q_table[state_tuple]]
            if len(legal_actions_filtered) == 0:
                # Then add actions with zero value
                for action in self.q_table[state_tuple].keys():
                    if self.q_table[state_tuple][action] == 0:
                        legal_actions_filtered.append(action)
            # Sample from this new filtered list of actions
            if len(legal_actions_filtered) > 0:
                if self.debugging:
                    print("\n --- Best known action value is zero, following actions have no knowledge: ")
                    print(legal_actions_filtered)
                action_code = random.choice(legal_actions_filtered)
            else:
                action_code = random.choice(legal_actions)
        # - have to have an action we know this is good -> Moved into new q_table_max check
        elif (list(self.q_table_max[state_tuple].items())[0][1]<0):
            if self.debugging:
                print("\n --- Best known action value less than zero: ")
                print(list(self.q_table_max[state_tuple].items())[0])
            action_code = random.choice(legal_actions)

        # -- Not sure if this is helping learning, 
        # -- logically we dont wan't agent making bad actions but small punishment makes 
        # -- all actions bad before more exploration   
        # --------------------------------------
        # - Epsilon Greedy best action selection 
        else:
            #Now uses q_table_max to lookup best known action
            action_code = list(self.q_table_max[state_tuple].items())[0][0]
            if self.debugging:
                print("\n --- Best known action is - ", action_code)
                print(list(self.q_table_max[state_tuple].items())[0])
            # Extra step in case top choice is invalid
            # -- Somewhat expensive but likely to be rare and a limited number of actions to iterate over
            if action_code not in legal_actions:
                logger.info(" ")
                logger.info("Best known action not a valid move, searching for next best alternative for action - ", action_code)
                state_action_df = {}
                i = 0
                for action,value in self.q_table[state_tuple].items():
                    state_action_df[action] = value
                    i = i+1
                state_action_df_sorted = dict(sorted(state_action_df.items(), key=lambda item: item[1], reverse=True))
                action_found = False
                i=0
                for action in list(state_action_df_sorted.keys()):
                    if action in legal_actions:
                        action_code = action
                        action_found = True
                        break
                    else:
                        i = i+1
                        logger.info("Top " + str(i) + " action not a valid move, searching for next best alternative")
                        continue
                # If no valid moves exists, pick random action
                if action_found == False:
                    action_code = random.choice(legal_actions)
        return action_code

    # We now break agent into a policy choice, action is taken in game_env then next state is used in learning function
    def learn(self, state: Tensor, next_state: Tensor, r_p: float, action_code: str) -> float:
        """Given action is taken, agent learns from outcome (i.e. next state)"""
        # Get q value for current state-action pairs
        # - If state-action pair unseen, add this to q_table with value initialized to 0
        state = state.cpu().numpy().flatten()
        state_tuple = tuple(state)
        next_state = next_state.cpu().numpy().flatten()
        next_state_tuple = tuple(next_state)

        if action_code not in self.q_table[state_tuple]:
            q = 0
            self.q_table[state_tuple][action_code] = 0
            self.q_size += 1
            self.q_zeros += 1
        # - Else simply select known value
        else:
            q = self.q_table[state_tuple][action_code]

        # Extract Q value of next state and apply update to current state-action pair
        if next_state_tuple not in self.q_table:
            q_update = q + self.alpha * (r_p + self.gamma * 0 - q)
        else:
            ns_max_q = max(self.q_table[next_state_tuple].values()) if (list(self.q_table[next_state_tuple].values())) else 0
            q_update = q + self.alpha * (r_p + self.gamma * ns_max_q - q)
        # Update the total Q value tallying (optimizing the results calculation).
        self.q_tally_update(state_tuple, action_code, q_update)
        # Now we replace the state-action lookup with new value and update the total tallying
        self.q_table[state_tuple][action_code] = q_update
        
        # New max q table to only log the best action in each state to save runtime
        if state_tuple not in self.q_table_max:
            self.q_table_max[state_tuple] = {}
            self.q_table_max[state_tuple][action_code] = q_update
        elif q_update > list(self.q_table_max[state_tuple].items())[0][1]:
            self.q_table_max[state_tuple] = {}
            self.q_table_max[state_tuple][action_code] = q_update

        return q_update

    # Merged these into one function to save time
    def q_result(self):
        """Summarizes all known values in Q table."""
        q_size = self.q_size
        total_q = self.total_q
        q_zeros = self.q_zeros
        
        # Ignores state-actions pairs with no knowledge
        if (q_size - q_zeros) != 0:
            mean_q = total_q/(q_size - q_zeros)
        else:
            mean_q = 0

        return total_q, mean_q

    def q_tally_update(self, state_tuple: Tuple[float], action_code: str, q_update: float):
        self.total_q += q_update - self.q_table[state_tuple][action_code]
        self.q_zeros -= int(self.q_table[state_tuple][action_code] == 0 and q_update != 0)
        self.q_zeros += int(self.q_table[state_tuple][action_code] != 0 and q_update == 0)