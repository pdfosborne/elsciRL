import pandas as pd


class ResultsTable:
    def __init__(self, local_setup_info:dict = None) -> None:
        if type(local_setup_info['training_results']) != type(pd.DataFrame()):
            self.agent:list = []
            self.opponent:list =[]
            self.episode:list = []
            self.num_actions:list = []
            self.episode_reward:list = []
            self.cumulative_reward:list = []
            self.time_per_episode:list = []
            self.action_history:list = []
            self.q_total:list = []
            self.q_mean:list = []
            # new
            self.cum_r = 0
        else:
            self.agent:list = local_setup_info['training_results'].agent.tolist()
            self.opponent:list =local_setup_info['training_results'].opponent.tolist()
            self.episode:list = local_setup_info['training_results'].episode.tolist()
            self.num_actions:list = local_setup_info['training_results'].num_actions.tolist()
            self.episode_reward:list = local_setup_info['training_results'].episode_reward.tolist()
            self.cumulative_reward:list = local_setup_info['training_results'].cumulative_reward.tolist()
            self.cum_r = self.cumulative_reward[-1]
            self.time_per_episode:list = local_setup_info['training_results'].time_per_episode.tolist()
            self.action_history:list = local_setup_info['training_results'].action_history.tolist()
            self.q_total:list = local_setup_info['training_results'].q_total.tolist()
            self.q_mean:list = local_setup_info['training_results'].q_mean.tolist()

    def results_per_episode(self,agent_name:str='missing', opponent_name:str='None', episode_num:int=0, action_num:int=0, episode_reward:float=0, time:float=0, episode_action_history:list=[], q_total:float=0, q_mean:float=0):
        self.agent.append(agent_name)
        self.opponent.append(opponent_name)
        self.episode.append(episode_num)
        self.num_actions.append(action_num)
        self.episode_reward.append(episode_reward)
        self.cum_r +=episode_reward
        self.cumulative_reward.append(self.cum_r)
        self.time_per_episode.append(time)
        self.action_history.append(episode_action_history)
        self.q_total.append(q_total)
        self.q_mean.append(q_mean)
        

    def results_table_format(self):
        results= pd.DataFrame({ 
                    'agent': self.agent,
                    'opponent': self.opponent,
                    'episode': self.episode, 
                    'num_actions': self.num_actions, 
                    'episode_reward': self.episode_reward,
                    "cumulative_reward": self.cumulative_reward,
                    "time_per_episode":self.time_per_episode,
                    "action_history": self.action_history,
                    "q_total":self.q_total,
                    "q_mean":self.q_mean})
        return results
    
    def reset(self):
        self.agent:list = []
        self.opponent:list =[]
        self.episode:list = []
        self.num_actions:list = []
        self.episode_reward:list = []
        self.cum_r = 0
        self.cumulative_reward:list = []
        self.time_per_episode:list = []
        self.action_history:list = []
        self.q_total:list = []
        self.q_mean:list = []

    def copy(self):
        results_copy= pd.DataFrame({ 
                    'agent': self.agent.copy(),
                    'opponent': self.opponent.copy(),
                    'episode': self.episode.copy(), 
                    'num_actions': self.num_actions.copy(), 
                    'episode_reward': self.episode_reward.copy(),
                    "cumulative_reward": self.cumulative_reward.copy(),
                    "time_per_episode":self.time_per_episode.copy(),
                    "action_history":self.action_history.copy(),
                    "q_total":self.q_total.copy(),
                    "q_mean":self.q_mean.copy()})
        return results_copy
    
    def load(self, results_copy):
        self.agent:list = results_copy.agent.tolist()
        self.opponent:list = results_copy.opponent.tolist()
        self.episode:list = results_copy.episode.tolist()
        self.num_actions:list = results_copy.num_actions.tolist()
        self.episode_reward:list = results_copy.episode_reward.tolist()
        self.cumulative_reward:list = results_copy.cumulative_reward.tolist()
        self.cum_r = self.cumulative_reward[-1]
        self.time_per_episode:list = results_copy.time_per_episode.tolist()
        self.action_history:list = results_copy.action_history.tolist()
        self.q_total:list = results_copy.q_total.tolist()
        self.q_mean:list = results_copy.q_mean.tolist()


