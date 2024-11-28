import numpy as np
import pandas as pd
import scipy.stats as st
# Graphs
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") # fixes error 'failed to allocate bitmap'
import seaborn as sns

class VisualOutput:
    def __init__(self, results_data, save_dir, show_figures, window_size):
        self.results_data = results_data
        self.save_dir = save_dir
        self.show_figures = show_figures
        self.num_episodes = np.max(results_data['episode'].max())
        self.agent_type = results_data['agent'][0].split('_')[0]
        if results_data['opponent'][0] is not None:
            self.opponent_type = results_data['opponent'][0].split('_')[0] # Not currently used but could be for titles?

        self.window_size = window_size

    def episode_reward_graph(self):
        # Plot RL Reward results for each approach
        # 1.1 Summary of total REWARD
        x =  self.results_data['episode']
        y = self.results_data['episode_reward']
        fig, ax1 = plt.subplots()
        ax1.plot(x,y,'r', label=self.agent_type)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel('Reward')
        ax1.axes.get_xaxis().set_ticks([0, self.num_episodes])
        plt.title("Reward of Agent by Episode", y=1.08)
        ax1.legend(loc=2, bbox_to_anchor=(0.8, 1.1), fancybox=True, shadow=True, framealpha=1)
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(self.save_dir+'/episode_R.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(self.save_dir+'/episode_R.png', dpi=fig.dpi)
            plt.close()

    def cumulative_reward_graph(self):
        # Plot RL Reward results for each approach
        # 1.1 Summary of total REWARD
        x =  self.results_data['episode']
        y = self.results_data['cumulative_reward']
        fig, ax1 = plt.subplots()
        ax1.plot(x,y,'r', label=self.agent_type)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel('Cumulative Reward')
        ax1.axes.get_xaxis().set_ticks([0, self.num_episodes])
        plt.title("Cumulative Reward of Agent by Episode", y=1.08)
        ax1.legend(loc=2, bbox_to_anchor=(0.8, 1.1), fancybox=True, shadow=True, framealpha=1)
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(self.save_dir+'/cumulative_R.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(self.save_dir+'/cumulative_R.png', dpi=fig.dpi)
            plt.close()

    def average_reward_graph(self):
        # Plot RL avg in Reward by episode for each approach
        # 1.3 Summary of avg in REWARD
        reward_delta = []
        ci_L_lst = []
        ci_U_lst = []
        for n,r in enumerate(self.results_data['episode_reward']): # End when window size no longer fits
            # At start number of prior episodes is less than window size
            window_reward = np.array(self.results_data['episode_reward'][:n])
            
            mean = np.mean(window_reward)
            if len(window_reward) <=5:
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            elif np.isnan(st.sem(window_reward)):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            elif np.isnan(np.mean(window_reward)):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            # CI produces error if all values the same
            elif np.count_nonzero(window_reward == window_reward[0]) == len(window_reward):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            else:
                ci_L, ci_U = st.t.interval(confidence=0.95, df=len(window_reward)-1, loc=np.mean(window_reward), scale=st.sem(window_reward))

            reward_delta.append(mean)
            ci_L_lst.append(ci_L)
            ci_U_lst.append(ci_U)
    
        x =  self.results_data['episode']
        y = np.array(reward_delta)
        ci_L_lst = np.array(ci_L_lst)
        ci_U_lst = np.array(ci_U_lst)
        fig, ax1 = plt.subplots()
        ax1.plot(x,y,'r', label=self.agent_type)
        ax1.fill_between(x,ci_L_lst, ci_U_lst, color='r', alpha = 0.25)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel('Average Reward')
        ax1.axes.get_xaxis().set_ticks([0, self.num_episodes])
        plt.title("Average Reward over all episodes \n with 95% Confidence Interval", y=1.08)
        ax1.legend(loc=2, bbox_to_anchor=(0.8, 1.1), fancybox=True, shadow=True, framealpha=1)
        fig.tight_layout()
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(self.save_dir+'/avg_R.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(self.save_dir+'/avg_R.png', dpi=fig.dpi)
            plt.close()

    def rolling_average_reward_graph(self):
        # Plot RL avg in Reward by episode for each approach
        # 1.3 Summary of avg in REWARD
        
        # Calculate delta as difference between reward obtained from prior episode
        # Segment delta into batches and average to 'smooth' results calculation        
        if (int(self.num_episodes*self.window_size)+1)<100:
            w = int(self.num_episodes*self.window_size)+1
        else:
            w = 100
        reward_delta = []
        ci_L_lst = []
        ci_U_lst = []
        for n,r in enumerate(self.results_data['episode_reward']): # End when window size no longer fits
            # At start number of prior episodes is less than window size
            if n<w:
                window_reward = np.array(self.results_data['episode_reward'][:n])
            else:
                window_reward = np.array(self.results_data['episode_reward'][n-w:n])
            
            mean = np.mean(window_reward)
            if len(window_reward) <=5:
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            elif np.isnan(st.sem(window_reward)):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            elif np.isnan(np.mean(window_reward)):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            # CI produces error if all values the same
            elif np.count_nonzero(window_reward == window_reward[0]) == len(window_reward):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            else:
                ci_L, ci_U = st.t.interval(confidence=0.95, df=len(window_reward)-1, loc=np.mean(window_reward), scale=st.sem(window_reward))

            reward_delta.append(mean)
            ci_L_lst.append(ci_L)
            ci_U_lst.append(ci_U)
    
        x =  self.results_data['episode']
        y = np.array(reward_delta)
        ci_L_lst = np.array(ci_L_lst)
        ci_U_lst = np.array(ci_U_lst)
        fig, ax1 = plt.subplots()
        ax1.plot(x,y,'r', label=self.agent_type)
        ax1.fill_between(x,ci_L_lst, ci_U_lst, color='r', alpha = 0.25)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel('Average Reward')
        ax1.axes.get_xaxis().set_ticks([0, self.num_episodes])
        plt.title("Rolling Average Reward for window sizes of "+str(w)+" episodes \n with 95% Confidence Interval", y=1.08)
        ax1.legend(loc=2, bbox_to_anchor=(0.8, 1.1), fancybox=True, shadow=True, framealpha=1)
        fig.tight_layout()
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(self.save_dir+'/rolling_avg_R.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(self.save_dir+'/rolling_avg_R.png', dpi=fig.dpi)
            plt.close()
            
    def avg_return_dist_graph(self):
        if int(self.num_episodes*self.window_size) <100:
            w = int(self.num_episodes*self.window_size)
        else:
            w = 100
        reward_delta = []
        for n,r in enumerate(self.results_data['episode_reward']): # End when window size no longer fits
            # At start number of prior episodes is less than window size
            if n<w:
                window_reward = np.mean(self.results_data['episode_reward'][:n])
            else:
                window_reward = np.mean(self.results_data['episode_reward'][n-w:n])
            reward_delta.append(window_reward)
        r_df = pd.DataFrame({'rolling_avg_R':reward_delta})

        sns.ecdfplot(data=r_df, x="rolling_avg_R", color='r', label=self.agent_type)
        #sns.displot(data=self.results_data, x="episode_reward", kind="ecdf", label=self.agent_type)
        plt.title("CDF of the Rolling Avg. Reward per Episode of Each Agent \n Window sizes of "+str(w)+" episodes")
        plt.xlabel("Reward per Episode")
        plt.ylabel("Cumulative Probability")
        plt.legend(fancybox=True, shadow=True, framealpha=1)
        plt.tight_layout()
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            plt.savefig(self.save_dir+'/cum_dist_avg_R.png', bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(self.save_dir+'/cum_dist_avg_R.png', bbox_inches='tight')
            plt.close()
    
    def episode_reward_dist_graph(self):
        # Plot RL Reward results for each approach
        # 1.2 Summary of REWARD per episode
        sns.displot(self.results_data['episode_reward'], label=self.agent_type,color='r')
        plt.title("Distribution of the Reward Obtained per Episode of Each Agent")
        plt.xlabel("Reward per Episode")
        plt.ylabel("Occurrence")
        plt.legend(fancybox=True, shadow=True, framealpha=1)
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            plt.savefig(self.save_dir+'/episode_R_dist.png', bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(self.save_dir+'/episode_R_dist.png', bbox_inches='tight')
            plt.close()

    def number_actions_graph(self):
        #2.1 Number of actions
        x =  self.results_data['episode']
        y = self.results_data['num_actions']
        fig, ax1 = plt.subplots()
        ax1.scatter(x,y,c='b', label=self.agent_type)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel('Number Actions')
        ax1.axes.get_xaxis().set_ticks([0, self.num_episodes])
        plt.title("Number of Actions of Agent per Episode", y=1.08)
        ax1.legend(loc=2, bbox_to_anchor=(0.8, 1.1), fancybox=True, shadow=True, framealpha=1)
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(self.save_dir+'/number_actions.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(self.save_dir+'/number_actions.png', dpi=fig.dpi)
            plt.close()

    def number_actions_dist_graph(self):    
        # 2.2 Plot distr of actions taken per game
        sns.displot(self.results_data['num_actions'], label=self.agent_type,color='b')
        plt.title("Distribution of the Number of Actions Taken per Episode")
        plt.xlabel("Num Actions per Episode")
        plt.ylabel("Occurrence")
        plt.legend(fancybox=True, shadow=True, framealpha=1)
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            plt.savefig(self.save_dir+'/number_actions_dist.png', bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(self.save_dir+'/number_actions_dist.png', bbox_inches='tight')
            plt.close()

    def runtime_per_episode_graph(self):
        # 3.1 Run time of training agent
        x = self.results_data['episode']
        y1 = self.results_data['time_per_episode']
        fig, ax1 = plt.subplots()
        ax1.plot(x,y1,'k', label=self.agent_type)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel('Time (seconds)')
        ax1.axes.get_xaxis().set_ticks([0, self.num_episodes])
        plt.title("Average Runtime per Episode", y=1.08)
        ax1.legend(loc=2, bbox_to_anchor=(0.8, 1.1), fancybox=True, shadow=True, framealpha=1)
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(self.save_dir+'/time_per_episode.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(self.save_dir+'/time_per_episode.png', dpi=fig.dpi)
            plt.close()

    def runtime_dist_graph(self):    
        # 3.2 Plot distr of time taken per game
        sns.displot(self.results_data['time_per_episode'], label=self.agent_type,color='k')
        plt.title("Distribution of the Time Taken per Episode")
        plt.xlabel("Time per Episode")
        plt.ylabel("Occurrence")
        plt.legend(fancybox=True, shadow=True, framealpha=1)
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            plt.savefig(self.save_dir+'/time_per_episode_dist.png', bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(self.save_dir+'/time_per_episode_dist.png', bbox_inches='tight')
            plt.close()        

    @staticmethod
    def combined_variance_analysis_graph(variance_results, save_dir:str, show_figures:str):
        
        cycol = 'brgcmyk'
        line_styles = ['solid','dotted','dashed','dashdot', 'solid','dotted','dashed','dashdot']
        col = 0
        fig, axs = plt.subplots(2,2)
        for results in variance_results:
            num_episode = np.max(results['episode'])
            avg_r_mean_sorted = np.sort(results['avg_R_mean'])
            cdf_mean = 1. * np.arange(len(avg_r_mean_sorted)) / (len(avg_r_mean_sorted) - 1)

            # Plot RL Reward results for each approach
            # 1.1 Summary of total REWARD
            c = cycol[col]
            l = line_styles[col]
            if col <= len(cycol):
                col+=1
            else:
                c = np.random.rand(len(x),3)
                l = 'solid'                
            x =  results['episode']
            avg_R = np.array(results['avg_R_mean'])
            avg_R_SE = np.array(results['avg_R_se'])
            cum_R = np.array(results['cum_R_mean'])
            cum_R_SE = np.array(results['cum_R_se'])
            time_mean = np.array(results['time_mean'])
            
            axs[0,0].plot(x,avg_R, color=c, linestyle=l, label=str(results))
            axs[0,0].fill_between(x,avg_R-avg_R_SE, avg_R+avg_R_SE, color=c, alpha = 0.2)
            axs[0,1].plot(avg_r_mean_sorted,cdf_mean, color=c, linestyle=l)
            axs[1,0].plot(x,cum_R, color=c, linestyle=l)
            axs[1,0].fill_between(x,cum_R-cum_R_SE, cum_R+cum_R_SE, color=c, alpha = 0.2)
            axs[1,1].hist(time_mean, color=c, alpha=0.25)

        axs[0,0].set_xlabel("Episode")
        axs[0,0].set_ylabel('Reward')
        axs[0,0].axes.get_xaxis().set_ticks([0, num_episode])
        axs[0,0].set_title("Mean and Std. Err. of Rolling Avg. R")
        
        axs[0,1].set_ylabel("Cumulative Probability")
        axs[0,1].set_xlabel("Mean Reward per Episode Window")
        axs[0,1].set_title("CDF of Rolling Average R")
        
        axs[1,0].set_xlabel("Episode")
        axs[1,0].set_ylabel('Cumulative Reward')
        axs[1,0].axes.get_xaxis().set_ticks([0, num_episode])
        axs[1,0].set_title("Cumulative R with Std. Err.")
        
        axs[1,1].set_ylabel("Occurence")
        axs[1,1].set_xlabel("Time")
        axs[1,1].set_title("Dist of Time per Episode")

        #ax1.legend(loc=2, bbox_to_anchor=(-0.05, 0), fancybox=True, shadow=True, framealpha=1)
        #ax2.legend(loc=2, bbox_to_anchor=(0, 1), fancybox=True, shadow=True, framealpha=1)
        fig.suptitle("Comparison of Agent Training Results")
        fig.legend(loc='upper right', fancybox=True, shadow=True, framealpha=1)
        fig.set_size_inches(12, 8)
        fig.tight_layout()

        if show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(save_dir+'/training_variance_comparison.png', dpi=100)
            plt.close()
        else:
            fig.savefig(save_dir+'/training_variance_comparison.png', dpi=100)
            plt.close()


    def convergence_analysis_graph(self):
        

        # 1 - Reward graph
        if (int(self.num_episodes*self.window_size)+1)<100:
            w = int(self.num_episodes*self.window_size)+1
        else:
            w = 100
        reward_delta = []
        ci_L_lst = []
        ci_U_lst = []
        for n,r in enumerate(self.results_data['episode_reward']): # End when window size no longer fits
            # At start number of prior episodes is less than window size
            if n<w:
                window_reward = np.array(self.results_data['episode_reward'][:n])
            else:
                window_reward = np.array(self.results_data['episode_reward'][n-w:n])
            
            mean = np.mean(window_reward)
            if len(window_reward) <=5:
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            elif np.isnan(st.sem(window_reward)):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            elif np.isnan(np.mean(window_reward)):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            # CI produces error if all values the same
            elif np.count_nonzero(window_reward == window_reward[0]) == len(window_reward):
                ci_L=np.mean(window_reward)
                ci_U=np.mean(window_reward)
            else:
                ci_L, ci_U = st.t.interval(confidence=0.95, df=len(window_reward)-1, loc=np.mean(window_reward), scale=st.sem(window_reward))

            reward_delta.append(mean)
            ci_L_lst.append(ci_L)
            ci_U_lst.append(ci_U)
    
        x =  self.results_data['episode']
        y = np.array(reward_delta)
        ci_L_lst = np.array(ci_L_lst)
        ci_U_lst = np.array(ci_U_lst)
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle("Rolling Average Reward vs Average Q Values for Agent Training Results", y=0.98)
        fig.set_size_inches(12, 5)
        
        ax1.plot(x,y,'r')
        ax1.fill_between(x,ci_L_lst, ci_U_lst, color='r', alpha = 0.25)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel('Rolling Average Reward')
        ax1.axes.get_xaxis().set_ticks([0, self.num_episodes])
        
        #2 - Average Q graph
        y_2 = self.results_data['q_mean']
        ax2.plot(x,y_2,'k-.')
        ax2.set_xlabel("Episode")
        ax2.set_ylabel('Average Q')
        ax2.axes.get_xaxis().set_ticks([0, self.num_episodes])
        
        #fig.legend(fancybox=True, shadow=True, framealpha=1)
        fig.tight_layout()
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(self.save_dir+'/convergence_analysis.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(self.save_dir+'/convergence_analysis.png', dpi=fig.dpi)
            plt.close()


        # 2 - Summary of total REWARD
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle("Cumulative Reward vs Average Q Values for Agent Training Results", y=0.98)
        fig.set_size_inches(12, 5)

        x =  self.results_data['episode']
        y = self.results_data['cumulative_reward']
        ax1.plot(x,y,'r', label=self.agent_type)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel('Cumulative Reward')
        ax1.axes.get_xaxis().set_ticks([0, self.num_episodes])
        
        #2 - Average Q graph
        y_2 = self.results_data['q_mean']
        ax2.plot(x,y_2,'k-.')
        ax2.set_xlabel("Episode")
        ax2.set_ylabel('Average Q')
        ax2.axes.get_xaxis().set_ticks([0, self.num_episodes])
        
        #fig.legend(fancybox=True, shadow=True, framealpha=1)
        fig.tight_layout()
        if self.show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(self.save_dir+'/convergence_analysis_totalR.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(self.save_dir+'/convergence_analysis_totalR.png', dpi=fig.dpi)
            plt.close()