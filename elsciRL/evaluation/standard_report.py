import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

from elsciRL.analysis.visual_output import VisualOutput
from elsciRL.analysis.tabular_output import TabularOutput
class Evaluation:
    """Calls the Local Environment function and produces analysis reports & figures."""
    def __init__(self, window_size) -> None:
        self.window_size = window_size
        self.timeout = 10
    
    # Training Plots
    def train_report(self, training_results:pd.DataFrame, save_dir:str, show_figures:str):
        """Produces visual and tabular analysis of individual training results."""
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        VisualAnalysis = VisualOutput(training_results, save_dir, show_figures, self.window_size)
        VisualAnalysis.episode_reward_graph()
        VisualAnalysis.cumulative_reward_graph()
        VisualAnalysis.episode_reward_dist_graph()
        VisualAnalysis.average_reward_graph()
        VisualAnalysis.rolling_average_reward_graph()
        VisualAnalysis.avg_return_dist_graph()
        VisualAnalysis.number_actions_graph()
        VisualAnalysis.number_actions_dist_graph()
        VisualAnalysis.runtime_per_episode_graph()
        # Catch runtime issues for dist plot
        #VisualAnalysis.runtime_dist_graph()
        
        TableOutput = TabularOutput(training_results, save_dir)
        TableOutput.save_results()
        return training_results[training_results['episode']==np.max(training_results['episode'])]['cumulative_reward'].iloc[0]
        
    def training_variance_report(self, save_dir:str, show_figures:str):
        """Extracts individual training reports from folders in save_dir and produces summary report if agent types match."""
        # Get first file dir name after 'output' and last
        save_dir_lst = save_dir.split("/")
        env_name = ''
        flag = False
        for st in save_dir_lst:
            if flag == True:
                env_name = env_name+st+'\n'
            if st=='output':
                flag = True

        sub_folders = [name for name in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, name))]
        sub_folders = [name for name in sub_folders if ('train' in name)]
        sub_folders.sort()
        results = {}
        current_list = []
        prior_agent_type = '_'.join(sub_folders[0].split('_')[:-1])
        for name in sub_folders:
            if ('train' in name) | ('test' in name):
                agent_type = '_'.join(name.split('_')[:-1])
            
            if agent_type not in results:
                results[agent_type] = {}

            #Generate list until type no longer matches
            if agent_type == prior_agent_type:
                current_list.append(name)
            else:
                results[prior_agent_type] = current_list
                current_list = [name]
            prior_agent_type = agent_type
        results[prior_agent_type] = current_list # needed to log final agent_type data
        
        # If path name has start position->goal in it then uses seeds, otherwise single start position repeats
        if agent_type.split("_")[-2]!='training':
            training_type = 'seeds'
        else:
            training_type = 'repeats'

        # Speed this up with dict search instead of df iteration
        #cycol = cycle('brcmk')
        cycol = 'brgcmyk'
        line_styles = ['solid','dotted','dashed','dashdot', 'solid','dotted','dashed','dashdot']
        col = 0
        fig, axs = plt.subplots(2,2)
        num_episodes_max = 0
        for agent_type in list(results.keys()): 
            combine_results = pd.DataFrame()
            final_results = pd.DataFrame()
            num_repeats=0
            for dir_path in results[agent_type]:    
                path = save_dir+'/'+dir_path+"/results.csv"
                df = pd.read_csv(path)
                # -----------------
                # Calculate delta as difference between reward obtained from prior episode
                # Segment delta into batches and average to 'smooth' results calculation        
                if (int(np.max(df['episode']*self.window_size))+1) <100:
                    w = int(np.max(df['episode']*self.window_size))+1
                else:
                    w = 100
                reward_delta = []
                for n,r in enumerate(df['episode_reward']): # End when window size no longer fits
                    if n < w:
                        window_reward = np.mean(df['episode_reward'][:n])
                    else:
                        window_reward = np.mean(df['episode_reward'][n-w:n])
                    reward_delta.append(window_reward)
                avg_r_df = pd.DataFrame({'episode':df['episode'], 'avg_R':reward_delta, 'cum_R':df['cumulative_reward'], 'time':df['time_per_episode']})
                combine_results = pd.concat([combine_results, avg_r_df], ignore_index=True)
                num_repeats+=1

            combine_results = combine_results.reset_index()
            num_episode = np.max(combine_results['episode'])
            if num_episode>num_episodes_max:
                num_episodes_max = num_episode
            for epi in range(0,(num_episode+1)):
                final_results = pd.concat([final_results, 
                                    pd.DataFrame.from_records([{ 
                                        'agent': str(agent_type),
                                        #'opponent': combine_results['opponent'].iloc[epi],
                                        'num_repeats': len(combine_results[combine_results['episode']==epi]),
                                        'episode': epi, 
                                        "avg_R_mean": np.mean(combine_results[combine_results['episode']==epi]['avg_R']),
                                        "avg_R_se": np.std(combine_results[combine_results['episode']==epi]['avg_R'], ddof=1) / np.sqrt(np.size(combine_results[combine_results['episode']==epi]['avg_R'])),
                                        "cum_R_mean": np.mean(combine_results[combine_results['episode']==epi]['cum_R']),
                                        "cum_R_se": np.std(combine_results[combine_results['episode']==epi]['cum_R'], ddof=1) / np.sqrt(np.size(combine_results[combine_results['episode']==epi]['cum_R'])),
                                        "time_mean": np.median(combine_results[combine_results['episode']==epi]['time'])
                                        }])],
                                    ignore_index=True)

            avg_r_mean_sorted = np.sort(final_results['avg_R_mean'])
            cdf_mean = 1. * np.arange(len(avg_r_mean_sorted)) / (len(avg_r_mean_sorted) - 1)

            # Plot RL Reward results for each approach
            # 1.1 Summary of total REWARD
            if col >= len(cycol):
                col = 0
            c = cycol[col]
            l = line_styles[col]
            if col <= len(cycol):
                col+=1
            else:
                c = np.random.rand(len(x),3)
                l = 'solid'                
            x =  final_results['episode']
            avg_R = np.array(final_results['avg_R_mean'])
            avg_R_SE = np.array(final_results['avg_R_se'])
            cum_R = np.array(final_results['cum_R_mean'])
            cum_R_SE = np.array(final_results['cum_R_se'])
            time_mean = np.array(final_results['time_mean'])
            
            axs[0,0].plot(x,avg_R, color=c, linestyle=l, label=agent_type)
            axs[0,0].fill_between(x,avg_R-avg_R_SE, avg_R+avg_R_SE, color=c, alpha = 0.2)
            axs[0,1].plot(avg_r_mean_sorted,cdf_mean, color=c, linestyle=l)
            axs[1,0].plot(x,cum_R, color=c, linestyle=l)
            axs[1,0].fill_between(x,cum_R-cum_R_SE, cum_R+cum_R_SE, color=c, alpha = 0.2)
            axs[1,1].hist(time_mean, color=c, alpha=0.25)
            # Save results to csv per agent type
            final_results.to_csv(save_dir+'/training_variance_results_'+str(agent_type.split('__')[0])+'.csv')

        axs[0,0].set_xlabel("Episode")
        axs[0,0].set_ylabel('Reward')
        axs[0,0].axes.get_xaxis().set_ticks([0, num_episodes_max])
        axs[0,0].set_title("Mean and Std. Err. of Rolling Avg. R (window size="+str(w)+ " epi)")
        
        axs[0,1].set_ylabel("Cumulative Probability")
        axs[0,1].set_xlabel("Mean Reward per Episode Window")
        axs[0,1].set_title("CDF of Rolling Average R")
        
        axs[1,0].set_xlabel("Episode")
        axs[1,0].set_ylabel('Cumulative Reward')
        axs[1,0].axes.get_xaxis().set_ticks([0, num_episodes_max])
        axs[1,0].set_title("Cumulative R with Std. Err.")
        
        axs[1,1].set_ylabel("Occurence")
        axs[1,1].set_xlabel("Time")
        axs[1,1].set_title("Dist of Time per Episode")

        #ax1.legend(loc=2, bbox_to_anchor=(-0.05, 0), fancybox=True, shadow=True, framealpha=1)
        #ax2.legend(loc=2, bbox_to_anchor=(0, 1), fancybox=True, shadow=True, framealpha=1)
        if training_type=='repeats':
            fig.suptitle("Training Results for: "+str(env_name)+'Variance over '+str(num_repeats)+" runs with random & independent repeats")
        else:
            fig.suptitle("Training Results for: "+str(env_name)+"with random environment seeds")
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

 
    # Compare the performance of trained agent vs trained agent
    def test_report(self, test_results:pd.DataFrame, save_dir:str, show_figures:str):
        """Produces visual and tabular analysis of testing results."""
        VisualAnalysis = VisualOutput(test_results, save_dir, show_figures, self.window_size)
        VisualAnalysis.episode_reward_graph()
        VisualAnalysis.cumulative_reward_graph()
        VisualAnalysis.episode_reward_dist_graph()
        VisualAnalysis.average_reward_graph()
        VisualAnalysis.rolling_average_reward_graph()
        VisualAnalysis.avg_return_dist_graph()
        VisualAnalysis.number_actions_graph()
        VisualAnalysis.number_actions_dist_graph()
        VisualAnalysis.runtime_per_episode_graph()
        # Catch runtime issues for dist plot
        #VisualAnalysis.runtime_dist_graph()

        TableOutput = TabularOutput(test_results, save_dir)
        TableOutput.save_results()
        return test_results[test_results['episode']==np.max(test_results['episode'])]['cumulative_reward'].iloc[0]

    def testing_variance_report(self, save_dir:str, show_figures:str):
        """Extracts individual training reports from folders in save_dir and produces summary report if agent types match."""
        # Get first file dir name after 'output' and last
        save_dir_lst = save_dir.split("/")
        env_name = ''
        flag = False
        for st in save_dir_lst:
            if flag == True:
                env_name = env_name+st+'\n'
            if st=='output':
                flag = True

        sub_folders = [name for name in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, name))]
        sub_folders = [name for name in sub_folders if ('train' in name)|('test' in name)]
        sub_folders = [name for name in sub_folders if name.split("__")[1].split("_")[0]=='testing']
        print(sub_folders)
        sub_folders.sort()
        results = {}
        current_list = []
        prior_agent_type = ''.join(sub_folders[0].split('__')[0])
        
        for name in sub_folders:
            agent_type = ''.join(name.split('__')[0])
            if agent_type not in results:
                results[agent_type] = {}

             #Generate list until type no longer matches
            if agent_type == prior_agent_type:
                current_list.append(name)
            else:
                results[prior_agent_type] = current_list
                current_list = [name]
            prior_agent_type = agent_type
        results[prior_agent_type] = current_list  # needed to log final agent_type data
        print(results)
        # Speed this up with dict search instead of df iteration
        cycol = 'brgcmyk'
        line_styles = ['solid','dotted','dashed','dashdot', 'solid','dotted','dashed','dashdot']
        col = 0
        fig, axs = plt.subplots(2,2)
        for agent_type in list(results.keys()): 
            combine_results = pd.DataFrame()
            final_results = pd.DataFrame()
            num_repeats=0
            test_folders = results[agent_type]
            #for dir_path in results[agent_type]:  
                #test_folders = [name for name in os.listdir(save_dir+'/'+dir_path) if os.path.isdir(os.path.join(save_dir+'/'+dir_path, name))]
            print(test_folders)
            for testing_path in test_folders:  
                path = save_dir+"/"+testing_path+"/results.csv"
                print(path)
                df = pd.read_csv(path)
                # -----------------
                # Calculate delta as difference between reward obtained from prior episode
                # Segment delta into batches and average to 'smooth' results calculation        
                if (int(np.max(df['episode']*self.window_size))+1) <100:
                    w = int(np.max(df['episode']*self.window_size))+1
                else:
                    w = 100
                reward_delta = []
                for n,r in enumerate(df['episode_reward']): # End when window size no longer fits
                    # At start number of prior episodes is less than window size
                    if n < w:
                        window_reward = np.mean(df['episode_reward'][:n])
                    else:
                        window_reward = np.mean(df['episode_reward'][n-w:n])
                    reward_delta.append(window_reward)
                avg_r_df = pd.DataFrame({'episode':df['episode'], 'avg_R':reward_delta, 'cum_R':df['cumulative_reward'], 'time':df['time_per_episode']})
                combine_results = pd.concat([combine_results, avg_r_df], ignore_index=True)
                num_repeats+=1

            combine_results = combine_results.reset_index()
            num_episode = np.max(combine_results['episode'])
            for epi in range(0,(num_episode+1)):
                final_results = pd.concat([final_results, 
                                    pd.DataFrame.from_records([{ 
                                        'agent': str(agent_type),
                                        #'opponent': combine_results['opponent'].iloc[epi],
                                        'num_repeats': len(combine_results[combine_results['episode']==epi]),
                                        'episode': epi, 
                                        "avg_R_mean": np.mean(combine_results[combine_results['episode']==epi]['avg_R']),
                                        "avg_R_se":  np.std(combine_results[combine_results['episode']==epi]['avg_R'], ddof=1) / np.sqrt(np.size(combine_results[combine_results['episode']==epi]['avg_R'])),
                                        "cum_R_mean": np.mean(combine_results[combine_results['episode']==epi]['cum_R']),
                                        "cum_R_se": np.std(combine_results[combine_results['episode']==epi]['cum_R'], ddof=1) / np.sqrt(np.size(combine_results[combine_results['episode']==epi]['cum_R'])),
                                        "time_mean": np.median(combine_results[combine_results['episode']==epi]['time'])
                                     }])],
                                    ignore_index=True)
            
            avg_r_mean_sorted = np.sort(final_results['avg_R_mean'])
            cdf_mean = 1. * np.arange(len(avg_r_mean_sorted)) / (len(avg_r_mean_sorted) - 1)

            # Plot RL Reward results for each approach
            # 1.1 Summary of total REWARD
            if col >= len(cycol):
                col = 0
            c = cycol[col]
            l = line_styles[col]
            if col <= len(cycol):
                col+=1
            else:
                c = np.random.rand(len(x),3)
                l = 'solid'

            x =  final_results['episode']
            avg_R = np.array(final_results['avg_R_mean'])
            avg_R_SE = np.array(final_results['avg_R_se'])
            cum_R = np.array(final_results['cum_R_mean'])
            cum_R_SE = np.array(final_results['cum_R_se'])
            time_mean = np.array(final_results['time_mean'])
            # Save results to csv per agent type
            final_results.to_csv(save_dir+'/testing_variance_results_'+str(agent_type)+'.csv')

        axs[0,0].plot(x,avg_R, color=c, linestyle=l, label=agent_type)
        axs[0,0].fill_between(x,avg_R-avg_R_SE, avg_R+avg_R_SE, color=c, alpha = 0.2)
        axs[0,1].plot(avg_r_mean_sorted,cdf_mean, color=c, linestyle=l)
        axs[1,0].plot(x,cum_R, color=c, linestyle=l)
        axs[1,0].fill_between(x,cum_R-cum_R_SE, cum_R+cum_R_SE, color=c, alpha = 0.2)
        axs[1,1].hist(time_mean, color=c, alpha=0.25)

        axs[0,0].set_xlabel("Episode")
        axs[0,0].set_ylabel('Reward')
        axs[0,0].axes.get_xaxis().set_ticks([0, num_episode])
        axs[0,0].set_title("Mean and Std. Err. of Rolling Avg. R (window size="+str(w)+ " epi)")
        
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
        fig.suptitle("Testing Results for fixed agent: "+str(agent_type)+'\n Variance over '+str(num_repeats)+" runs with randomly generated environment seeds")
        fig.legend(loc='upper right', fancybox=True, shadow=True, framealpha=1)
        fig.set_size_inches(12, 8)
        fig.tight_layout()
        if show_figures == 'Y':
            plt.show(block=False)
            plt.pause(5)
            fig.savefig(save_dir+'/testing_variance_comparison.png', dpi=fig.dpi)
            plt.close()
        else:
            fig.savefig(save_dir+'/testing_variance_comparison.png', dpi=fig.dpi)
            plt.close()



    def convergence_analysis(self, results_table_dir:str, save_dir:str, show_figures:str):
        """Produces visual and tabular analysis of individual training results."""
        training_results = pd.read_csv(results_table_dir+'/results.csv')
        training_results.opponent = training_results.opponent.astype(str)
        
        VisualAnalysis = VisualOutput(training_results, save_dir, show_figures, self.window_size)
        VisualAnalysis.convergence_analysis_graph()