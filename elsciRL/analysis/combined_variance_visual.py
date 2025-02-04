import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 


def combined_variance_analysis_graph(results_dir:str='', analysis_type='training', 
                            show_figures:str='N', results_to_show:str='all',
                            experiment_names:list=[]):
    
    if results_dir == '':
        raise ValueError("Save directory not specified.")
    analysis_type = analysis_type.lower() # lowercase analysis type input
    # Get sub-dir for each problem-experiment type
    instruction_folders = [os.path.join(results_dir, instr) for instr in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, instr))]
    print(instruction_folders)
    variance_results = {}
    for instr_folder_dir in instruction_folders:
        instr_id = instr_folder_dir.split('/')[-1]
        if instr_id not in variance_results.keys():
            variance_results[instr_id] = {}
        problem_folders = [name for name in os.listdir(instr_folder_dir) if os.path.isdir(os.path.join(instr_folder_dir, name))]
        # Find experiment folders
        # - Capture case where there is only one experiment type
        # and so wont have sub-directory for experiments to search
        for experiment_dir in problem_folders:
            if analysis_type == 'training':
                experiment_name = experiment_dir+'_training'
                file_names = [name for name in os.listdir(instr_folder_dir+'/'+experiment_dir) if name[0:25] == 'training_variance_results']
            elif analysis_type == 'testing':
                experiment_name = experiment_dir+'_testing'
                file_names = [name for name in os.listdir(instr_folder_dir+'/'+experiment_dir) if name[0:24] == 'testing_variance_results']
            else:
                raise ValueError("Analysis type must be either 'training' or 'testing'.")

            if experiment_name not in variance_results[instr_id].keys():
                variance_results[instr_id][experiment_name] = {}

            for file in file_names:
                results = pd.read_csv(instr_folder_dir+'/'+experiment_dir+'/'+file)
                agent = results['agent'].iloc[0].split('__')[0]
                if agent not in variance_results[instr_id][experiment_name].keys():
                    variance_results[instr_id][experiment_name][agent] = {}
                variance_results[instr_id][experiment_name][agent]['results'] = results
                variance_results[instr_id][experiment_name][agent]['num_repeats'] = results['num_repeats'].iloc[0]
    
    cycol = 'brgcmykbrgcmyk'
    line_styles = ['solid','dotted','dashed','dashdot', 'solid','dotted','dashed','dashdot',
                   'dashdot','dashdot','dashdot','dashdot','dashdot','dashdot','dashdot','dashdot']
    col = 0
    if results_to_show == 'simple':
        fig, axs = plt.subplots(1,1)
        for i,instr_id in enumerate(list(variance_results.keys())):
            for n,experiment in enumerate(list(variance_results[instr_id].keys())):
                for ag,agent in enumerate(list(variance_results[instr_id][experiment].keys())):
                    results = variance_results[instr_id][experiment][agent]['results']
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

                    if 'instr_exp' in str(experiment).lower():
                        label = str(instr_id) + ' - ' + str(agent)
                    else:
                        label = 'No Instr - ' + str(agent)

                    axs.plot(x,avg_R, color=c, linestyle=l, label=label)
                    if len(experiment_names) > 0:
                        axs.set_label(experiment_names[n])
                    axs.fill_between(x,avg_R-avg_R_SE, avg_R+avg_R_SE, color=c, alpha = 0.2)
                
        axs.set_xlabel("Episode")
        axs.set_ylabel('Reward')
        axs.axes.get_xaxis().set_ticks([0, num_episode])
        axs.set_title("Mean Reward per Episode")
        fig.legend(loc='upper left', fancybox=True, shadow=True, 
                   framealpha=1, prop={'size': 14})
        fig.set_size_inches(12, 8)
        
    else:
        fig, axs = plt.subplots(2,2)
        for i,instr_id in enumerate(list(variance_results.keys())):
            for n,experiment in enumerate(list(variance_results[instr_id].keys())):
                for ag,agent in enumerate(list(variance_results[instr_id][experiment].keys())):
                    results = variance_results[instr_id][experiment]['results']
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

                    if 'instr_exp' in str(experiment).lower():
                        label = str(instr_id) + ' - ' + str(agent)
                    else:
                        label = 'No Instr - ' + str(agent)

                    axs[0,0].plot(x,avg_R, color=c, linestyle=l, label=label)
                    axs[0,0].fill_between(x,avg_R-avg_R_SE, avg_R+avg_R_SE, color=c, alpha = 0.2)
                    axs[0,1].plot(avg_r_mean_sorted,cdf_mean, color=c, linestyle=l)
                    axs[1,0].plot(x,cum_R, color=c, linestyle=l)
                    axs[1,0].fill_between(x,cum_R-cum_R_SE, cum_R+cum_R_SE, color=c, alpha = 0.2)
                    axs[1,1].hist(time_mean, color=c, alpha=0.25)
                    if len(experiment_names) > 0:
                        axs.set_label(experiment_names[n])

        axs[0,0].set_xlabel("Episode")
        axs[0,0].set_ylabel('Reward')
        axs[0,0].axes.get_xaxis().set_ticks([0, num_episode])
        axs[0,0].set_title("Mean and Std. Err. of Rolling Avg. R epi)")
        
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
        fig.suptitle("COMBINED VARIANCE ANALYSIS")
        fig.legend(loc='upper right', fancybox=True, shadow=True, framealpha=1)
        fig.set_size_inches(12, 8)
        fig.tight_layout()

    if show_figures == 'Y':
        fig.savefig(results_dir+'/variance_comparison_'+str(analysis_type)+'.png', dpi=100)
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    else:
        fig.savefig(results_dir+'/variance_comparison_'+str(analysis_type)+'.png', dpi=100)
        plt.show()
        plt.close()

