import numpy as np
import pandas as pd
import os 
import json

def combined_tabular_analysis_results(results_dir:str='', analysis_type='training'):
    if results_dir == '':
        raise ValueError("Save directory not specified.")
    analysis_type = analysis_type.lower() # lowercase analysis type input
    # Get sub-dir for each problem-experiment type
    instruction_folders = [os.path.join(results_dir, instr) for instr in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, instr))]
    variance_results = {}
    for instr_folder_dir in instruction_folders:
        instr_id = instr_folder_dir.split('/')[-1].split('//')[-1].split('\\')[-1].split('\\\\')[-1]
        if instr_id not in variance_results.keys():
            variance_results[instr_id] = {}
        print(f"Processing {instr_id} for {analysis_type} analysis.")
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
                
                # Calculate Mean and Std Dev
                variance_results[instr_id][experiment_name][agent]['num_repeats'] = results['num_repeats'].iloc[0]
                variance_results[instr_id][experiment_name][agent]['number_episodes'] = results.index.max() + 1
                # - rolling avg R per episode
                variance_results[instr_id][experiment_name][agent]['mean'] = results['avg_R_mean'].mean()
                variance_results[instr_id][experiment_name][agent]['median'] = results['avg_R_mean'].median()
                variance_results[instr_id][experiment_name][agent]['std_error'] = results['avg_R_mean'].sem()
                variance_results[instr_id][experiment_name][agent]['std_dev'] = results['avg_R_mean'].std()
                variance_results[instr_id][experiment_name][agent]['variance'] = results['avg_R_mean'].var()
                # - cumulative R per episode
                variance_results[instr_id][experiment_name][agent]['cum_R_mean'] = results['cum_R_mean'].mean()
                variance_results[instr_id][experiment_name][agent]['cum_R_median'] = results['cum_R_mean'].median()
                variance_results[instr_id][experiment_name][agent]['cum_R_std_error'] = results['cum_R_mean'].sem()
                variance_results[instr_id][experiment_name][agent]['cum_R_std_dev'] = results['cum_R_mean'].std()
                variance_results[instr_id][experiment_name][agent]['cum_R_variance'] = results['cum_R_mean'].var()
                # - time avg per episode
                variance_results[instr_id][experiment_name][agent]['time_avg'] = results['time_mean'].mean()

    variance_results_df = pd.DataFrame.from_dict(
        {f"{instr}/{experiment}/{agent}": data for instr, experiments in variance_results.items() 
         for experiment, agents in experiments.items() 
         for agent, data in agents.items()},
        orient='index'
    ).reset_index()
    variance_results_df.columns = ['Instruction/Experiment/Agent', 'Num Repeats', 'Number Episodes', 
                                    'Avg R Mean', 'Avg R Median', 'Avg R Std Error', 'Avg R Std Dev', 'Avg R Variance',
                                    'Cumulative R Mean', 'Cumulative R Median', 'Cumulative R Std Error',
                                    'Cumulative R Std Dev', 'Cumulative R Variance', 'Time Avg']
    # Save the combined results to a CSV file
    combined_results_filename = f"{analysis_type}_combined_results.csv"
    combined_results_path = os.path.join(results_dir, combined_results_filename)
    variance_results_df.to_csv(combined_results_path, index=False)
    