from datetime import datetime
import pandas as pd
import os
# ====== elsciRL IMPORTS =========================================
# ------ Train/Test Function Imports ----------------------------
from elsciRL.experiments.standard import Experiment as STANDARD_RL
# ====== LOCAL IMPORTS ==========================================
# ------ Local Environment --------------------------------------
from elsciRL.examples.environments.elsciRL_sailing import Engine as Sailing
from elsciRL.examples.environments.gym_frozenlake import Engine as GymFrozenLake
# ------ Local Adapters -----------------------------------------
from elsciRL.examples.adapters.elsciRL_sailing_default import DefaultAdapter as SailingDefault
from elsciRL.examples.adapters.elsciRL_sailing_language import LanguageAdapter as SailingLanguage
from elsciRL.examples.adapters.gym_frozenlake_default import DefaultAdapter as GymFrozenLakeDefault
from elsciRL.examples.adapters.gym_frozenlake_language import LanguageAdapter as GymFrozenLakeLanguage
# ------ Benchmark Fixed Config -----------------------------------------------
# Meta parameters
from elsciRL.examples import experiment_config as ExperimentConfig
# Local Parameters
from elsciRL.examples.local_configs import sailing_config_local as SailingLocalConfig
from elsciRL.examples.local_configs import gym_frozenlake_config_local as GymFrozenLakeLocalConfig
# --------------------------------------------------------------------
# ------ Visual Analysis -----------------------------------------------
from elsciRL.analysis.combined_variance_visual import combined_variance_analysis_graph as COMBINED_VARIANCE_ANALYSIS_GRAPH

class DemoExperiment:
    def __init__(self):
        # Lookup tables
        self.EXAMPLE_ENGINES:dict = {'sailing':Sailing,
                        'frozenlake':GymFrozenLake}
        self.ADAPTERS:dict = {'sailing':{'Default':SailingDefault,'Language':SailingLanguage},
                    'frozenlake':{'Default':GymFrozenLakeDefault,'Language':GymFrozenLakeLanguage}}
        self.CONFIGS:dict = {'sailing':SailingLocalConfig.LocalConfigData, 
                'frozenlake':GymFrozenLakeLocalConfig.LocalConfigData}
        # Default settings
        self.ExperimentConfig = ExperimentConfig.ExperimentConfigData
        self.experiment_settings:list=['sailing','frozenlake']
        self.num_train_epi:int = 0 # hard over-ride for developers
        self.adapter_overwrite:list = []
        # Create output directory if it doesn't exist
        self.cwd = os.getcwd()+'/elsciRL-EXAMPLE-output'
        if not os.path.exists(self.cwd):
            os.mkdir(self.cwd)
    
    def help(self):
        config = """
        Configures the experiment settings, edit any of the following inputs:
            - num_train_episodes: Number of training episodes
            - num_train_repeats: Number of training repeats
            - num_test_episodes: Number of test episodes
            - number_test_repeats: Number of test repeats
            - Qlearntab_params: Dictionary of Q-learning parameters
                |--> e.g. {alpha:0.1, gamma:0.95, epsilon:0.2, epsilon_step:0.01}
            """
        print(config)

    def input(self):
        # ----- User Input -----
        # 1. Number training episodes
        print("Please enter the number of ... (skip to use default) ")
        num_train_epi = input('\t - Training episodes: ')
        if num_train_epi == '':
            num_train_epi = 1000
        else:
            num_train_epi = int(num_train_epi)

        # Update experiment config
        self.num_train_epi = num_train_epi
        # ----------------------

    def config(self, num_train_episodes:int=100, num_train_repeats:int=5,
        test_agent_type:str='all', num_test_episodes:int=25, number_test_repeats:int=5,
        agent_select:list=['Qlearntab'],
        adapter_select:list=['Default'],
        Qlearntab_params:dict={
            'alpha':0.1, 'gamma':0.95, 'epsilon':0.2, 'epsilon_step':0.01}
                ):
        
        # Update Experiment Config
        self.num_train_epi = num_train_episodes
        self.ExperimentConfig['number_training_repeats'] = num_train_repeats
        self.ExperimentConfig['test_agent_type'] = test_agent_type
        if num_test_episodes !=0:
            self.ExperimentConfig['number_test_episodes'] = num_test_episodes
        self.ExperimentConfig['number_test_repeats'] = number_test_repeats
        self.ExperimentConfig['agent_select'] = agent_select
        self.ExperimentConfig['agent_parameters']['Qlearntab'] = Qlearntab_params
        # Pass adapter select to local configs
        if adapter_select != ['Default', 'Language']:
            self.adapter_overwrite = adapter_select

    def results_save_dir(self):
        # Specify save dir
        # - Needs to be performed here in case user changes parameters and re-runs
        time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.save_dir = self.cwd+'/test_'+time 
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        # ---

    def experiment(self, problem:str, exp_save_dir:str):
        ExperimentConfig = self.ExperimentConfig
        LocalConfig = self.CONFIGS[problem]
        if len(self.adapter_overwrite)>0:
            LocalConfig['adapter_select'] = self.adapter_overwrite
        if self.num_train_epi != 0:
            ExperimentConfig['number_training_episodes'] = self.num_train_epi
            if int(self.num_train_epi/10) > 10:
                ExperimentConfig['number_test_episodes'] = int(self.num_train_epi/10)
            else:
                ExperimentConfig['number_test_episodes'] = 10
        # --------------------------------------------------------------------
        # Flat Baselines
        exp = STANDARD_RL(Config=ExperimentConfig, ProblemConfig=LocalConfig, 
                    Engine=self.EXAMPLE_ENGINES[problem], Adapters=self.ADAPTERS[problem],
                    save_dir=exp_save_dir, show_figures = 'No', window_size=0.1)
        # --------------------------------------------------------------------
        return exp

    def run(self, problem:str|list=['frozenlake','sailing']):        # Added problem selection and num train episodes to run function
        # |--> Convert to list if needed
        if type(problem) == str:
            experiment_settings = [problem]
        elif type(problem) == list:
            experiment_settings = problem
        else:
            print("ERROR: Please input a 'problem' as string or [problem1, problem2] as list.")
        self.experiment_settings = experiment_settings
        
        self.results_save_dir()
        self.results_dir = []
        for chosen_problem in experiment_settings:
            save_dir = self.save_dir + '/'+chosen_problem
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            self.results_dir.append(save_dir)
            print("\n \n --------------------------------------------------")
            print('Training and Testing on {p} environment'.format(p=chosen_problem))
            print("-------------------------------------------------- \n \n ")
            exp = self.experiment(chosen_problem, save_dir)
            exp.train()
            exp.test()
            exp.render_results()

    def evaluate(self):
        COMBINED_VARIANCE_ANALYSIS_GRAPH(self.save_dir, 'TRAINING', show_figures='Yes')
        COMBINED_VARIANCE_ANALYSIS_GRAPH(self.save_dir, 'TESTING', show_figures='Yes')