from datetime import datetime
import os
# ====== elsciRL IMPORTS ===============================================
# ------ EXPERIMENT ----------------------------------------------------
from elsciRL.experiments.standard import Experiment as STANDARD_RL
# ------ Visual Analysis -----------------------------------------------
from elsciRL.analysis.combined_variance_visual import combined_variance_analysis_graph as COMBINED_VARIANCE_ANALYSIS_GRAPH
# ----------------------------------------------------------------------

class DemoExperiment:
    def __init__(self):        
        # Create output directory if it doesn't exist
        self.cwd = os.getcwd()+'/elsciRL-EXAMPLE-output'
        if not os.path.exists(self.cwd):
            os.mkdir(self.cwd)
    
    def help(self):
        help_output = """
        This is a demo experiment script for elsciRL.
        It allows you to run a standard RL experiment on a selected problem from the elsciRL application suite.
        The script will guide you through the process of selecting a problem, configuring the experiment, and running it.
        You can also evaluate the results of the experiment after it has been run.
        Usage:
        1. Run the script.
        2. Follow the prompts to select a problem and configure the experiment.
        3. The experiment will be run and the results will be saved in a directory.
        4. You can evaluate the results by calling the evaluate() method.
        Example:
        >>> demo = DemoExperiment()
        >>> demo.run()
        >>> demo.evaluate()
        """
        print(help_output)

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

    def results_save_dir(self):
        # Specify save dir
        # - Needs to be performed here in case user changes parameters and re-runs
        time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.save_dir = self.cwd+'/test_'+time 
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        # ---

    def experiment(self, problem:str, exp_save_dir:str, num_train_epi:int=0):

        # --- Select local config and experiment config ---
        print("--- LOCAL CONFIGURATION SELECTION ---")
        for i, local_config in enumerate(self.pull_app_data[problem]['local_configs'].keys()):
            print(f"{i+1}. {local_config}")
        local_config_id = input("Please select the local config number (default 1): ")
        if local_config_id.isdigit() and 0 < int(local_config_id) <= len(self.pull_app_data[problem]['local_configs']):
            local_config_id = int(local_config_id) - 1
        else:
            local_config_id = 0
        LocalConfig = self.pull_app_data[problem]['local_configs'][list(self.pull_app_data[problem]['local_configs'].keys())[local_config_id]]
        
        print("\n --- EXPERIMENT CONFIGURATION SELECTION ---")
        for i, experiment_config in enumerate(self.pull_app_data[problem]['experiment_configs'].keys()):
            print(f"{i+1}. {experiment_config}")
        experiment_config_id = input("Please select the experiment config number (default 1): ")
        if experiment_config_id.isdigit() and 0 < int(experiment_config_id) <= len(self.pull_app_data[problem]['experiment_configs']):
            experiment_config_id = int(experiment_config_id) - 1
        else:
            experiment_config_id = 0
        ExperimentConfig = self.pull_app_data[problem]['experiment_configs'][list(self.pull_app_data[problem]['experiment_configs'].keys())[experiment_config_id]]

        if num_train_epi != 0:
            ExperimentConfig['number_training_episodes'] = num_train_epi
            if int(num_train_epi/10) > 10:
                ExperimentConfig['number_test_episodes'] = int(num_train_epi/10)
            else:
                ExperimentConfig['number_test_episodes'] = 10

        # ------------------------------------------------------
        # Adapter Selection
        print("\n --- ADAPTER SELECTION ---")
        for i, adapter in enumerate(self.pull_app_data[problem]['adapters'].keys()):
            if not adapter.startswith('LLM'):
                print(f"{i+1}. {adapter}")
        adapter_id = input("Please select the adapter number (default 1): ")
        if adapter_id.isdigit() and 0 < int(adapter_id) <= len(self.pull_app_data[problem]['adapters']):
            adapter_id = int(adapter_id) - 1
        else:
            adapter_id = 0

        # --------------------------------------------------------------------
        # Set the selected agent
        ExperimentConfig['agent_select'] = ['Qlearntab']
        ExperimentConfig['adapter_select'] = [list(self.pull_app_data[problem]['adapters'].keys())[adapter_id]]
        ExperimentConfig['adapter_input_dict'] = {'Qlearntab': [list(self.pull_app_data[problem]['adapters'].keys())[adapter_id]]}
        if ExperimentConfig['number_training_repeats'] > 1:
            ExperimentConfig['number_training_repeats'] = 5
        if ExperimentConfig['number_training_seeds'] > 1:
            ExperimentConfig['number_training_seeds'] = 5
        # Flat Baselines
        exp = STANDARD_RL(Config=ExperimentConfig, ProblemConfig=LocalConfig, 
                    Engine=self.pull_app_data[problem]['engine'], Adapters=self.pull_app_data[problem]['adapters'],
                    save_dir=exp_save_dir, show_figures = 'No', window_size=0.1)
        # --------------------------------------------------------------------
        return exp

    def run(self):
        # IMPORT HERE SO ITS NOT LOADED ON STARTUP
        from elsciRL.application_suite.import_tool import PullApplications
        self.application_data = PullApplications()
        self.application_list:list=['Classroom', 'Gym-FrozenLake', 'Sailing']
        self.pull_app_data = self.application_data.pull(problem_selection=self.application_list)
        print("\n --- PULLING APPLICATION DATA ---")
        for app in self.application_list:
            print("--------------------------------------------------")
            print(f"Application: {app}")
            print("Engine:", self.pull_app_data[app]['engine'])
            print("Adapters:", self.pull_app_data[app]['adapters'])
            print("Experiment Configs:", self.pull_app_data[app]['experiment_configs'])
            print("Local Configs:", self.pull_app_data[app]['local_configs'])
        print("-------------------------------------------------- \n ")
        # USER INPUTS FOR BASIC SELECTION OPTIONS
        # --- Problem selection ---
        print("\n --- PROBLEM SELECTION ---")
        for i, prob in enumerate(self.application_list):
            print(f"{i+1}. {prob}")
        problem_id = input("Please enter the problem number to run (default 1): ")
        if problem_id.isdigit() and 0 < int(problem_id) <= len(self.application_list):
            problem = self.application_list[int(problem_id) - 1]
        else:
            problem = self.application_list[0]

        # --- TRAINING EPISODES INPUT ---
        num_train_episodes = input("Please enter the number of training episodes (default 1000): ")
        if num_train_episodes == '':
            num_train_episodes = 1000
        try:
            num_train_episodes = int(num_train_episodes)
        except ValueError:
            print("Invalid input for number of training episodes. Using default value of 1000.")
            num_train_episodes = 1000
        # -------------------------------
        
        self.results_save_dir()
        problem_save_dir = self.save_dir + '/' + problem
        if not os.path.exists(problem_save_dir):
            os.mkdir(problem_save_dir)
        print("\n --------------------------------------------------")
        print('Training and Testing on {p} environment'.format(p=problem))
        print("-------------------------------------------------- \n ")
        exp = self.experiment(problem, problem_save_dir, num_train_epi=int(num_train_episodes))
        exp.train()
        exp.test()
        # exp.render_results()

    def evaluate(self):
        COMBINED_VARIANCE_ANALYSIS_GRAPH(self.save_dir, 'TRAINING', show_figures='Yes')
        COMBINED_VARIANCE_ANALYSIS_GRAPH(self.save_dir, 'TESTING', show_figures='Yes')