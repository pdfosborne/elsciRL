import sys
import os
from datetime import datetime
import json
import urllib.request

from elsciRL.application_suite.import_data import Applications
from elsciRL.application_suite.import_tool import PullApplications


from elsciRL.instruction_following.elsciRL_instruction_following import elsciRLOptimize
from elsciRL.experiments.standard import Experiment as StandardExperiment

from elsciRL.analysis.combined_variance_visual import combined_variance_analysis_graph as COMBINED_VARIANCE_ANALYSIS_GRAPH

def agent_selection_data(application:str):
    agent_selection = {
        'Chess': ['Qlearntab','DQN'],
        'Sailing': ['Qlearntab','DQN'],
        'Classroom': ['Qlearntab','DQN'],
        'Gym-FrozenLake': ['Qlearntab','DQN'],
        'Maze': ['Qlearntab','DQN'],
    }
    return agent_selection[application]

def adapter_selection_data(application:str):
    adapter_selection = {
        'Chess': {'numeric_board_mapping','numeric_piece_counter',
                  'active_pieces_language','LLM'},
        'Sailing': {'default','language','LLM'},
        'Classroom': {'default','classroom_A_language','LLM'},
        'Gym-FrozenLake': {'numeric_encoder','language','LLM'},
        'Maze': {'language_default','LLM'},
    }
    return adapter_selection[application]

def instruction_selection_data(application:str):
    # selection = {'local_config_name':['instruction_name']}
    instruction_selection = {
        'Chess': {'Osborne2024_env':['osborne2025']},
        'Sailing': {'easy':['osborne2025']},
        'Classroom': {'classroom_A':['osborne2025']},
        'Gym-FrozenLake': {'Osborne2024_env':['osborne2025']},
        'Maze': {'umaze':['osborne2025umaze'], 'double_t_maze':['osborne2025double_t_maze'], 
                'medium':['osborne2025medium'], 'large':['osborne2025large']},
    }
    return instruction_selection[application]


def local_config_selection_data(application:str):
    local_config_selection = {
        'Chess': ['Osborne2024_env'],
        'Sailing': ['easy'],
        'Classroom': ['classroom_A'],
        'Gym-FrozenLake': ['Osborne2024_env'],
        'Maze': ['umaze', 'double_t_maze', 'medium', 'large'],
    }
    return local_config_selection[application]
     
def experiment_config_selection_data(number_training_episodes:int = 10000):
    # TODO: SET DQN OUTPUT AND HIDDEN SIZE INSIDE ADAPTERS
    experiment_config_selection = {
            'number_training_episodes': number_training_episodes,
            'number_training_repeats': 10,
            'number_training_seeds': 1,

            "instruction_chain": True,
            "instruction_chain_how": "continuous",

            'number_test_episodes': int(number_training_episodes*0.1),
            "test_agent_type": "best",
            "number_test_repeats": 10,

            "agent_parameters":{
                "Qlearntab":{
                    "alpha": 0.05,
                    "gamma": 0.95,
                    "epsilon": 0.2,
                    "epsilon_step":0.01
                    },
                "DQN":{
                        "hidden_size": 128,
                        "learning_rate": 0.001,
                        "gamma": 0.99,
                        "epsilon": 1.0,
                        "epsilon_min": 0.01,
                        "epsilon_decay": 0.995,
                        "memory_size": 10000,
                        "batch_size": 64,
                        "target_update": 10,
                    },
        }
    }
    return experiment_config_selection


class Osborne2025:
    def __init__(self, experiment_name:str|list='ALL', number_training_episodes:int=10000):
        # ------------------------------------------------
        # LOAD APPLICATION DATA
        # Get all application data
        self.application_data = Applications()
        imports = self.application_data.data
        possible_applications = list(imports.keys())
        self.application_data_import = PullApplications()

        # Get chosen applications
        if experiment_name == 'ALL':
            self.chosen_applications = possible_applications
        else:
            if isinstance(experiment_name, list):
                self.chosen_applications = experiment_name
            elif isinstance(experiment_name, str):
                self.chosen_applications = [experiment_name]
            else:
                raise ValueError(f"Experiment name {experiment_name} must be a string or list")

        # Check chosen applications are valid
        for application in self.chosen_applications:
            if application not in possible_applications:
                raise ValueError(f"Application {application} not found in possible applications: {possible_applications}")
        # Pull application data
        self.pull_app_data = self.application_data_import.pull(problem_selection=self.chosen_applications)
        self.config = self.application_data_import.setup()
        # ------------------------------------------------
        # SET UP SAVE DIRECTORY
        if not os.path.exists('./elsciRL-Published-Experiments'):
            os.mkdir('./elsciRL-Published-Experiments')

        time_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
        save_dir = './elsciRL-Published-Experiments/' + str('results') + '_' + time_str
        if not os.path.exists(save_dir):                
            os.mkdir(save_dir)
        self.save_dir = save_dir
        # ------------------------------------------------
        # SET UP EXPERIMENT CONFIG
        self.ExperimentConfig = experiment_config_selection_data(number_training_episodes=number_training_episodes)
        # ------------------------------------------------


    def run(self):
        for application in self.chosen_applications:
            app_save_dir = os.path.join(self.save_dir, application)
            if not os.path.exists(app_save_dir):
                os.mkdir(app_save_dir)
            # Get engine class
            engine_class = self.pull_app_data[application]['engine']
            # ------------------------------------------------
            # Get agent and adapter selection
            agent_selection = agent_selection_data(application)
            self.ExperimentConfig['agent_select'] = agent_selection
            adapters = adapter_selection_data(application)
            ADAPTERS_DICT = {}
            for adapter_name in adapters:
                ADAPTERS_DICT[adapter_name] = self.pull_app_data[application]['adapters'][adapter_name]
            # Get agent adapter selection with a complete mapping
            all_adapters = list(ADAPTERS_DICT.keys())
            agent_adapter_dict = {agent_name: list(all_adapters) for agent_name in agent_selection}
            self.ExperimentConfig['adapter_input_dict'] = agent_adapter_dict
            # ------------------------------------------------
            # Get local config selection
            local_config_list = local_config_selection_data(application)
            for local_config_selection in local_config_list:
                local_config_save_dir = os.path.join(app_save_dir, local_config_selection)
                if not os.path.exists(local_config_save_dir):
                    os.mkdir(local_config_save_dir)
                local_config = self.pull_app_data[application]['local_configs'][local_config_selection]
                # ------------------------------------------------
                # Get preset instruction data
                instruction_selection = instruction_selection_data(application)[local_config_selection]
                for instruction_data_source in instruction_selection:
                    instr_data = self.pull_app_data[application]['instructions'][instruction_data_source]
                    # Instruction filename matches a local config
                    # ------------------------------------------------
                    # Run reinforced experiment
                    for instr_key, instr_data_input in instr_data.items():
                        reinforced_experiment = elsciRLOptimize(
                                Config=self.ExperimentConfig, LocalConfig=local_config, Engine=engine_class, Adapters=ADAPTERS_DICT,
                                save_dir=local_config_save_dir+'/instr', show_figures='No', window_size=0.1,
                                instruction_path=instr_data_input, predicted_path=None, instruction_episode_ratio=0.1,
                                instruction_chain=True, instruction_chain_how='exact')
                        reinforced_experiment.train()
                        reinforced_experiment.test()
                # ------------------------------------------------
                # Run standard experiment
                standard_experiment = StandardExperiment(
                        Config=self.ExperimentConfig, ProblemConfig=local_config, Engine=engine_class, Adapters=ADAPTERS_DICT,
                        save_dir=local_config_save_dir+'/no_instr', show_figures='No', window_size=0.1)
                standard_experiment.train()
                standard_experiment.test()
                        
                # ------------------------------------------------
                COMBINED_VARIANCE_ANALYSIS_GRAPH(results_dir=local_config_save_dir, analysis_type='training', results_to_show='simple')
                COMBINED_VARIANCE_ANALYSIS_GRAPH(results_dir=local_config_save_dir, analysis_type='testing', results_to_show='simple')

if __name__ == "__main__":
    args = sys.argv[1:]
    experiment_name = args[0]
    number_training_episodes = int(args[1])
    osborne_2025 = Osborne2025(experiment_name, number_training_episodes)
    osborne_2025.run()