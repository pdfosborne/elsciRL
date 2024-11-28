from datetime import datetime
import os
import urllib.request
import json 
import httpimport
# Local imports
from elsciRL.benchmarking_suite.imports import Applications
from elsciRL.benchmarking_suite.default_agent import DefaultAgentConfig

# Imports needed for applications to run
# import numpy as np
# import pandas as pd
# from typing import Dict, List
# import torch
# from torch import Tensor
# from elsciRL.encoders.poss_state_encoded import StateEncoder
# from elsciRL.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

# elsciRL Imports
from elsciRL import COMBINED_VARIANCE_ANALYSIS_GRAPH
from elsciRL import STANDARD_RL
from elsciRL.experiments.GymExperiment import GymExperiment

class test:
    """Simply applications class to run a setup tests of experiments.
        - Experiment type: 'standard' or 'gym' experiment depending on agent types selected
        - Problem selection: problems to run in format {''problem1': ['config1', 'config2'], 'problem2': ['config1', 'config2']}
        - Agent config: custom agent configurations

    Applications:
        - Sailing: {'easy', 'medium'}
    """
    def __init__(self, experiment_type:str='standard', 
                 problem_selection:dict={}, agent_config:dict={}, 
                 save_results:bool=True, save_dir:str=None) -> None:
        self.experiment_type = experiment_type
        imports = Applications()
        self.imports = imports.data
        self.problem_selection = problem_selection
        if agent_config == {}:
            agent_config = DefaultAgentConfig()
            self.ExperimentConfig = agent_config.data  
        else:
            self.ExperimentConfig = agent_config
        
        # Save results to a directory
        if save_results:
            cwd = os.getcwd()
            if save_dir is None:
                time = datetime.now().strftime("%d-%m-%Y_%H-%M")
                if not os.path.exists(cwd+'/elsciRL-BENCHMARK-output'):
                    os.mkdir(cwd+'/elsciRL-BENCHMARK-output')
                self.save_dir = cwd+'/elsciRL-BENCHMARK-output/'+str('test')+'_'+time 
            else:
                self.save_dir = save_dir
        # ---
        # Extract data from imports
        self.current_test = {}
        adapters = self.ExperimentConfig['adapter_select']
        for problem in list(self.problem_selection.keys()):
            if problem not in self.imports:
                raise ValueError(f"Problem {problem} not found in the setup tests.")
            else:
                self.current_test[problem] = {}
                # TODO: LOCK DOWN TO A SPECIFIC COMMIT FOR SAFTEY ON IMPORTS
                # current_test = {'problem1': {'engine':engine.py, 'local_configs': {'config1':config.json, 'config2':config.json}, 'adapters': {'adapter1':adapter.py, 'adapter2':adapter.py}}}
                root = 'https://raw.githubusercontent.com/'+ self.imports[problem]['github_user'] + "/" + self.imports[problem]['repository'] + "/" + self.imports[problem]['commit_id']
                # NOTE - This requires repo to match structure with engine inside environment folder
                engine_module = httpimport.load(self.imports[problem]['engine_filenames'], root+'/environment') 
                self.current_test[problem]['engine'] = engine_module.Engine
                # - Subset selection of adapters
                if problem not in adapters:
                    raise ValueError(f"Problem {problem} not found in the adapter selection.")
                else:    
                    problem_adapter = adapters[problem]
                    # - Get all adapters from agent config input
                    # -- Adapter_list is unique to decide the imports
                    # -- Adapter_select is the list to match agent input used to run the benchmark
                    adapter_list = []
                    adapter_select = []
                    for agent_compatible in problem_adapter:
                        for adapter in problem_adapter[agent_compatible]:
                            if problem_adapter[agent_compatible][adapter] == True:
                                if adapter not in adapter_list:
                                    adapter_list.append(adapter)
                                if agent_compatible in self.ExperimentConfig['agent_select']:
                                    adapter_select.append(adapter)

                self.current_test[problem]['adapters'] = {}
                for selected_adapter in adapter_list:
                    if adapter not in self.imports[problem]['adapter_filenames']:
                        raise ValueError(f"Adapter {adapter} not found in the setup tests for problem {problem}.")
                    else:
                        # NOTE - This requires repo to match structure with engine inside adapters folder
                        adapter_module = httpimport.load(self.imports[problem]['adapter_filenames'][selected_adapter], root+'/adapters')   
                        try:
                            self.current_test[problem]['adapters'][selected_adapter] = adapter_module.DefaultAdapter
                        except:
                            self.current_test[problem]['adapters'][selected_adapter] = adapter_module.LanguageAdapter

                # - Subset selection of configs
                for config in self.problem_selection[problem]:
                    if config not in self.imports[problem]['local_config_filenames']:
                        raise ValueError(f"Configuration {config} not found in the setup tests suite for problem {problem}.")
                    else:
                        self.current_test[problem]['local_configs'] = {}
                        # - Replace adapter select with extracted one from agent_config input  
                        local_config = json.loads(urllib.request.urlopen(root+'/benchmark/'+self.imports[problem]['local_config_filenames'][config]).read())
                        local_config['adapter_select'] = adapter_select
                        self.current_test[problem]['local_configs'][config] = local_config
                                        
    
    def run(self):

        for problem in list(self.current_test.keys()):
            engine = self.current_test[problem]['engine']
            adapters = self.current_test[problem]['adapters']
            print(self.current_test[problem])
            for local_config in list(self.current_test[problem]['local_configs'].keys()):
                ProblemConfig = self.current_test[problem]['local_configs'][local_config]

                if self.experiment_type.lower() == 'standard':
                    exp = STANDARD_RL(Config=self.ExperimentConfig, ProblemConfig=ProblemConfig, 
                                Engine=engine, Adapters=adapters,
                                save_dir=self.save_dir, show_figures = 'No', window_size=0.1)
                elif self.experiment_type.lower() == 'gym':
                    exp = GymExperiment(Config=self.ExperimentConfig, ProblemConfig = ProblemConfig,
                       Engine=engine, Adapters=adapters,
                       save_dir=self.save_dir, show_figures = 'Yes', window_size=0.1)
                else:
                    raise ValueError(f"Experiment type {self.experiment_type} not found.")
                
                # Run train/test operations
                exp.train()  
                exp.test()


    def analyze(self):
        # Combines training and testing results into a single graph
        COMBINED_VARIANCE_ANALYSIS_GRAPH(results_dir=self.save_dir, analysis_type='training')
        COMBINED_VARIANCE_ANALYSIS_GRAPH(results_dir=self.save_dir, analysis_type='testing')