from datetime import datetime
import os
import urllib.request
import json 
import httpimport
# Local imports
from elsciRL.benchmarking_suite.imports import Applications
from elsciRL.benchmarking_suite.default_agent import DefaultAgentConfig

class PullApplications:
    """Simple applications class to run a setup tests of experiments.
        - Problem selection: problems to run in format ['problem1', 'problem2',...]

    Applications:
        - Sailing: {'easy'},
        - Classroom: {'classroom_A'}
    """
    # TODO: Make it so it pulls all possible configs and adapters from the repo
    def __init__(self) -> None:
        imports = Applications()
        self.imports = imports.data
        
        
    def pull(self, problem_selection:list=[]):
        # Pull all problems if none are selected
        if len(problem_selection)>0:
            self.problem_selection = problem_selection
        else:
            self.problem_selection = list(self.imports.keys())
        # Extract data from imports
        self.current_test = {}
        #adapters = self.ExperimentConfig['adapter_select']
        for problem in list(self.problem_selection):
            engine = self.imports[problem]['engine_filename']
            if problem not in self.imports:
                raise ValueError(f"Problem {problem} not found in the setup tests.")
            else:
                self.current_test[problem] = {}
                # current_test = {'problem1': {'engine':engine.py, 'local_configs': {'config1':config.json, 'config2':config.json}, 'adapters': {'adapter1':adapter.py, 'adapter2':adapter.py}}}
                root = 'https://raw.githubusercontent.com/'+ self.imports[problem]['github_user'] + "/" + self.imports[problem]['repository'] + "/" + self.imports[problem]['commit_id']
                # NOTE - This requires repo to match structure with engine inside environment folder
                engine_module = httpimport.load(engine, root+'/'+self.imports[problem]['engine_folder']) 
                # TODO: Pull class name directly from engine file to be called
                self.current_test[problem]['engine'] = engine_module.Engine
            # ------------------------------------------------
            # - Pull Adapters, Configs and Analysis
            self.current_test[problem]['adapters'] = {}
            for adapter_name, adapter in self.imports[problem]['adapter_filenames'].items():
                adapter_module = httpimport.load(adapter, root+'/'+self.imports[problem]['local_adapter_folder'])   
                # TODO: Pull class name directly from adapter file to be called
                try:
                    self.current_test[problem]['adapters'][adapter_name] = adapter_module.DefaultAdapter
                except:
                    self.current_test[problem]['adapters'][adapter_name] = adapter_module.LanguageAdapter
            # ---
            self.current_test[problem]['local_configs'] = {}
            for config_name,config in self.imports[problem]['local_config_filenames'].items():
                local_config = json.loads(urllib.request.urlopen(root+'/'+self.imports[problem]['local_config_folder']+'/'+config).read())
                self.current_test[problem]['local_configs'][config_name] = local_config
            # ---
            self.current_test[problem]['local_analysis'] = {}
            for analysis_name,analysis in self.imports[problem]['local_analysis_filenames'].items():
                try:
                    local_analysis = httpimport.load(analysis, root+'/'+self.imports[problem]['local_analysis_folder'])  
                    # TODO: Pull class name directly from analysis file to be called 
                    self.current_test[problem]['local_analysis'][analysis_name] = local_analysis.Analysis
                except:
                    print("No analysis file found.")
                    self.current_test[problem]['local_analysis'][analysis_name] = {}
            
            # ------------------------------------------------
            # Pull prerender data
            print("-----------------------------------------------")
            print(problem)
            self.current_test[problem]['prerender_data'] = {}
            if self.imports[problem]['prerender_data_folder'] != '':
                try:
                    for prerender_name, prerender in self.imports[problem]['prerender_data_filenames'].items():
                        if prerender.endswith(('.txt', '.json', '.xml')):
                            data = json.loads(urllib.request.urlopen(root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender).read().decode('utf-8'))
                            self.current_test[problem]['prerender_data'][prerender_name] = data
                    print("Pulling prerender data...")
                except:
                    print(root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender)
                    print("No prerender data found.")
                    self.current_test[problem]['prerender_data'] = {}
            else:
                print("No prerender data found.")
                self.current_test[problem]['prerender_data'] = {}

            # Pull prerender images
            self.current_test[problem]['prerender_images'] = {}
            if self.imports[problem]['prerender_data_folder'] != '':
                try:
                    for image_name, image in self.imports[problem]['prerender_image_filenames'].items():
                        if image.endswith(('.png', '.jpg', '.svg', '.gif')):
                            image_url = root + '/' + self.imports[problem]['prerender_data_folder'] + '/' + image
                            image_data = urllib.request.urlopen(image_url).read()
                            self.current_test[problem]['prerender_images'][image_name] = image_data
                    print("Pulling prerender images...")
                except:
                    print("No prerender images found.")
                    self.current_test[problem]['prerender_images'] = {}
            else:
                print("No prerender images found.")
                self.current_test[problem]['prerender_images'] = {}
            # -----------------------------------------------
        print("-----------------------------------------------")

        return self.current_test

    def setup(self, agent_config:dict={}) -> None:
        if agent_config == {}:
            agent_config = DefaultAgentConfig()
            self.ExperimentConfig = agent_config.data  
        else:
            self.ExperimentConfig = agent_config 

        return self.ExperimentConfig
            