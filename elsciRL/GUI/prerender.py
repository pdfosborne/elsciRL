# Get application data
from elsciRL.application_suite.import_data import Applications
from elsciRL.application_suite.import_tool import PullApplications

# Get search method
from elsciRL.GUI.elsciRL_demo_search import elsciRLSearch as elsci_search
import os
import json
from datetime import datetime

class Prerender:
    def __init__(self) -> None:
         # Get application data
        imports = Applications().data
        self.possible_applications = list(imports.keys())
        pull_data = PullApplications()
        self.pull_app_data = pull_data.pull(problem_selection=self.possible_applications)
        self.ExperimentConfig = pull_data.setup()

        

    def get_observed_states(self, engine, 
                            selected_application:str='', selected_config:str='', selected_adapter:list=[''],
                            local_config:dict={}, 
                            adapters:dict={},  
                            num_explor_episodes:int=1000):
            
            # Search Save Directory
            if not os.path.exists('./prerender-data'):
                os.mkdir('./prerender-data')
                            
            time = datetime.now().strftime("%d-%m-%Y_%H-%M")
            save_dir = './prerender-data/' #+ str('search') + '_' + time
            if not os.path.exists(save_dir):                
                os.mkdir(save_dir)
            # UPDATE EXPERIMENT CONFIG FOR SEARCH
            # - only use q learn tab agent
            selected_agents =  ['Qlearntab']
            self.ExperimentConfig.update({
                'number_training_episodes': int(num_explor_episodes),
                'agent_select': selected_agents         
            })
            local_config.update({   
                'adapter_select': selected_adapter    
            })

            elsci_run = elsci_search(Config=self.ExperimentConfig,
                                        LocalConfig=local_config,
                                        Engine=engine, Adapters=adapters,
                                        save_dir=save_dir,
                                        number_exploration_episodes=num_explor_episodes,
                                        match_sim_threshold=0.9,
                                        observed_states=None)
            observed_states = elsci_run.search()
            print(f"\nNumber of observed states: {len(observed_states)}")
            # Create save directory structure if not exists
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            if not os.path.exists(os.path.join(save_dir, selected_application)):
                os.mkdir(os.path.join(save_dir, selected_application))

            if not os.path.exists(os.path.join(save_dir, selected_application, selected_adapter[0])):
                os.mkdir(os.path.join(save_dir, selected_application, selected_adapter[0]))
            # Add problem name to save_dir
            save_dir = os.path.join(save_dir, selected_application, selected_adapter[0])


            # Update file name to include problem name, number of episodes, and timestamp
            file_name = f"observed_states_{selected_application}_{selected_config}_{selected_adapter[0]}_{num_explor_episodes}_{time}.txt"
            file_path = os.path.join(save_dir, file_name)

            # Save observed states only
            with open(file_path, 'w') as f:
                json.dump(observed_states, f)


    def run(self):

        # Allow terminal input to select application
        print("Select an application from the following options:")
        for i, app in enumerate(self.possible_applications):
            print(f"{i + 1}: {app}")

        selected_index = int(input("Enter the number of the application you want to select: ")) - 1
        selected_application = self.possible_applications[selected_index]

        # Allow terminal input to select configuration
        print("Select a configuration from the following options:")
        for i, config in enumerate(self.pull_app_data[selected_application]['local_configs']):
            print(f"{i + 1}: {config}")

        config_input_index = int(input("Enter the number of the configuration you want to select: ")) - 1
        config_input = list(self.pull_app_data[selected_application]['local_configs'].keys())[config_input_index]

        # Allow terminal input to select adapter
        adapter_list = self.pull_app_data[selected_application]['adapters']
        print("Select an adapter from the following options:")
        for i, adapter in enumerate(adapter_list):
            print(f"{i + 1}: {adapter}")

        selected_adapter_index = int(input("Enter the number of the adapter you want to select: ")) - 1
        selected_adapter = [list(adapter_list.keys())[selected_adapter_index]]

        # Get data for the selected application
        engine = self.pull_app_data[selected_application]['engine']
        local_config = self.pull_app_data[selected_application]['local_configs'][config_input]
        adapters = self.pull_app_data[selected_application]['adapters']
        
        num_explor_episodes = int(input("Enter the number of the exploration episodes: "))

        print("--------------------------------")
        print("-Selected options-")
        print(f"-- Selected application: {selected_application}")
        print(f"-- Selected configuration: {config_input}")
        print(f"-- Selected adapter: {selected_adapter[0]}")
        print(f"-- Number of exploration episodes: {num_explor_episodes}")
        print("--------------------------------")

        self.get_observed_states(engine, 
                                selected_application, config_input, selected_adapter,
                                local_config, 
                                adapters, 
                                num_explor_episodes)