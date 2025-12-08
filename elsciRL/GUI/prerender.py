# Get application data
import numpy as np
from elsciRL.application_suite.import_data import Applications
from elsciRL.application_suite.import_tool import PullApplications

# Get search method
from elsciRL.instruction_following.elsciRL_GUI_search import elsciRLSearch as elsci_search
import os
import json
import torch
from torch import Tensor
from datetime import datetime
import ollama
from tqdm import tqdm
import sys

# Get language encoder
from elsciRL.encoders.language_transformers.MiniLM_L6v2 import LanguageEncoder as MiniLM_L6v2

class Prerender:
    def __init__(self) -> None:
         # Get application data
        imports = Applications().data
        self.possible_applications = list(imports.keys())
        pull_data = PullApplications()
        self.pull_app_data = pull_data.pull(problem_selection=self.possible_applications)
        self.ExperimentConfig = pull_data.setup()
        self.observed_states = None
        self.max_file_size_gb = 1  # Maximum size per file in GB

    def _estimate_size_bytes(self, data):
        """Estimate the size of data in bytes when saved as .pt file."""
        import io
        buffer = io.BytesIO()
        torch.save(data, buffer)
        size = buffer.tell()
        buffer.close()
        return size

    def _save_large_file(self, data, base_file_path):
        """
        Save data to file(s), splitting into parts if larger than max_file_size_gb.
        
        Args:
            data (dict): Dictionary to save
            base_file_path (str): Base file path (will be modified if splitting is needed)
            
        Returns:
            list: List of file names that were created
        """
        max_size_bytes = self.max_file_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        
        # Estimate total size
        print("Estimating data size...")
        total_size = self._estimate_size_bytes(data)
        size_gb = total_size / (1024 ** 3)
        
        print(f"Total data size: {size_gb:.2f} GB")
        
        if total_size <= max_size_bytes:
            # Save as single file
            print(f"Saving as single file: {base_file_path}")
            torch.save(data, base_file_path)
            return [os.path.basename(base_file_path)]
        
        # Need to split into parts
        print(f"Data exceeds {self.max_file_size_gb} GB limit. Splitting into parts...")
        
        # Split dictionary into chunks
        items = list(data.items())
        num_parts = int(np.ceil(total_size / max_size_bytes))
        items_per_part = int(np.ceil(len(items) / num_parts))
        
        saved_files = []
        base_name = base_file_path.replace('.pt', '')
        
        for part_idx in range(num_parts):
            start_idx = part_idx * items_per_part
            end_idx = min((part_idx + 1) * items_per_part, len(items))
            part_data = dict(items[start_idx:end_idx])
            
            part_file_path = f"{base_name}_part{part_idx + 1}.pt"
            part_size = self._estimate_size_bytes(part_data)
            part_size_gb = part_size / (1024 ** 3)
            
            print(f"Saving part {part_idx + 1}/{num_parts}: {os.path.basename(part_file_path)} ({part_size_gb:.2f} GB, {len(part_data)} states)")
            torch.save(part_data, part_file_path)
            saved_files.append(os.path.basename(part_file_path))
        
        print(f"Successfully split data into {num_parts} parts")
        return saved_files

    def _load_large_file(self, file_path):
        """
        Load data from file(s), combining parts if necessary.
        
        Args:
            file_path (str): Path to file or base path for multi-part files
            
        Returns:
            dict: Combined dictionary data
        """
        # Check if this is a multi-part file
        if '_part' in file_path:
            base_path = file_path.split('_part')[0]
        else:
            base_path = file_path.replace('.pt', '')
        
        # Look for part files
        directory = os.path.dirname(base_path) or '.'
        base_name = os.path.basename(base_path)
        part_files = sorted([
            f for f in os.listdir(directory) 
            if f.startswith(base_name) and '_part' in f and f.endswith('.pt')
        ])
        
        if not part_files:
            # Single file
            print(f"Loading single file: {file_path}")
            return torch.load(file_path)
        
        # Multi-part file
        print(f"Found {len(part_files)} parts. Loading and combining...")
        combined_data = {}
        
        for part_file in tqdm(part_files, desc="Loading parts"):
            part_path = os.path.join(directory, part_file)
            part_data = torch.load(part_path)
            combined_data.update(part_data)
        
        print(f"Successfully loaded {len(combined_data)} total states from {len(part_files)} parts")
        return combined_data

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
        if selected_adapter[0].split('_')[0] == 'LLM':
            # i.e. if the model name is 'llama3.2', we want to shorten it to 'llama32'
            if len({local_config['model_name'].split('.')}) > 2:
                mod_name_short = {local_config['model_name'].split('.')[0]}+{local_config['model_name'].split('.')[1]}
            else:
                mod_name_short = local_config['model_name'].split('.')[0]
            file_name = f"observed_states_{selected_application}_{selected_config}_{selected_adapter[0]}_{mod_name_short}_{num_explor_episodes}_{time}.pt"
        else:
            file_name = f"observed_states_{selected_application}_{selected_config}_{selected_adapter[0]}_{num_explor_episodes}_{time}.pt"
        file_path = os.path.join(save_dir, file_name)

        # Save observed states only (with automatic splitting if too large)
        saved_files = self._save_large_file(observed_states, file_path)
        
        # Return the base file name or list of part names
        if len(saved_files) == 1:
            return observed_states, saved_files[0]
        else:
            return observed_states, saved_files
            


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

        if selected_adapter[0].split('_')[0] == 'LLM':
            ollama_models = ollama.list()
            print("Select an ollama model from the following options:")
            for i, model in enumerate(ollama_models['models']):
                print(f"{i + 1}: {model['model']}")
            select_ollama_model_index = int(input("Enter the number of the ollama model you want to select: ")) - 1
            select_ollama_model = ollama_models['models'][select_ollama_model_index]['model']
            local_config['model_name'] = select_ollama_model
            print(f"-- Selected ollama model: {select_ollama_model}")
        
        num_explor_episodes = int(input("Enter the number of the exploration episodes: "))


        print("--------------------------------")
        print("-Selected options-")
        print(f"-- Selected application: {selected_application}")
        print(f"-- Selected configuration: {config_input}")
        print(f"-- Selected adapter: {selected_adapter[0]}")
        if selected_adapter[0].split('_')[0] == 'LLM':
            print(f"-- Selected ollama model: {select_ollama_model}")
        print(f"-- Number of exploration episodes: {num_explor_episodes}")
        print("--------------------------------")

        self.observed_states, self.observed_states_file_name = self.get_observed_states(engine, 
                                selected_application, config_input, selected_adapter,
                                local_config, 
                                adapters, 
                                num_explor_episodes)
        
    def adapter_apply(self):
        # Get unique base names (exclude part files from display)
        all_files = [file for file in os.listdir('./') if file.endswith('.pt')]
        file_names = []
        seen_bases = set()
        
        for file in all_files:
            if '_part' in file:
                base_name = file.split('_part')[0] + '.pt'
                if base_name not in seen_bases:
                    file_names.append(base_name + ' (multi-part)')
                    seen_bases.add(base_name)
            elif file not in seen_bases:
                file_names.append(file)
                seen_bases.add(file)
        
        for n, file in enumerate(file_names):
            print(f"- {n}: {file}")
        selection = input("\n Select the file to transform with the adapter (by number): ")
        selected_display_name = file_names[int(selection)]
        
        # Get actual file path
        if '(multi-part)' in selected_display_name:
            selected_file = selected_display_name.replace(' (multi-part)', '').replace('.pt', '_part1.pt')
        else:
            selected_file = selected_display_name

        # Get application name from file name
        selected_application = selected_file.split('_')[2]
        if selected_application not in self.possible_applications:
            # Allow terminal input to select application
            print("Application name not found, please confirm the application name from the following options:")
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
        selected_adapter = list(adapter_list.keys())[selected_adapter_index]

        # Get data for the selected application
        engine = self.pull_app_data[selected_application]['engine']
        local_config = self.pull_app_data[selected_application]['local_configs'][config_input]
        adapters = self.pull_app_data[selected_application]['adapters']

        if selected_adapter.split('_')[0] == 'LLM':
            ollama_models = ollama.list()
            print("Select an ollama model from the following options:")
            for i, model in enumerate(ollama_models['models']):
                print(f"{i + 1}: {model['model']}")
            select_ollama_model_index = int(input("Enter the number of the ollama model you want to select: ")) - 1
            select_ollama_model = ollama_models['models'][select_ollama_model_index]['model']
            local_config['model_name'] = select_ollama_model
            print(f"-- Selected ollama model: {select_ollama_model}")

        observed_data_in = self._load_large_file(selected_file)
        number_episodes = len(observed_data_in)
        # Save the transformed data
        time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        print("--------------------------------")
        print("-Selected options-")
        print(f"-- Selected application: {selected_application}")
        print(f"-- Selected adapter: {selected_adapter}")
        if selected_adapter.split('_')[0] == 'LLM':
            print(f"-- Selected ollama model: {select_ollama_model}")
            file_name = 'observed_states_'+selected_application+'_'+config_input+'_'+selected_adapter+'_'+select_ollama_model+'_'+str(number_episodes)+'_'+time+'.pt'
        else:
            file_name = 'observed_states_'+selected_application+'_'+config_input+'_'+selected_adapter+'_'+str(number_episodes)+'_'+time+'.pt'
        print("--------------------------------")

        ENGINE_APPLY = engine(local_setup_info=local_config)
        ADAPTER_APPLY = adapters[selected_adapter](setup_info=local_config)
        print(f"Using engine: {ENGINE_APPLY} to generate legal moves.")
        print(f"Using adapter: {ADAPTER_APPLY}")

        print(f"\n Transforming {len(observed_data_in)} observed states from {selected_file}...\n")
        observed_data_out = {}
        for key in tqdm(observed_data_in.keys(), desc="Transforming observed states with adapter", total=len(observed_data_in)):
            legal_moves_sampled = ENGINE_APPLY.legal_move_generator(obs=key)
            observed_data_out[key] = ADAPTER_APPLY.adapter(state=key, legal_moves=legal_moves_sampled, encode=False)

        self._save_large_file(observed_data_out, file_name)


    def encode(self, observed_states:dict|str=None,
                        directory_search:bool=False,
                        save_dir:str=None,
                        encoder:str ='MiniLM_L6v2') -> Tensor:
        """    Encodes the observed states using a language encoder.
        Args:
            observed_states (dict or str): The observed states to encode, can be the dictionary or the directory path string.
            directory_search (bool): Whether to search for the observed states in a directory. Defaults to False.
            save_dir (str): The directory where the encoded states will be saved. If None, defaults to './encoded-prerender-data'.
            encoder (str): The name of the encoder to use. Defaults to 'MiniLM_L6v2', options include:
                - 'MiniLM_L6v2': A lightweight language model suitable for encoding text.
                - ~~Other encoders can be added in the future.~~
        Returns:
            Tensor: The encoded representation of the observed states.
        """
        # ------------------------------------------------------------------
        # Define the available encoders
        # Currently only MiniLM_L6v2 is available, but can be extended in the future.
        ENCODERS = {'MiniLM_L6v2': MiniLM_L6v2}
        encoder = ENCODERS[encoder]()
        # ------------------------------------------------------------------
        if observed_states is None:
            if (self.observed_states is not None) and (not directory_search):
                observed_states = self.observed_states
                save_dir  = './prerender-data/'
                observed_states_filename = self.observed_states_file_name
            else:
                print("\n ----------------------------------------------------")
                print(" No observed states provided. Please select a file to encode.")
                print(" ----------------------------------------------------\n")
                # Get unique base names (exclude part files from display)
                all_files = [file for file in os.listdir('./') if file.endswith('.pt')]
                file_names = []
                seen_bases = set()
                
                for file in all_files:
                    if '_part' in file:
                        base_name = file.split('_part')[0] + '.pt'
                        if base_name not in seen_bases:
                            file_names.append(base_name + ' (multi-part)')
                            seen_bases.add(base_name)
                    elif file not in seen_bases:
                        file_names.append(file)
                        seen_bases.add(file)
                
                for n, file in enumerate(file_names):
                    print(f"- {n}: {file}")
                selection = input("\n Select the file to encode (by number): ")
                selected_display_name = file_names[int(selection)]
                
                # Get actual file path
                if '(multi-part)' in selected_display_name:
                    observed_states_filename = selected_display_name.replace(' (multi-part)', '')
                    observed_states_path = os.path.join('./', observed_states_filename.replace('.pt', '_part1.pt'))
                else:
                    observed_states_filename = selected_display_name
                    observed_states_path = os.path.join('./', observed_states_filename)
                
                observed_states = self._load_large_file(observed_states_path)
                save_dir = './'
        else:
            if isinstance(observed_states, str):
                observed_states_filename = observed_states.split('/')[-1].split('.')[0]
                if not save_dir:
                    save_dir = os.path.dirname(observed_states)
                observed_states = self._load_large_file(observed_states)
            else:
                observed_states_filename = 'observed_states'
                if not save_dir:
                    save_dir = './'

        # Encode the observed states
        print(f"\n Encoding observed state file {observed_states_filename} using {encoder.name}...")
        str_states = [str_state for str_state in observed_states.values()]
        observed_states_encoded = encoder.encode(str_states)


        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, 'encoded_' + observed_states_filename.split('.')[0] + '.pt')
        
        # Save as .pt file for better compression and storage efficiency
        torch.save(observed_states_encoded, file_path)
        
        # Get file size for reporting
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        print(f"Encoded states saved to {file_path}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Number of States: {len(observed_states_encoded)}")

        return observed_states_encoded