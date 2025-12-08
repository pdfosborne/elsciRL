from datetime import datetime
import os
import torch
import urllib.request
import json 
import numpy as np
import httpimport
import subprocess
import sys
import pickle
import hashlib
import shutil
import importlib.util
from tqdm import tqdm


# Local imports
from elsciRL.application_suite.import_data import Applications
from elsciRL.application_suite.experiment_agent import DefaultAgentConfig

class PullApplications:
    """Simple applications class to run a setup tests of experiments.
        - Problem selection: problems to run in format ['problem1', 'problem2',...]

    Applications:
        - Sailing: {'easy'},
        - Classroom: {'classroom_A'}
    """
    # [x]: Make it so it pulls all possible configs and adapters from the repo
    # [x] Allow a blank entry for repo for experimental testing to pull most recent commit by default
    # [x]: Auto install libraries from application repo requirements.txt
    # [x]: Improve process for adding local application, cache to local directory to check from
    def __init__(self) -> None:
        imports = Applications()
        self.imports = imports.data
        self.current_test = {}
        
        # Cache directory structure
        self.cache_dir = os.path.join(os.getcwd(), '.cache')
        self.log_file = os.path.join(self.cache_dir, 'import_log.json')
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Load existing log or create new one
        self._load_import_log()
        
    def _load_large_pt_file(self, file_path):
        """
        Load data from .pt file(s), combining parts if necessary.
        
        Args:
            file_path (str): Path to file or base path for multi-part files
            
        Returns:
            dict or other: Combined data
        """
        # Check if this is a multi-part file
        if '_part' in file_path:
            base_path = file_path.split('_part')[0]
        else:
            base_path = file_path.replace('.pt', '')
        
        # Look for part files
        directory = os.path.dirname(base_path) or '.'
        base_name = os.path.basename(base_path)
        
        # Get all files in the directory
        if os.path.exists(directory):
            all_files = os.listdir(directory)
            part_files = sorted([
                f for f in all_files
                if f.startswith(base_name) and '_part' in f and f.endswith('.pt')
            ])
        else:
            part_files = []
        
        if not part_files:
            # Single file
            if os.path.exists(file_path):
                print(f"Loading single .pt file: {os.path.basename(file_path)}")
                return torch.load(file_path)
            else:
                return None
        
        # Multi-part file
        print(f"Found {len(part_files)} parts. Loading and combining...")
        combined_data = {}
        
        for part_file in tqdm(part_files, desc="Loading parts"):
            part_path = os.path.join(directory, part_file)
            part_data = torch.load(part_path)
            if isinstance(part_data, dict):
                combined_data.update(part_data)
            else:
                # If not a dict, return the first part (might be a tensor or other type)
                return part_data
        
        print(f"Successfully loaded {len(combined_data)} total items from {len(part_files)} parts")
        return combined_data
    
    def _get_cache_dir(self, problem):
        """Get the cache directory for a specific problem."""
        return os.path.join(self.cache_dir, problem)
    
    def _get_cache_metadata_file(self, problem):
        """Get the metadata file path for a problem."""
        return os.path.join(self._get_cache_dir(problem), 'cache_metadata.json')
    
    def _load_import_log(self):
        """Load existing import log or create new one."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.import_log = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Failed to load import log: {e}")
                self.import_log = {}
        else:
            self.import_log = {}
    
    def _save_import_log(self):
        """Save import log to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.import_log, f, indent=2, default=str)
    
    def _generate_cache_key(self, problem, commit_id, source_data):
        """Generate a unique cache key based on problem, commit_id, and source data."""
        # Create a hash of the source data to detect changes
        source_hash = hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest()
        return f"{problem}_{commit_id}_{source_hash}"
    
    def _save_to_cache(self, problem, data):
        """Save imported data to cache directory structure."""
        try:
            cache_dir = self._get_cache_dir(problem)
            metadata_file = self._get_cache_metadata_file(problem)
            
            # Create cache directory if it doesn't exist
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            # Save metadata
            metadata = data.get('cache_metadata', {})
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save engine file
            if 'engine' in data:
                engine_dir = os.path.join(cache_dir, 'engine')
                if not os.path.exists(engine_dir):
                    os.makedirs(engine_dir)
                # Save engine file
                engine_file = os.path.join(engine_dir, f"{data['engine_filename']}")
                engine_filename = self.imports[problem]['engine_filename']
                engine_path = os.path.join(self.root, self.imports[problem]['engine_folder'], engine_filename)
                
                if os.path.exists(engine_path):
                    with open(engine_path, 'r') as f:
                        engine_content = f.read()
                    with open(engine_file, 'w') as f:
                        f.write(engine_content)
                    print(f"Saved engine file: {data['engine_filename']}")
            
            # Save adapters
            if 'adapters' in data:
                adapters_dir = os.path.join(cache_dir, 'adapters')
                if not os.path.exists(adapters_dir):
                    os.makedirs(adapters_dir)
                # Save adapter files
                for adapter_name, adapter_filename in self.imports[problem]['adapter_filenames'].items():
                    adapter_file = os.path.join(adapters_dir, f"{adapter_name}.py")
                    adapter_path = os.path.join(self.root, self.imports[problem]['local_adapter_folder'], adapter_filename)
                    
                    if os.path.exists(adapter_path):
                        with open(adapter_path, 'r') as f:
                            adapter_content = f.read()
                        with open(adapter_file, 'w') as f:
                            f.write(adapter_content)
                        print(f"Saved adapter file: {adapter_filename}")
            
            # Save experiment and local configs in unified configs directory
            configs_dir = os.path.join(cache_dir, 'configs')
            if not os.path.exists(configs_dir):
                os.makedirs(configs_dir)
            
            # Save experiment configs
            if 'experiment_configs' in data:
                for config_name, config_data in data['experiment_configs'].items():
                    # Remove extension from config_name if present
                    clean_name = config_name
                    if config_name.endswith('.json'):
                        clean_name = config_name[:-5]  # Remove .json extension
                    config_file = os.path.join(configs_dir, f"experiment_{clean_name}.json")
                    with open(config_file, 'w') as f:
                        json.dump(config_data, f, indent=2)
            
            # Save local configs
            if 'local_configs' in data:
                for config_name, config_data in data['local_configs'].items():
                    # Remove extension from config_name if present
                    clean_name = config_name
                    if config_name.endswith('.json'):
                        clean_name = config_name[:-5]  # Remove .json extension
                    config_file = os.path.join(configs_dir, f"local_{clean_name}.json")
                    with open(config_file, 'w') as f:
                        json.dump(config_data, f, indent=2)
            
            # Save prerender data (all types in single prerender directory)
            prerender_dir = os.path.join(cache_dir, 'prerender')
            if not os.path.exists(prerender_dir):
                os.makedirs(prerender_dir)
            
            # Save prerender data
            if 'prerender_data' in data:
                for data_name, data_content in data['prerender_data'].items():
                    # Remove extension from data_name if present
                    clean_name = data_name
                    for ext in ['.txt', '.json', '.jsonl']:
                        if data_name.endswith(ext):
                            clean_name = data_name[:-len(ext)]
                            break
                    
                    # Determine file extension based on data type
                    if isinstance(data_content, (list, dict)):
                        data_file = os.path.join(prerender_dir, f"{clean_name}.json")
                        with open(data_file, 'w') as f:
                            json.dump(data_content, f, indent=2)
                    else:
                        data_file = os.path.join(prerender_dir, f"{clean_name}.txt")
                        with open(data_file, 'w') as f:
                            f.write(str(data_content))
            
            # Save prerender data encoded (as numpy arrays, txt, or json)
            if 'prerender_data_encoded' in data:
                for data_name, data_content in data['prerender_data_encoded'].items():
                    # Get the actual filename from import data
                    data_filename = self.imports[problem]['prerender_data_encoded_filenames'].get(data_name, f"encoded_{data_name}")
                    
                    # Remove extension from data_filename if present
                    clean_filename = data_filename
                    for ext in ['.npy', '.txt', '.json', '.jsonl']:
                        if data_filename.endswith(ext):
                            clean_filename = data_filename[:-len(ext)]
                            break
                    
                    if isinstance(data_content, torch.Tensor):
                        # Save as .npy for tensor data
                        data_file = os.path.join(prerender_dir, f"{clean_filename}.npy")
                        np.save(data_file, data_content.cpu().numpy())
                    elif isinstance(data_content, (list, dict)):
                        # Save as .json for structured data
                        data_file = os.path.join(prerender_dir, f"{clean_filename}.json")
                        with open(data_file, 'w') as f:
                            json.dump(data_content, f, indent=2)
                    else:
                        # Save as .txt for other data types
                        data_file = os.path.join(prerender_dir, f"{clean_filename}.txt")
                        with open(data_file, 'w') as f:
                            f.write(str(data_content))
            
            # Save prerender images
            if 'prerender_images' in data:
                for image_name, image_data in data['prerender_images'].items():
                    # Get the actual filename from import data
                    image_filename = self.imports[problem]['prerender_image_filenames'].get(image_name, f"{image_name}.png")
                    
                    # Use the filename directly as it already includes the extension
                    image_file = os.path.join(prerender_dir, image_filename)
                    with open(image_file, 'wb') as f:
                        f.write(image_data)
            
            # Save instructions
            if 'instructions' in data:
                instructions_dir = os.path.join(cache_dir, 'instructions')
                if not os.path.exists(instructions_dir):
                    os.makedirs(instructions_dir)
                for instruction_name, instruction_data in data['instructions'].items():
                    # Remove extension from instruction_name if present
                    clean_name = instruction_name
                    if instruction_name.endswith('.json'):
                        clean_name = instruction_name[:-5]  # Remove .json extension
                    instruction_file = os.path.join(instructions_dir, f"{clean_name}.json")
                    with open(instruction_file, 'w') as f:
                        json.dump(instruction_data, f, indent=2)
            
            # Save analysis files
            if 'local_analysis' in data:
                analysis_dir = os.path.join(cache_dir, 'analysis')
                if not os.path.exists(analysis_dir):
                    os.makedirs(analysis_dir)
                for analysis_name, analysis_data in data['local_analysis'].items():
                    # Save analysis .py files
                    analysis_filename = self.imports[problem]['local_analysis_filenames'].get(analysis_name, f"{analysis_name}.py")
                    analysis_path = os.path.join(self.root, self.imports[problem]['local_analysis_folder'], analysis_filename)
                    
                    if os.path.exists(analysis_path):
                        analysis_file = os.path.join(analysis_dir, f"{analysis_name}.py")
                        with open(analysis_path, 'r') as f:
                            analysis_content = f.read()
                        with open(analysis_file, 'w') as f:
                            f.write(analysis_content)
                        print(f"Saved analysis file: {analysis_filename}")
            
            # Save README files
            if 'readme_files' in data:
                readme_dir = os.path.join(cache_dir, 'readme_files')
                if not os.path.exists(readme_dir):
                    os.makedirs(readme_dir)
                for readme_name, readme_content in data['readme_files'].items():
                    # Determine file extension from readme name or use default
                    if '.' in readme_name:
                        readme_file = os.path.join(readme_dir, readme_name)
                    else:
                        readme_file = os.path.join(readme_dir, f"{readme_name}.md")
                    with open(readme_file, 'w', encoding='utf-8') as f:
                        f.write(readme_content)
                    print(f"Saved README file: {readme_name}")
            
            print(f"Cached data for {problem} in directory structure")
        except Exception as e:
            print(f"Failed to save cache for {problem}: {e}")
    
    def _load_from_cache(self, problem, commit_id, source_data):
        """Load data from cache directory structure if available and up-to-date using git checking."""
        try:
            cache_dir = self._get_cache_dir(problem)
            metadata_file = self._get_cache_metadata_file(problem)
            
            if not os.path.exists(cache_dir) or not os.path.exists(metadata_file):
                return None
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse cache metadata for {problem}: {e}")
                    return None
            
            # Check if cache is valid by comparing commit_id and source data
            cached_commit = metadata.get('commit_id')
            cached_source_hash = metadata.get('source_hash')
            current_source_hash = hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest()
            
            # For 'main' branch, use git to check for updates
            if commit_id == 'main':
                # Clone repository if it doesn't exist
                github_user = self.imports[problem]['github_user']
                repository = self.imports[problem]['repository']
                if not self._clone_repository(problem, github_user, repository, commit_id):
                    return None
                
                # Check for updates using git fetch
                has_updates, current_commit, remote_commit = self._check_repository_updates(problem, commit_id)
                
                if has_updates:
                    print(f"Repository has updates for {problem}, will pull fresh data")
                    return None
                else:
                    print(f"Repository is up to date for {problem}, using cached data")
                    # Use cached import data if available, otherwise use current source_data
                    cached_import_data = metadata.get('import_data', source_data)
                    return self._load_cached_data(problem, cache_dir, cached_import_data)
            
            # For specific commit IDs, check if we have the exact commit
            if cached_commit == commit_id and cached_source_hash == current_source_hash:
                print(f"Using cached data for {problem} (commit: {commit_id})")
                # Use cached import data if available, otherwise use current source_data
                cached_import_data = metadata.get('import_data', source_data)
                return self._load_cached_data(problem, cache_dir, cached_import_data)
            
            return None
        except Exception as e:
            print(f"Failed to load cache for {problem}: {e}")
            return None
    
    def _load_cached_data(self, problem, cache_dir, import_data=None):
        """Load cached data from directory structure using import data for file names."""
        try:
            data = {'cache_metadata': {}}
            
            # Load metadata
            metadata_file = self._get_cache_metadata_file(problem)
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    try:
                        data['cache_metadata'] = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse cache metadata for {problem}: {e}")
                        data['cache_metadata'] = {}
            
            # Use cached import data if available, otherwise use current import_data
            if import_data is None:
                import_data = data['cache_metadata'].get('import_data', {})
        
            # Reconstruct source data structure from cached import data
            cache_root_path = cache_dir
            data['source'] = {str(cache_root_path): import_data}

            # Load engine
            engine_dir = os.path.join(cache_dir, import_data.get('engine_folder', self.imports[problem]['engine_folder']))
            print(">>>> Engine dir: ", engine_dir)
            if os.path.exists(engine_dir):
                try:
                    # Get the engine filename from the import data
                    engine_filename = import_data.get('engine_filename', self.imports[problem]['engine_filename'])
                    engine_file_path = os.path.join(engine_dir, engine_filename)
                    
                    # Load module directly from file path
                    engine_module_name = engine_filename.split('.')[0]
                    spec = importlib.util.spec_from_file_location(engine_module_name, engine_file_path)
                    engine_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(engine_module)

                    data['engine'] = engine_module.Engine
                except Exception as e:
                    print(f"Failed to load cached engine: {e}")
            
            # Load adapters
            adapters_dir = os.path.join(cache_dir, 'adapters')
            if os.path.exists(adapters_dir):
                data['adapters'] = {}
                adapter_files = [f for f in os.listdir(adapters_dir) if f.endswith('.py')]
                if adapter_files:
                    try:
                        # Get adapter filenames from import data
                        adapter_filenames = import_data.get('adapter_filenames', self.imports[problem]['adapter_filenames'])
                        for adapter_name, adapter_filename in adapter_filenames.items():
                            adapter_file_path = os.path.join(adapters_dir, adapter_filename)
                            
                            # Load module directly from file path
                            adapter_module_name = adapter_filename.split('.')[0]
                            spec = importlib.util.spec_from_file_location(adapter_module_name, adapter_file_path)
                            adapter_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(adapter_module)
                            
                            data['adapters'][adapter_name] = adapter_module.Adapter
                            print(f"Loaded cached adapter: {adapter_filename}")
                    except Exception as e:
                        print(f"Failed to load cached adapters: {e}")
            
            # Load experiment and local configs from unified configs directory using import data
            configs_dir = os.path.join(cache_dir, 'configs')
            if os.path.exists(configs_dir):
                data['experiment_configs'] = {}
                data['local_configs'] = {}
                
                # Load experiment configs using filenames from import data
                experiment_config_filenames = import_data.get('experiment_config_filenames', {})
                for config_name, config_filename in experiment_config_filenames.items():
                    # Remove extension from config_filename if present
                    clean_filename = config_filename
                    if config_filename.endswith('.json'):
                        clean_filename = config_filename[:-5]  # Remove .json extension
                    config_file = os.path.join(configs_dir, f"{clean_filename}.json")
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            data['experiment_configs'][config_name] = json.load(f)
                            print(f"Loaded cached experiment config: {config_name}")
                
                # Load local configs using filenames from import data
                local_config_filenames = import_data.get('local_config_filenames', {})
                for config_name, config_filename in local_config_filenames.items():
                    # Remove extension from config_filename if present
                    clean_filename = config_filename
                    if config_filename.endswith('.json'):
                        clean_filename = config_filename[:-5]  # Remove .json extension
                    config_file = os.path.join(configs_dir, f"{clean_filename}.json")
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            data['local_configs'][config_name] = json.load(f)
                            print(f"Loaded cached local config: {config_name}")

            # Load prerender data (all types from single prerender directory) using import data
            prerender_dir = os.path.join(cache_dir, 'prerender')
            if os.path.exists(prerender_dir):
                data['prerender_data'] = {}
                data['prerender_data_encoded'] = {}
                data['prerender_images'] = {}
                
                # Load regular prerender data using filenames from import data
                prerender_data_filenames = import_data.get('prerender_data_filenames', {})
                for data_name, data_filename in prerender_data_filenames.items():
                    # Remove extension from data_filename if present
                    clean_filename = data_filename
                    for ext in ['.pt', '.txt', '.json', '.jsonl']:
                        if data_filename.endswith(ext):
                            clean_filename = data_filename[:-len(ext)]
                            break
                    
                    # Try different extensions (prioritize .pt)
                    for ext in ['.pt', '.txt', '.json', '.jsonl']:
                        file_path = os.path.join(prerender_dir, f"{clean_filename}{ext}")
                        
                        # For .pt files, check for multi-part files
                        if ext == '.pt':
                            # Check if it's a multi-part file
                            part1_path = os.path.join(prerender_dir, f"{clean_filename}_part1.pt")
                            if os.path.exists(part1_path):
                                file_path = part1_path
                            
                            if os.path.exists(file_path) or os.path.exists(part1_path):
                                loaded_data = self._load_large_pt_file(file_path)
                                if loaded_data is not None:
                                    data['prerender_data'][data_name] = loaded_data
                                    print(f"Loaded cached prerender data: {data_name}")
                                    break
                        elif os.path.exists(file_path):
                            if ext == '.json':
                                with open(file_path, 'r') as f:
                                    data['prerender_data'][data_name] = json.load(f)
                            elif ext == '.jsonl':
                                # Handle JSONL files
                                jsonl_data = {}
                                with open(file_path, 'r') as f:
                                    for line in f:
                                        if line.strip():
                                            row = json.loads(line)
                                            jsonl_data.update(row)
                                data['prerender_data'][data_name] = jsonl_data
                            else:  # .txt files
                                with open(file_path, 'r') as f:
                                    content = f.read()
                                    try:
                                        data['prerender_data'][data_name] = json.loads(content)
                                    except json.JSONDecodeError:
                                        # If not valid JSON, treat as plain text
                                        data['prerender_data'][data_name] = content
                            print(f"Loaded cached prerender data: {data_name}")
                            break
                
                # Load encoded prerender data using filenames from import data
                prerender_data_encoded_filenames = import_data.get('prerender_data_encoded_filenames', {})
                for data_name, data_filename in prerender_data_encoded_filenames.items():
                    # The data_filename already includes "encoded_" prefix, so use it directly
                    # Remove extension from data_filename if present
                    clean_filename = data_filename
                    for ext in ['.pt', '.npy', '.txt', '.json', '.jsonl']:
                        if data_filename.endswith(ext):
                            clean_filename = data_filename[:-len(ext)]
                            break
                    
                    # Try different extensions (prioritize .pt and .npy)
                    for ext in ['.pt', '.npy', '.txt', '.json']:
                        file_path = os.path.join(prerender_dir, f"{clean_filename}{ext}")
                        
                        # For .pt files, check for multi-part files
                        if ext == '.pt':
                            # Check if it's a multi-part file
                            part1_path = os.path.join(prerender_dir, f"{clean_filename}_part1.pt")
                            if os.path.exists(part1_path):
                                file_path = part1_path
                            
                            if os.path.exists(file_path) or os.path.exists(part1_path):
                                loaded_data = self._load_large_pt_file(file_path)
                                if loaded_data is not None:
                                    data['prerender_data_encoded'][data_name] = loaded_data
                                    print(f"Loaded cached encoded prerender data: {data_name}")
                                    break
                        elif os.path.exists(file_path):
                            if ext == '.npy':
                                array_data = np.load(file_path)
                                data['prerender_data_encoded'][data_name] = torch.from_numpy(array_data)
                            elif ext == '.json':
                                with open(file_path, 'r') as f:
                                    json_data = json.load(f)
                                data['prerender_data_encoded'][data_name] = json_data
                            else:  # .txt files
                                with open(file_path, 'r') as f:
                                    content = f.read()
                                    try:
                                        text_data = json.loads(content)
                                    except json.JSONDecodeError:
                                        # If not valid JSON, treat as plain text
                                        text_data = content
                                data['prerender_data_encoded'][data_name] = text_data
                            print(f"Loaded cached encoded prerender data: {data_name}")
                            break
                
                # Load prerender images using filenames from import data
                prerender_image_filenames = import_data.get('prerender_image_filenames', {})
                for image_name, image_filename in prerender_image_filenames.items():
                    # Use the image filename directly as it already includes the extension
                    file_path = os.path.join(prerender_dir, image_filename)
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            data['prerender_images'][image_name] = f.read()
                        print(f"Loaded cached prerender image: {image_name}")
            
            # Load instructions using filenames from import data
            instructions_dir = os.path.join(cache_dir, 'instructions')
            if os.path.exists(instructions_dir):
                data['instructions'] = {}
                instruction_filenames = import_data.get('instruction_filenames', {})
                for instruction_name, instruction_filename in instruction_filenames.items():
                    # Remove extension from instruction_filename if present
                    clean_filename = instruction_filename
                    if instruction_filename.endswith('.json'):
                        clean_filename = instruction_filename[:-5]  # Remove .json extension
                    instruction_file = os.path.join(instructions_dir, f"{clean_filename}.json")
                    if os.path.exists(instruction_file):
                        with open(instruction_file, 'r') as f:
                            data['instructions'][instruction_name] = json.load(f)
                            print(f"Loaded cached instruction: {instruction_name}")
            
            # Load analysis files using filenames from import data
            analysis_dir = os.path.join(cache_dir, 'analysis')
            if os.path.exists(analysis_dir):
                data['local_analysis'] = {}
                analysis_filenames = import_data.get('local_analysis_filenames', {})
                for analysis_name, analysis_filename in analysis_filenames.items():
                    # Remove extension from analysis_filename if present
                    clean_filename = analysis_filename
                    if analysis_filename.endswith('.py'):
                        clean_filename = analysis_filename[:-3]  # Remove .py extension
                    analysis_file = os.path.join(analysis_dir, f"{clean_filename}.py")
                    if os.path.exists(analysis_file):
                        # Add analysis directory to Python path temporarily
                        import sys
                        sys.path.insert(0, analysis_dir)
                        
                        try:
                            analysis_module = __import__(clean_filename)
                            data['local_analysis'][analysis_name] = analysis_module.Analysis
                            print(f"Loaded cached analysis: {analysis_name}")
                        except Exception as e:
                            print(f"Failed to load cached analysis {analysis_name}: {e}")
                            data['local_analysis'][analysis_name] = {}
                        finally:
                            # Remove from path
                            sys.path.pop(0)
            # Load README files
            readme_dir = os.path.join(cache_dir, 'readme_files')
            if os.path.exists(readme_dir):
                data['readme_files'] = {}
                for readme_file in os.listdir(readme_dir):
                    if readme_file.endswith(('.md', '.txt', '.rst')):
                        readme_name = readme_file
                        with open(os.path.join(readme_dir, readme_file), 'r', encoding='utf-8') as f:
                            data['readme_files'][readme_name] = f.read()
                        print(f"Loaded cached README file: {readme_name}")
            
            return data
        except Exception as e:
            print(f"Failed to load cached data for {problem}: {e}")
            return None
    
    def _log_import(self, problem, commit_id, source_data, cache_hit=False):
        """Log import activity."""
        timestamp = datetime.now().isoformat()
        source_hash = hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest()
        
        if problem not in self.import_log:
            self.import_log[problem] = []
        
        log_entry = {
            'timestamp': timestamp,
            'commit_id': commit_id,
            'source_hash': source_hash,
            'cache_hit': cache_hit,
            'source_data': source_data
        }
        
        self.import_log[problem].append(log_entry)
        self._save_import_log()
        
        print(f"Logged import for {problem} (commit: {commit_id}, cache_hit: {cache_hit})")
    
    def get_latest_import_info(self, problem):
        """Get information about the most recent import for a problem."""
        if problem in self.import_log and self.import_log[problem]:
            latest = self.import_log[problem][-1]
            return {
                'timestamp': latest['timestamp'],
                'commit_id': latest['commit_id'],
                'cache_hit': latest['cache_hit']
            }
        return None
    

    
    def _extract_last_commit_id(self, readme_content):
        """Extract the last-commit-id value from README content."""
        import re
        if not readme_content:
            return None
        
        # Look for patterns like 'last-commit-id: xxxxxx' or 'last-commit-id:xxxxxx'
        patterns = [
            r'last-commit-id:\s*([a-f0-9]{7,40})',  # SHA hash
            r'last-commit-id:\s*([a-zA-Z0-9_-]+)',   # Any alphanumeric identifier
        ]
        
        for pattern in patterns:
            match = re.search(pattern, readme_content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _get_latest_commit_id(self, github_user, repository, branch='main'):
        """Get the latest commit ID from GitHub API."""
        try:
            import urllib.request
            import json
            
            # Use GitHub API to get the latest commit
            api_url = f"https://api.github.com/repos/{github_user}/{repository}/commits/{branch}"
            
            # Add timeout and user agent to avoid rate limiting
            request = urllib.request.Request(api_url)
            request.add_header('User-Agent', 'elsciRL-Application-Downloader')
            
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data['sha']
                
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"Repository {github_user}/{repository} not found or branch {branch} doesn't exist")
            else:
                print(f"HTTP error getting latest commit ID for {github_user}/{repository}: {e}")
            return 'main'
        except Exception as e:
            print(f"Error getting latest commit ID for {github_user}/{repository}: {e}")
            # Fallback to 'main' if API fails
            return 'main'
    
    def _get_git_repo_dir(self, problem):
        """Get the git repository directory for a specific problem."""
        return os.path.join(self.cache_dir, f"{problem}")
    
    def _clone_repository(self, problem, github_user, repository, commit_id='main'):
        """Clone the repository for a problem if it doesn't exist."""
        repo_dir = self._get_git_repo_dir(problem)
        
        if not os.path.exists(repo_dir):
            try:
                repo_url = f"https://github.com/{github_user}/{repository}.git"
                print(f"Cloning repository for {problem}: {repo_url}")
                subprocess.check_call(['git', 'clone', repo_url, repo_dir], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Successfully cloned repository for {problem}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone repository for {problem}: {e}")
                return False
        else:
            print(f"Repository already exists for {problem}")
        
        return True
    
    def _check_repository_updates(self, problem, commit_id='main'):
        """Check if the repository has updates using git fetch."""
        repo_dir = self._get_git_repo_dir(problem)
        
        if not os.path.exists(repo_dir):
            return False
        
        try:
            # Fetch latest changes from remote
            subprocess.check_call(['git', 'fetch', 'origin'], 
                                cwd=repo_dir, 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Get current commit hash
            current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                   cwd=repo_dir).decode('utf-8').strip()
            
            # Get remote commit hash for the specified branch/commit
            if commit_id == 'main':
                remote_commit = subprocess.check_output(['git', 'rev-parse', 'origin/main'], 
                                                      cwd=repo_dir).decode('utf-8').strip()
            else:
                remote_commit = subprocess.check_output(['git', 'rev-parse', commit_id], 
                                                      cwd=repo_dir).decode('utf-8').strip()
            
            # Check if there are updates
            has_updates = current_commit != remote_commit
            
            if has_updates:
                print(f"Repository updates available for {problem}: {current_commit[:7]} -> {remote_commit[:7]}")
            else:
                print(f"Repository is up to date for {problem}: {current_commit[:7]}")
            
            return has_updates, current_commit, remote_commit
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to check repository updates for {problem}: {e}")
            return False, None, None
    
    def _update_repository(self, problem, commit_id='main'):
        """Update the repository using git pull."""
        repo_dir = self._get_git_repo_dir(problem)
        
        if not os.path.exists(repo_dir):
            return False
        
        try:
            # Checkout the specified commit/branch
            if commit_id == 'main':
                subprocess.check_call(['git', 'checkout', 'main'], 
                                    cwd=repo_dir, 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.check_call(['git', 'pull', 'origin', 'main'], 
                                    cwd=repo_dir, 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.check_call(['git', 'checkout', commit_id], 
                                    cwd=repo_dir, 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Get the new commit hash
            new_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                               cwd=repo_dir).decode('utf-8').strip()
            
            print(f"Successfully updated repository for {problem} to commit: {new_commit[:7]}")
            return True, new_commit
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to update repository for {problem}: {e}")
            return False, None
    

    
    def _pull_fresh_data(self, problem, commit_id, source_data):
        """Pull fresh data for a problem and cache it using git."""
        try:
            print(f"Pulling fresh data for {problem}...")
            
            # Clone or update repository
            github_user = self.imports[problem]['github_user']
            repository = self.imports[problem]['repository']
            
            if not self._clone_repository(problem, github_user, repository, commit_id):
                return None
            
            # Update repository if needed
            if commit_id == 'main':
                has_updates, current_commit, remote_commit = self._check_repository_updates(problem, commit_id)
                if has_updates:
                    success, new_commit = self._update_repository(problem, commit_id)
                    if not success:
                        return None
                else:
                    print(f"Repository is already up to date for {problem}")
            else:
                # For specific commits, checkout the exact commit
                success, new_commit = self._update_repository(problem, commit_id)
                if not success:
                    return None
            
            # Set root to the local repository directory
            repo_dir = self._get_git_repo_dir(problem)
            self.root = repo_dir
            print("Source: ", self.root)
            
            # Initialize the problem data structure
            self.current_test[problem] = {}
            self.current_test[problem]['engine_filename'] = self.imports[problem]['engine_filename']
            self.current_test[problem]['source'] = {str(self.root): source_data}
            
            # Load engine
            engine = self.imports[problem]['engine_filename']
            engine_path = os.path.join(self.root, self.imports[problem]['engine_folder'], engine)
            
            # Add the engine directory to Python path temporarily
            engine_dir = os.path.join(self.root, self.imports[problem]['engine_folder'])
            sys.path.insert(0, engine_dir)
            
            try:
                engine_module = __import__(engine.split('.')[0])
                self.current_test[problem]['engine'] = engine_module.Engine
            except:
                print("Engine error, attempting to install requirements.")
                try:
                    requirements_path = os.path.join(self.root, 'requirements.txt')
                    if os.path.exists(requirements_path):
                        with open(requirements_path, 'r') as f:
                            requirements = f.read().split('\n')
                        for req in requirements:
                            if req.strip():
                                try:
                                    subprocess.check_call([sys.executable, "-m", "pip", "install", req.strip()])
                                    print(f"Successfully installed {req}")
                                except subprocess.CalledProcessError:
                                    print(f"Failed to install {req}")
                        self.current_test[problem]['engine'] = engine_module.Engine
                        print("Successfully loaded engine after installing requirements.")
                    else:
                        print("No requirements.txt found.")
                except Exception as e:
                    print(f"Failed to load engine: {e}")
            finally:
                # Remove from path
                sys.path.pop(0)

            # Load adapters
            self.current_test[problem]['adapters'] = {}
            adapter_dir = os.path.join(self.root, self.imports[problem]['local_adapter_folder'])
            sys.path.insert(0, adapter_dir)
            
            try:
                for adapter_name, adapter in self.imports[problem]['adapter_filenames'].items():
                    adapter_module = __import__(adapter.split('.')[0])
                    self.current_test[problem]['adapters'][adapter_name] = adapter_module.Adapter
            finally:
                # Remove from path
                sys.path.pop(0)
            
            # Load experiment configs
            self.current_test[problem]['experiment_configs'] = {}
            config_dir = os.path.join(self.root, self.imports[problem]['config_folder'])
            for config_name, config in self.imports[problem]['experiment_config_filenames'].items():
                config_path = os.path.join(config_dir, config)
                with open(config_path, 'r') as f:
                    experiment_config = json.load(f)
                self.current_test[problem]['experiment_configs'][config_name] = experiment_config
            
            # Load local configs
            self.current_test[problem]['local_configs'] = {}
            for config_name, config in self.imports[problem]['local_config_filenames'].items():
                config_path = os.path.join(config_dir, config)
                with open(config_path, 'r') as f:
                    local_config = json.load(f)
                self.current_test[problem]['local_configs'][config_name] = local_config
            
            # Load local analysis
            self.current_test[problem]['local_analysis'] = {}
            analysis_dir = os.path.join(self.root, self.imports[problem]['local_analysis_folder'])
            sys.path.insert(0, analysis_dir)
            
            try:
                for analysis_name, analysis in self.imports[problem]['local_analysis_filenames'].items():
                    try:
                        local_analysis = __import__(analysis)
                        self.current_test[problem]['local_analysis'][analysis_name] = local_analysis.Analysis
                    except:
                        print("No analysis file found.")
                        self.current_test[problem]['local_analysis'][analysis_name] = {}
            finally:
                # Remove from path
                sys.path.pop(0)
            
            # Load prerender data
            self.current_test[problem]['prerender_data'] = {}
            self.current_test[problem]['prerender_data_encoded'] = {}
            if self.imports[problem]['prerender_data_folder'] != '':
                print("Pulling prerender data...")
                prerender_dir = os.path.join(self.root, self.imports[problem]['prerender_data_folder'])
                try:
                    for prerender_name, prerender in self.imports[problem]['prerender_data_filenames'].items():
                        prerender_path = os.path.join(prerender_dir, prerender)
                        
                        # Check for .pt files (including multi-part)
                        if prerender.endswith('.pt'):
                            # Check for multi-part files
                            base_path = prerender_path.replace('.pt', '')
                            part1_path = f"{base_path}_part1.pt"
                            
                            if os.path.exists(prerender_path) or os.path.exists(part1_path):
                                if os.path.exists(part1_path):
                                    data = self._load_large_pt_file(part1_path)
                                else:
                                    data = self._load_large_pt_file(prerender_path)
                                print(f"Pulling prerender data for {prerender_name}...")
                                self.current_test[problem]['prerender_data'][prerender_name] = data
                        elif os.path.exists(prerender_path):
                            if prerender.endswith(('.txt', '.json', '.jsonl')):
                                if prerender.endswith('.jsonl'):
                                    data = {}
                                    with open(prerender_path, 'r') as f:
                                        for line in f:
                                            if line.strip():
                                                row = json.loads(line)
                                                data.update(row)
                                elif prerender.endswith('.json'):
                                    with open(prerender_path, 'r') as f:
                                        data = json.load(f)
                                elif prerender.endswith('.txt'):
                                    with open(prerender_path, 'r') as f:
                                        content = f.read()
                                        try:
                                            data = json.loads(content)
                                        except json.JSONDecodeError:
                                            # If not valid JSON, treat as plain text
                                            data = content
                                else:
                                    raise ValueError(f"Unsupported file format for prerender data: {prerender}")
                                print(f"Pulling prerender data for {prerender_name}...")
                                self.current_test[problem]['prerender_data'][prerender_name] = data
                except Exception as e:
                    print(f"No prerender data found: {e}")
                    self.current_test[problem]['prerender_data'] = {}
                
                try:
                    for prerender_name, prerender in self.imports[problem]['prerender_data_encoded_filenames'].items():
                        prerender_path = os.path.join(prerender_dir, prerender)
                        
                        # Check for .pt files (including multi-part)
                        if prerender.endswith('.pt'):
                            # Check for multi-part files
                            base_path = prerender_path.replace('.pt', '')
                            part1_path = f"{base_path}_part1.pt"
                            
                            if os.path.exists(prerender_path) or os.path.exists(part1_path):
                                if os.path.exists(part1_path):
                                    data = self._load_large_pt_file(part1_path)
                                else:
                                    data = self._load_large_pt_file(prerender_path)
                                print(f"Pulling prerender encoded data for {prerender_name}...")
                                self.current_test[problem]['prerender_data_encoded'][prerender_name] = data
                        elif os.path.exists(prerender_path):
                            if prerender.endswith(('.txt', '.json', '.jsonl', '.npy')):
                                if prerender.endswith('.npy'):
                                    # Direct numpy file - convert to tensor
                                    map_location = 'cpu' if torch.cuda.is_available() else 'cpu'
                                    data = torch.from_numpy(np.load(prerender_path)).to(map_location)
                                elif prerender.endswith('.json'):
                                    # JSON file - load as structured data
                                    with open(prerender_path, 'r') as f:
                                        data = json.load(f)
                                elif prerender.endswith('.jsonl'):
                                    # JSONL file - convert to tensor
                                    map_location = 'cpu' if torch.cuda.is_available() else 'cpu'
                                    data = []
                                    with open(prerender_path, 'r') as f:
                                        for line in f:
                                            if line.strip():
                                                data.append(json.loads(line))
                                    data = torch.tensor(data, dtype=torch.float32).to(map_location)
                                elif prerender.endswith('.txt'):
                                    # If not numeric, load as text
                                    with open(prerender_path, 'r') as f:
                                        content = f.read()
                                        try:
                                            data = json.loads(content)
                                        except json.JSONDecodeError:
                                            # If not valid JSON, treat as plain text
                                            data = content
                                else:
                                    raise ValueError(f"Unsupported file format for prerender encoded data: {prerender}")
                                print(f"Pulling prerender encoded data for {prerender_name}...")
                                self.current_test[problem]['prerender_data_encoded'][prerender_name] = data
                except Exception as e:
                    print(f"No prerender encoded data found: {e}")
                    self.current_test[problem]['prerender_data_encoded'] = {}
            else:
                print("No prerender data found.")
                self.current_test[problem]['prerender_data'] = {}
                self.current_test[problem]['prerender_data_encoded'] = {}
            
            # Load prerender images
            self.current_test[problem]['prerender_images'] = {}
            if self.imports[problem]['prerender_data_folder'] != '':
                try:
                    for image_name, image in self.imports[problem]['prerender_image_filenames'].items():
                        if image.endswith(('.png', '.jpg', '.gif')):
                            image_path = os.path.join(prerender_dir, image)
                            if os.path.exists(image_path):
                                with open(image_path, 'rb') as f:
                                    image_data = f.read()
                                self.current_test[problem]['prerender_images'][image_name] = image_data
                    print("Pulling prerender images...")
                except Exception as e:
                    print(f"No prerender images found: {e}")
                    self.current_test[problem]['prerender_images'] = {}
            else:
                print("No prerender images found.")
                self.current_test[problem]['prerender_images'] = {}
            
            # Load instructions
            if self.imports[problem]['instruction_filenames'] != {}:
                try:
                    self.current_test[problem]['instructions'] = {}
                    instruction_dir = os.path.join(self.root, self.imports[problem]['instruction_folder'])
                    for instruction_name, instruction in self.imports[problem]['instruction_filenames'].items():
                        instruction_path = os.path.join(instruction_dir, instruction)
                        if os.path.exists(instruction_path):
                            with open(instruction_path, 'r') as f:
                                instruction_data = json.load(f)
                            self.current_test[problem]['instructions'][instruction_name] = instruction_data
                            print(f"Pulling instruction data for {instruction_name}...")
                except Exception as e:
                    print(f"No instruction data found: {e}")
                    self.current_test[problem]['instructions'] = {}
            else:
                print("No instructions found.")
                self.current_test[problem]['instructions'] = {}
            
            # Load README files
            self.current_test[problem]['readme_files'] = {}
            try:
                # Try to pull README.md from the root of the repository
                readme_path = os.path.join(self.root, 'README.md')
                if os.path.exists(readme_path):
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                    self.current_test[problem]['readme_files']['README.md'] = readme_content
                    print("Pulling README.md...")
                else:
                    # Try to pull README.txt from the root of the repository
                    readme_path = os.path.join(self.root, 'README.txt')
                    if os.path.exists(readme_path):
                        with open(readme_path, 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                        self.current_test[problem]['readme_files']['README.txt'] = readme_content
                        print("Pulling README.txt...")
                    else:
                        # Try to pull README.rst from the root of the repository
                        readme_path = os.path.join(self.root, 'README.rst')
                        if os.path.exists(readme_path):
                            with open(readme_path, 'r', encoding='utf-8') as f:
                                readme_content = f.read()
                            self.current_test[problem]['readme_files']['README.rst'] = readme_content
                            print("Pulling README.rst...")
                        else:
                            print("No README file found.")
                            self.current_test[problem]['readme_files'] = {}
            except Exception as e:
                print(f"Error loading README files: {e}")
                self.current_test[problem]['readme_files'] = {}
            
            # Add cache metadata and save to cache
            cache_metadata = {
                'commit_id': commit_id,
                'source_hash': hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest(),
                'timestamp': datetime.now().isoformat(),
                'import_data': source_data  # Store the import data specification
            }
            
            # For 'main' branch, check for last-commit-id marker in README files
            if commit_id == 'main':
                # Check for last-commit-id marker in README files
                if 'readme_files' in self.current_test[problem]:
                    for readme_content in self.current_test[problem]['readme_files'].values():
                        last_commit_id = self._extract_last_commit_id(readme_content)
                        if last_commit_id:
                            cache_metadata['last_commit_id'] = last_commit_id
                            print(f"Found last-commit-id marker in README for {problem}: {last_commit_id}")
                            break
            
            self.current_test[problem]['cache_metadata'] = cache_metadata
            
            # Save to cache and log the import
            self._save_to_cache(problem, self.current_test[problem])
            self._log_import(problem, commit_id, source_data, cache_hit=False)
            
            print(f"Successfully pulled fresh data for {problem}")
            
        except Exception as e:
            print(f"Error pulling fresh data for {problem}: {e}")
            # Fall back to normal import process
            pass
    
    def clear_cache(self, problem=None):
        """Clear cache for a specific problem or all problems."""
        try:
            if problem:
                # Clear specific problem cache
                cache_dir = self._get_cache_dir(problem)
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
                    print(f"Cleared cache for {problem}")
                else:
                    print(f"No cache found for {problem}")
            else:
                # Clear all cache
                if os.path.exists(self.cache_dir):
                    import shutil
                    shutil.rmtree(self.cache_dir)
                    print("Cleared all cache data")
                    # Recreate the cache directory
                    os.makedirs(self.cache_dir)
                else:
                    print("No cache directory found")
        except Exception as e:
            print(f"Failed to clear cache: {e}")
    
    def clear_corrupted_cache(self, problem=None):
        """Clear cache files that have JSON parsing errors."""
        try:
            if problem:
                # Check specific problem cache
                cache_dir = self._get_cache_dir(problem)
                metadata_file = self._get_cache_metadata_file(problem)
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            json.load(f)
                        print(f"Cache metadata for {problem} is valid")
                    except json.JSONDecodeError as e:
                        print(f"Cache metadata for {problem} is corrupted: {e}")
                        if os.path.exists(cache_dir):
                            import shutil
                            shutil.rmtree(cache_dir)
                            print(f"Cleared corrupted cache for {problem}")
            else:
                # Check all cache files
                if os.path.exists(self.cache_dir):
                    for problem_dir in os.listdir(self.cache_dir):
                        if os.path.isdir(os.path.join(self.cache_dir, problem_dir)):
                            self.clear_corrupted_cache(problem_dir)
        except Exception as e:
            print(f"Failed to clear corrupted cache: {e}")
    
    def get_cache_info(self):
        """Get information about cached data."""
        try:
            info = {}
            
            if os.path.exists(self.cache_dir):
                # Iterate through all problem directories
                for problem_dir in os.listdir(self.cache_dir):
                    if os.path.isdir(os.path.join(self.cache_dir, problem_dir)):
                        metadata_file = self._get_cache_metadata_file(problem_dir)
                        if os.path.exists(metadata_file):
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                
                                info[problem_dir] = {
                                    'commit_id': metadata.get('commit_id'),
                                    'timestamp': metadata.get('timestamp'),
                                    'source_hash': metadata.get('source_hash'),
                                    'last_commit_id': metadata.get('last_commit_id')
                                }
                                

                            except Exception as e:
                                info[problem_dir] = {'status': f'Error loading metadata: {e}'}
                        else:
                            info[problem_dir] = {'status': 'No metadata available'}
            
            return info
        except Exception as e:
            print(f"Failed to get cache info: {e}")
            return {}
    
    def force_refresh(self, problem_selection:list=[]):
        """Force refresh by clearing cache and re-importing."""
        if len(problem_selection) > 0:
            for problem in problem_selection:
                self.clear_cache(problem)
        else:
            self.clear_cache()  # Clear all cache
        
        return self.pull(problem_selection)
    
    def check_for_updates(self, problem: str):
        """Check if an application has updates available."""
        if problem not in self.imports:
            return {'has_updates': False, 'error': f"Application '{problem}' not found"}
        
        source_data = self.imports[problem]
        commit_id = source_data.get('commit_id', 'main')
        
        # If commit_id is '*', get the latest commit
        if commit_id == '*':
            try:
                github_user = source_data['github_user']
                repository = source_data['repository']
                commit_id = self._get_latest_commit_id(github_user, repository)
            except Exception as e:
                return {'has_updates': False, 'error': f"Failed to get latest commit: {e}"}
        
        # Check if repository exists
        repo_dir = self._get_git_repo_dir(problem)
        if not os.path.exists(repo_dir):
            return {'has_updates': False, 'error': 'Repository not found locally'}
        
        # Check for updates
        try:
            has_updates, current_commit, remote_commit = self._check_repository_updates(problem, commit_id)
            if has_updates is False and current_commit is None:
                # This means there was an error checking for updates
                return {'has_updates': False, 'error': 'Failed to check for updates'}
            
            return {
                'has_updates': has_updates,
                'current_commit': current_commit,
                'remote_commit': remote_commit,
                'current_commit_short': current_commit[:7] if current_commit else None,
                'remote_commit_short': remote_commit[:7] if remote_commit else None
            }
        except Exception as e:
            return {'has_updates': False, 'error': f"Error checking updates: {e}"}

    def download_application(self, problem: str, force_update: bool = False):
        """Download a specific application and cache it locally."""
        if problem not in self.imports:
            raise ValueError(f"Application '{problem}' not found in available applications")
        
        source_data = self.imports[problem]
        github_user = source_data['github_user']
        repository = source_data['repository']
        commit_id = source_data.get('commit_id', 'main')
        
        print(f"Downloading application: {problem}")
        print(f"Repository: {github_user}/{repository}")
        print(f"Commit: {commit_id}")
        
        try:
            # Clone or update the repository
            if commit_id == '*':
                # Get the latest commit
                print(f"Getting latest commit for {github_user}/{repository}...")
                commit_id = self._get_latest_commit_id(github_user, repository)
                print(f"Using commit: {commit_id}")
            
            # Check if repository exists and is up to date
            repo_dir = self._get_git_repo_dir(problem)
            if not os.path.exists(repo_dir):
                print(f"Cloning repository for {problem}...")
                if not self._clone_repository(problem, github_user, repository, commit_id):
                    print(f"Failed to clone repository for {problem}")
                    return False
            else:
                # Check for updates if not forcing
                if not force_update:
                    has_updates, current_commit, remote_commit = self._check_repository_updates(problem, commit_id)
                    if has_updates:
                        print(f"Updates available for {problem}: {current_commit[:7]} -> {remote_commit[:7]}")
                        # Return update info instead of proceeding
                        return {
                            'needs_confirmation': True,
                            'current_commit': current_commit,
                            'remote_commit': remote_commit,
                            'current_commit_short': current_commit[:7],
                            'remote_commit_short': remote_commit[:7]
                        }
                
                print(f"Updating repository for {problem}...")
                update_result = self._update_repository(problem, commit_id)
                if not update_result or (isinstance(update_result, tuple) and not update_result[0]):
                    print(f"Failed to update repository for {problem}")
                    return False
            
            # Pull fresh data and cache it
            print(f"Processing and caching data for {problem}...")
            self._pull_fresh_data(problem, commit_id, source_data)
            
            print(f"Successfully downloaded and cached {problem}")
            return True
            
        except Exception as e:
            print(f"Error downloading {problem}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def pull(self, problem_selection:list=[]):
        # Pull all problems if none are selected
        if len(problem_selection)>0:
            self.problem_selection = problem_selection
        else:
            self.problem_selection = list(self.imports.keys())
        
        # Extract data from imports
        for problem in list(self.problem_selection):
            print("-----------------------------------------------")
            print(problem)
            engine = self.imports[problem]['engine_filename']
            if problem not in self.imports:
                raise ValueError(f"Problem {problem} not found in the setup tests.")
            else:
                self.current_test[problem] = {}
                # Store engine filename
                self.current_test[problem]['engine_filename'] = engine
                # If commit ID is '*' or empty, use main branch
                if self.imports[problem]['commit_id'] in ['*', '']:
                    # Update commit_id to use main branch
                    self.imports[problem]['commit_id'] = 'main'
                    print('Pulling data from current version of main branch.')
                
                # Prepare source data for caching
                source_data = {
                    'engine_folder': self.imports[problem]['engine_folder'],
                    'engine_filename': self.imports[problem]['engine_filename'],
                    'config_folder': self.imports[problem]['config_folder'],
                    'experiment_config_filenames': self.imports[problem]['experiment_config_filenames'],
                    'local_config_filenames': self.imports[problem]['local_config_filenames'],
                    'local_adapter_folder': self.imports[problem]['local_adapter_folder'],
                    'adapter_filenames': self.imports[problem]['adapter_filenames'],
                    'local_analysis_folder': self.imports[problem]['local_analysis_folder'],
                    'local_analysis_filenames': self.imports[problem]['local_analysis_filenames'],
                    'prerender_data_folder': self.imports[problem]['prerender_data_folder'],
                    'prerender_data_filenames': self.imports[problem]['prerender_data_filenames'],
                    'prerender_data_encoded_filenames': self.imports[problem]['prerender_data_encoded_filenames'],
                    'prerender_image_filenames': self.imports[problem]['prerender_image_filenames'],
                    'instruction_folder': self.imports[problem]['instruction_folder'],
                    'instruction_filenames': self.imports[problem]['instruction_filenames'],
                    'readme_files': ['README.md', 'README.txt', 'README.rst']  # Standard README file names to check
                }
                
                commit_id = self.imports[problem]['commit_id']
                
                # Try to load from cache first
                cached_data = self._load_from_cache(problem, commit_id, source_data)
                if cached_data:
                    self.current_test[problem] = cached_data
                    self._log_import(problem, commit_id, source_data, cache_hit=True)
                    continue
                
                # If not in cache, proceed with git-based import
                print(f"Cache miss for {problem}, importing from git repository...")
                self._pull_fresh_data(problem, commit_id, source_data)
            
            # Add cache metadata and save to cache (only if not already cached)
            if 'cache_metadata' not in self.current_test[problem]:
                cache_metadata = {
                    'commit_id': commit_id,
                    'source_hash': hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest(),
                    'timestamp': datetime.now().isoformat(),
                    'import_data': source_data  # Store the import data specification
                }
                
                # For 'main' branch, check for last-commit-id marker in README files
                if commit_id == 'main':
                    # Check for last-commit-id marker in README files
                    if 'readme_files' in self.current_test[problem]:
                        for readme_content in self.current_test[problem]['readme_files'].values():
                            last_commit_id = self._extract_last_commit_id(readme_content)
                            if last_commit_id:
                                cache_metadata['last_commit_id'] = last_commit_id
                                print(f"Found last-commit-id marker in README for {problem}: {last_commit_id}")
                                break
                
                self.current_test[problem]['cache_metadata'] = cache_metadata
                
                # Save to cache and log the import
                self._save_to_cache(problem, self.current_test[problem])
                self._log_import(problem, commit_id, source_data, cache_hit=False)
            
        print("-----------------------------------------------")
        return self.current_test

    def setup(self, agent_config:dict={}) -> None:
        if agent_config == {}:
            agent_config = DefaultAgentConfig()
            self.ExperimentConfig = agent_config.data  
        else:
            self.ExperimentConfig = agent_config 

        return self.ExperimentConfig
    
    def add_applicaiton(self, problem:str, application_data:dict) -> None:
        """ Add a new application to the list of applications. 
        Reqired form:
            - engine: <engine.py>
            - experiment_configs: {experiment_config:experiment_config_path|<experiment_config.json>}
            - local_configs: {local_config:experiment_config_path|<local_config.json>}
            - adapters: {adapter:<adapter.py>}
            - local_analysis: {analysis:<analysis.py>}
            - prerender_data: {data:data_path|<data.txt>}
            - prerender_images: {image:<image.png>}
        """
        # ---
        # Get configs and data from path directory to imported json/txt files
        if type(list(application_data['experiment_configs'].values())[0])==str:
            experiment_config = {}
            for name,experiment_config_dir in application_data['experiment_configs'].items():
                with open (experiment_config_dir, 'r') as f:
                    # Load the JSON data from the file
                    agent_config = json.loads(f.read())
                    experiment_config[name] = agent_config
            application_data['experiment_configs'] = experiment_config

        if type(list(application_data['local_configs'].values())[0])==str:
            local_config = {}
            for name,local_config_dir in application_data['local_configs'].items():
                with open (local_config_dir, 'r') as f:
                    # Load the JSON data from the file
                    agent_config = json.loads(f.read())
                    local_config[name] = agent_config
            application_data['local_configs'] = local_config

        if len(application_data['prerender_data'])>0:
            if type(list(application_data['prerender_data'].values())[0])==str:
                data = {}
                for name,data_dir in application_data['prerender_data'].items():
                    with open (data_dir, 'r') as f:
                        # Load the JSON data from the file
                        agent_config = json.loads(f.read().decode('utf-8'))
                        data[name] = agent_config
                application_data['prerender_data'] = data
        # ---
        self.imports[problem] = application_data
        self.current_test[problem] = application_data
        print(f"Added {problem} to the list of applications.")
        print(f"Current applications: {self.imports.keys()}")

        return self.current_test
    
    def remove_application(self, problem:str) -> None:
        # Remove an application from the list of applications
        if problem in self.imports:
            del self.imports[problem]
            del self.current_test[problem]
            print(f"Removed {problem} from the list of applications.")
        else:
            print(f"{problem} not found in the list of applications.")

        return self.imports

            