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
            except (json.JSONDecodeError, FileNotFoundError):
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
                engine = self.imports[problem]['engine_filename']
                engine_module = httpimport.load(engine.split('.')[0], self.root+'/'+self.imports[problem]['engine_folder'])  
                engine_content = str(urllib.request.urlopen(engine_module.__file__).read().decode('utf-8')) 
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
                    adapter = self.imports[problem]['adapter_filenames'][adapter_name]
                    adapter_module = httpimport.load(adapter.split('.')[0], self.root+'/'+self.imports[problem]['local_adapter_folder'])
                    adapter_content = str(urllib.request.urlopen(adapter_module.__file__).read().decode('utf-8'))
                    with open(adapter_file, 'w') as f:
                        f.write(adapter_content)
                    print(f"Saved adapter file: {adapter_filename}")
            
            # Save experiment configs
            if 'experiment_configs' in data:
                configs_dir = os.path.join(cache_dir, 'experiment_configs')
                if not os.path.exists(configs_dir):
                    os.makedirs(configs_dir)
                for config_name, config_data in data['experiment_configs'].items():
                    config_file = os.path.join(configs_dir, f"{config_name}.json")
                    with open(config_file, 'w') as f:
                        json.dump(config_data, f, indent=2)
            
            # Save local configs
            if 'local_configs' in data:
                local_configs_dir = os.path.join(cache_dir, 'local_configs')
                if not os.path.exists(local_configs_dir):
                    os.makedirs(local_configs_dir)
                for config_name, config_data in data['local_configs'].items():
                    config_file = os.path.join(local_configs_dir, f"{config_name}.json")
                    with open(config_file, 'w') as f:
                        json.dump(config_data, f, indent=2)
            
            # Save prerender data
            if 'prerender_data' in data:
                prerender_dir = os.path.join(cache_dir, 'prerender_data')
                if not os.path.exists(prerender_dir):
                    os.makedirs(prerender_dir)
                for data_name, data_content in data['prerender_data'].items():
                    data_file = os.path.join(prerender_dir, f"{data_name}.json")
                    with open(data_file, 'w') as f:
                        json.dump(data_content, f, indent=2)
            
            # Save prerender data encoded (as numpy arrays)
            if 'prerender_data_encoded' in data:
                prerender_encoded_dir = os.path.join(cache_dir, 'prerender_data_encoded')
                if not os.path.exists(prerender_encoded_dir):
                    os.makedirs(prerender_encoded_dir)
                for data_name, data_content in data['prerender_data_encoded'].items():
                    data_file = os.path.join(prerender_encoded_dir, f"{data_name}.npy")
                    np.save(data_file, data_content.cpu().numpy())
            
            # Save prerender images
            if 'prerender_images' in data:
                images_dir = os.path.join(cache_dir, 'prerender_images')
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                for image_name, image_data in data['prerender_images'].items():
                    # Determine file extension from image name or use default
                    if '.' in image_name:
                        ext = image_name.split('.')[-1]
                        image_file = os.path.join(images_dir, image_name)
                    else:
                        image_file = os.path.join(images_dir, f"{image_name}.png")
                    with open(image_file, 'wb') as f:
                        f.write(image_data)
            
            # Save instructions
            if 'instructions' in data:
                instructions_dir = os.path.join(cache_dir, 'instructions')
                if not os.path.exists(instructions_dir):
                    os.makedirs(instructions_dir)
                for instruction_name, instruction_data in data['instructions'].items():
                    instruction_file = os.path.join(instructions_dir, f"{instruction_name}.json")
                    with open(instruction_file, 'w') as f:
                        json.dump(instruction_data, f, indent=2)
            
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
        """Load data from cache directory structure if available and up-to-date."""
        try:
            cache_dir = self._get_cache_dir(problem)
            metadata_file = self._get_cache_metadata_file(problem)
            
            if not os.path.exists(cache_dir) or not os.path.exists(metadata_file):
                return None
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is valid by comparing commit_id and source data
            cached_commit = metadata.get('commit_id')
            cached_source_hash = metadata.get('source_hash')
            current_source_hash = hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest()
            
            # For 'main' branch, check if we need to update based on last-commit-id marker
            if commit_id == 'main':
                # Load cached data to check README content
                cached_data = self._load_cached_data(problem, cache_dir)
                if cached_data and 'readme_files' in cached_data:
                    # Check if any README file contains a last-commit-id marker
                    cached_last_commit_id= None
                    for readme_content in cached_data['readme_files'].values():
                        cached_last_commit_id= self._extract_last_commit_id(readme_content)
                        if cached_last_commit_id:
                            break
                    
                    if cached_last_commit_id:
                        # For main branch with last-commit-id marker, use cache
                        print(f"Using cached data for {problem} (main branch with last-commit-id: {cached_last_commit_id})")
                        return cached_data
                    else:
                        print(f"No last-commit-id marker found in README for {problem}, pulling fresh data")
                        return None
                else:
                    print(f"No cached README data for {problem}, pulling fresh data")
                    return None
            
            if cached_commit == commit_id and cached_source_hash == current_source_hash:
                print(f"Using cached data for {problem} (commit: {commit_id})")
                return self._load_cached_data(problem, cache_dir)
            
            return None
        except Exception as e:
            print(f"Failed to load cache for {problem}: {e}")
            return None
    
    def _load_cached_data(self, problem, cache_dir):
        """Load cached data from directory structure."""
        try:
            data = {'cache_metadata': {}}
            
            # Load metadata
            metadata_file = self._get_cache_metadata_file(problem)
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    data['cache_metadata'] = json.load(f)
            
            # Load engine
            engine_dir = os.path.join(cache_dir, 'engine')
            if os.path.exists(engine_dir):
                engine_files = [f for f in os.listdir(engine_dir) if f.endswith('.py')]
                if engine_files:
                    # Add engine directory to Python path temporarily
                    import sys
                    sys.path.insert(0, engine_dir)
                    
                    try:
                        # Get the engine filename from the imports data
                        engine_filename = self.imports[problem]['engine_filename']
                        engine_module_name = engine_filename.split('.')[0]
                        engine_module = __import__(engine_module_name)
                        data['engine'] = engine_module.Engine
                        print(f"Loaded cached engine: {engine_filename}")
                    except Exception as e:
                        print(f"Failed to load cached engine: {e}")
                    finally:
                        # Remove from path
                        sys.path.pop(0)
            
            # Load adapters
            adapters_dir = os.path.join(cache_dir, 'adapters')
            if os.path.exists(adapters_dir):
                data['adapters'] = {}
                adapter_files = [f for f in os.listdir(adapters_dir) if f.endswith('.py')]
                if adapter_files:
                    # Add adapters directory to Python path temporarily
                    import sys
                    sys.path.insert(0, adapters_dir)
                    
                    try:
                        for adapter_name, adapter_filename in self.imports[problem]['adapter_filenames'].items():
                            adapter_module_name = adapter_filename.split('.')[0]
                            adapter_module = __import__(adapter_module_name)
                            data['adapters'][adapter_name] = adapter_module.Adapter
                            print(f"Loaded cached adapter: {adapter_filename}")
                    except Exception as e:
                        print(f"Failed to load cached adapters: {e}")
                    finally:
                        # Remove from path
                        sys.path.pop(0)
            
            # Load experiment configs
            configs_dir = os.path.join(cache_dir, 'experiment_configs')
            if os.path.exists(configs_dir):
                data['experiment_configs'] = {}
                for config_file in os.listdir(configs_dir):
                    if config_file.endswith('.json'):
                        config_name = config_file[:-5]  # Remove .json extension
                        with open(os.path.join(configs_dir, config_file), 'r') as f:
                            data['experiment_configs'][config_name] = json.load(f)
            
            # Load local configs
            local_configs_dir = os.path.join(cache_dir, 'local_configs')
            if os.path.exists(local_configs_dir):
                data['local_configs'] = {}
                for config_file in os.listdir(local_configs_dir):
                    if config_file.endswith('.json'):
                        config_name = config_file[:-5]  # Remove .json extension
                        with open(os.path.join(local_configs_dir, config_file), 'r') as f:
                            data['local_configs'][config_name] = json.load(f)
            
            # Load prerender data
            prerender_dir = os.path.join(cache_dir, 'prerender_data')
            if os.path.exists(prerender_dir):
                data['prerender_data'] = {}
                for data_file in os.listdir(prerender_dir):
                    if data_file.endswith('.json'):
                        data_name = data_file[:-5]  # Remove .json extension
                        with open(os.path.join(prerender_dir, data_file), 'r') as f:
                            data['prerender_data'][data_name] = json.load(f)
            
            # Load prerender data encoded
            prerender_encoded_dir = os.path.join(cache_dir, 'prerender_data_encoded')
            if os.path.exists(prerender_encoded_dir):
                data['prerender_data_encoded'] = {}
                for data_file in os.listdir(prerender_encoded_dir):
                    if data_file.endswith('.npy'):
                        data_name = data_file[:-4]  # Remove .npy extension
                        array_data = np.load(os.path.join(prerender_encoded_dir, data_file))
                        data['prerender_data_encoded'][data_name] = torch.from_numpy(array_data)
            
            # Load prerender images
            images_dir = os.path.join(cache_dir, 'prerender_images')
            if os.path.exists(images_dir):
                data['prerender_images'] = {}
                for image_file in os.listdir(images_dir):
                    image_name = image_file
                    with open(os.path.join(images_dir, image_file), 'rb') as f:
                        data['prerender_images'][image_name] = f.read()
            
            # Load instructions
            instructions_dir = os.path.join(cache_dir, 'instructions')
            if os.path.exists(instructions_dir):
                data['instructions'] = {}
                for instruction_file in os.listdir(instructions_dir):
                    if instruction_file.endswith('.json'):
                        instruction_name = instruction_file[:-5]  # Remove .json extension
                        with open(os.path.join(instructions_dir, instruction_file), 'r') as f:
                            data['instructions'][instruction_name] = json.load(f)
            
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
    

    
    def _pull_fresh_data(self, problem, commit_id, source_data):
        """Pull fresh data for a problem and cache it."""
        try:
            print(f"Pulling fresh data for {problem}...")
            self.root = 'https://raw.githubusercontent.com/'+ self.imports[problem]['github_user'] + "/" + self.imports[problem]['repository'] + "/" + commit_id
            print("Source: ", self.root)
            
            # Initialize the problem data structure
            self.current_test[problem] = {}
            self.current_test[problem]['engine_filename'] = self.imports[problem]['engine_filename']
            self.current_test[problem]['source'] = {str(self.root): source_data}
            
            # Load engine
            engine = self.imports[problem]['engine_filename']
            engine_module = httpimport.load(engine.split('.')[0], self.root+'/'+self.imports[problem]['engine_folder'])
            try:
                self.current_test[problem]['engine'] = engine_module.Engine
            except:
                print("Engine error, attempting to install requirements.")
                try:
                    requirements = urllib.request.urlopen(self.root+'/'+'requirements.txt').read()
                    requirements = requirements.decode('utf-8').split('\n')
                    for req in requirements:
                        if req.strip():
                            try:
                                subprocess.check_call([sys.executable, "-m", "pip", "install", req.strip()])
                                print(f"Successfully installed {req}")
                            except subprocess.CalledProcessError:
                                print(f"Failed to install {req}")
                    self.current_test[problem]['engine'] = engine_module.Engine
                    print("Successfully loaded engine after installing requirements.")
                except:
                    print("Failed to load engine and no requirements.txt found.")

            # Load adapters
            self.current_test[problem]['adapters'] = {}
            for adapter_name, adapter in self.imports[problem]['adapter_filenames'].items():
                adapter_module = httpimport.load(adapter.split('.')[0], self.root+'/'+self.imports[problem]['local_adapter_folder'])
                self.current_test[problem]['adapters'][adapter_name] = adapter_module.Adapter
            
            # Load experiment configs
            self.current_test[problem]['experiment_configs'] = {}
            for config_name, config in self.imports[problem]['experiment_config_filenames'].items():
                experiment_config = json.loads(urllib.request.urlopen(self.root+'/'+self.imports[problem]['config_folder']+'/'+config).read())
                self.current_test[problem]['experiment_configs'][config_name] = experiment_config
            
            # Load local configs
            self.current_test[problem]['local_configs'] = {}
            for config_name, config in self.imports[problem]['local_config_filenames'].items():
                local_config = json.loads(urllib.request.urlopen(self.root+'/'+self.imports[problem]['config_folder']+'/'+config).read())
                self.current_test[problem]['local_configs'][config_name] = local_config
            
            # Load local analysis
            self.current_test[problem]['local_analysis'] = {}
            for analysis_name, analysis in self.imports[problem]['local_analysis_filenames'].items():
                try:
                    local_analysis = httpimport.load(analysis, self.root+'/'+self.imports[problem]['local_analysis_folder'])
                    self.current_test[problem]['local_analysis'][analysis_name] = local_analysis.Analysis
                except:
                    print("No analysis file found.")
                    self.current_test[problem]['local_analysis'][analysis_name] = {}
            
            # Load prerender data
            self.current_test[problem]['prerender_data'] = {}
            self.current_test[problem]['prerender_data_encoded'] = {}
            if self.imports[problem]['prerender_data_folder'] != '':
                print("Pulling prerender data...")
                try:
                    for prerender_name, prerender in self.imports[problem]['prerender_data_filenames'].items():
                        if prerender.endswith(('.txt', '.json', '.xml', '.jsonl')):
                            if prerender.endswith('.jsonl') or prerender.endswith('.json'):
                                data = {}
                                with urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender) as f:
                                    for line in f:
                                        row = (json.loads(line.decode('utf-8')))
                                        data.update(row)
                            elif prerender.endswith('.txt'):
                                data = json.loads(urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender).read().decode('utf-8'))
                            elif prerender.endswith('.xml'):
                                import xml.etree.ElementTree as ET
                                tree = ET.parse(urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender))
                                root_xml = tree.getroot()
                                data = []
                                for elem in root_xml.findall('.//data'):
                                    data.append(float(elem.text))
                            else:
                                raise ValueError(f"Unsupported file format for prerender data: {prerender}")
                            print(f"Pulling prerender data for {prerender_name}...")
                            self.current_test[problem]['prerender_data'][prerender_name] = data
                except:
                    print("No prerender data found.")
                    self.current_test[problem]['prerender_data'] = {}
                
                try:
                    for prerender_name, prerender in self.imports[problem]['prerender_data_encoded_filenames'].items():
                        if prerender.endswith(('.txt', '.json', '.xml', '.jsonl')):
                            map_location = 'cpu' if torch.cuda.is_available() else 'cpu'
                            if (prerender.endswith('.jsonl') or prerender.endswith('.json')):
                                data = []
                                with urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender) as f:
                                    for line in f:
                                        data.append(json.loads(line.decode('utf-8')))
                                data = torch.tensor(data, dtype=torch.float32).to(map_location)
                            elif prerender.endswith('.txt'):
                                data = torch.from_numpy(np.loadtxt(urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender), dtype=np.float32)).to(map_location)
                            elif prerender.endswith('.xml'):
                                import xml.etree.ElementTree as ET
                                tree = ET.parse(urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender))
                                root_xml = tree.getroot()
                                data = []
                                for elem in root_xml.findall('.//data'):
                                    data.append(float(elem.text))
                                data = torch.tensor(data, dtype=torch.float32).to(map_location)
                            else:
                                raise ValueError(f"Unsupported file format for prerender data: {prerender}")
                            print(f"Pulling prerender encoded data for {prerender_name}...")
                            self.current_test[problem]['prerender_data_encoded'][prerender_name] = data
                except:
                    print("No prerender encoded data found.")
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
                        if image.endswith(('.png', '.jpg', '.svg', '.gif')):
                            image_url = self.root + '/' + self.imports[problem]['prerender_data_folder'] + '/' + image
                            image_data = urllib.request.urlopen(image_url).read()
                            self.current_test[problem]['prerender_images'][image_name] = image_data
                    print("Pulling prerender images...")
                except:
                    print("No prerender images found.")
                    self.current_test[problem]['prerender_images'] = {}
            else:
                print("No prerender images found.")
                self.current_test[problem]['prerender_images'] = {}
            
            # Load instructions
            if self.imports[problem]['instruction_filenames'] != {}:
                try:
                    self.current_test[problem]['instructions'] = {}
                    for instruction_name, instruction in self.imports[problem]['instruction_filenames'].items():
                        instruction_data = json.loads(urllib.request.urlopen(self.root+'/'+self.imports[problem]['instruction_folder']+'/'+instruction).read())
                        self.current_test[problem]['instructions'][instruction_name] = instruction_data
                        print(f"Pulling instruction data for {instruction_name}...")
                except:
                    print("No instruction data found.")
                    self.current_test[problem]['instructions'] = {}
            else:
                print("No instructions found.")
                self.current_test[problem]['instructions'] = {}
            
            # Load README files
            self.current_test[problem]['readme_files'] = {}
            try:
                # Try to pull README.md from the root of the repository
                readme_url = self.root + '/README.md'
                readme_content = urllib.request.urlopen(readme_url).read().decode('utf-8')
                self.current_test[problem]['readme_files']['README.md'] = readme_content
                print("Pulling README.md...")
            except:
                try:
                    # Try to pull README.txt from the root of the repository
                    readme_url = self.root + '/README.txt'
                    readme_content = urllib.request.urlopen(readme_url).read().decode('utf-8')
                    self.current_test[problem]['readme_files']['README.txt'] = readme_content
                    print("Pulling README.txt...")
                except:
                    try:
                        # Try to pull README.rst from the root of the repository
                        readme_url = self.root + '/README.rst'
                        readme_content = urllib.request.urlopen(readme_url).read().decode('utf-8')
                        self.current_test[problem]['readme_files']['README.rst'] = readme_content
                        print("Pulling README.rst...")
                    except:
                        print("No README file found.")
                        self.current_test[problem]['readme_files'] = {}
            
            # Add cache metadata and save to cache
            cache_metadata = {
                'commit_id': commit_id,
                'source_hash': hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest(),
                'timestamp': datetime.now().isoformat()
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
                
                # If not in cache, proceed with normal import
                print(f"Cache miss for {problem}, importing from source...")
                self.root = 'https://raw.githubusercontent.com/'+ self.imports[problem]['github_user'] + "/" + self.imports[problem]['repository'] + "/" + commit_id
                print("Source: ", self.root)
                # ------------------------------------------------
                # - Pull Engine
                # NOTE - This requires repo to match structure with engine inside environment folder
                engine_module = httpimport.load(engine.split('.')[0], self.root+'/'+self.imports[problem]['engine_folder']) 
                # TODO: Pull class name directly from engine file to be called
                self.current_test[problem]['source'] = {str(self.root): source_data}
                try:
                    self.current_test[problem]['engine'] = engine_module.Engine
                except:
                    print("Engine error, attempting to install requirements.")
                    try:
                        requirements = urllib.request.urlopen(self.root+'/'+'requirements.txt').read()
                        # Install packages from requirements.txt
                        requirements = requirements.decode('utf-8').split('\n')
                        for req in requirements:
                            if req.strip():  # Skip empty lines
                                try:
                                    subprocess.check_call([sys.executable, "-m", "pip", "install", req.strip()])
                                    print(f"Successfully installed {req}")
                                except subprocess.CalledProcessError:
                                    print(f"Failed to install {req}")
                        # Try importing engine again after installing requirements
                        self.current_test[problem]['engine'] = engine_module.Engine
                        print("Successfully loaded engine after installing requirements.")
                    except:
                        print("Failed to load engine and no requirements.txt found.")
                # ------------------------------------------------
                # - Pull Adapters, Configs and Analysis
                self.current_test[problem]['adapters'] = {}
                for adapter_name, adapter in self.imports[problem]['adapter_filenames'].items():
                    adapter_module = httpimport.load(adapter.split('.')[0], self.root+'/'+self.imports[problem]['local_adapter_folder'])   
                    # TODO: Pull class name directly from adapter file to be 
                    self.current_test[problem]['adapters'][adapter_name] = adapter_module.Adapter
                # ---
                self.current_test[problem]['experiment_configs'] = {}
                for config_name,config in self.imports[problem]['experiment_config_filenames'].items():
                    experiment_config = json.loads(urllib.request.urlopen(self.root+'/'+self.imports[problem]['config_folder']+'/'+config).read())
                    self.current_test[problem]['experiment_configs'][config_name] = experiment_config
                # ---
                self.current_test[problem]['local_configs'] = {}
                for config_name,config in self.imports[problem]['local_config_filenames'].items():
                    local_config = json.loads(urllib.request.urlopen(self.root+'/'+self.imports[problem]['config_folder']+'/'+config).read())
                    self.current_test[problem]['local_configs'][config_name] = local_config
                # ---
                self.current_test[problem]['local_analysis'] = {}
                for analysis_name,analysis in self.imports[problem]['local_analysis_filenames'].items():
                    try:
                        local_analysis = httpimport.load(analysis, self.root+'/'+self.imports[problem]['local_analysis_folder'])  
                        # TODO: Pull class name directly from analysis file to be called 
                        self.current_test[problem]['local_analysis'][analysis_name] = local_analysis.Analysis
                    except:
                        print("No analysis file found.")
                        self.current_test[problem]['local_analysis'][analysis_name] = {}
                
                # ------------------------------------------------
                # Pull prerender data
                self.current_test[problem]['prerender_data'] = {}
                self.current_test[problem]['prerender_data_encoded'] = {}
                if self.imports[problem]['prerender_data_folder'] != '':
                    print("Pulling prerender data...")
                    try:
                        for prerender_name, prerender in self.imports[problem]['prerender_data_filenames'].items():
                            if prerender.endswith(('.txt', '.json', '.xml', '.jsonl')):
                                # Load JSON or text file
                                if prerender.endswith('.jsonl') or prerender.endswith('.json'):
                                    # Load JSONL file
                                    data = {}
                                    with urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender) as f:
                                        for line in f:
                                            row = (json.loads(line.decode('utf-8')))
                                            data.update(row)
                                elif prerender.endswith('.txt'):
                                    # Load text file
                                    data = json.loads(urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender).read().decode('utf-8'))
                                elif prerender.endswith('.xml'):
                                    # Load XML file (assuming it contains numerical data)
                                    import xml.etree.ElementTree as ET
                                    tree = ET.parse(urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender))
                                    root_xml = tree.getroot()
                                    data = []
                                    for elem in root_xml.findall('.//data'):
                                        data.append(float(elem.text))
                                else:
                                    raise ValueError(f"Unsupported file format for prerender data: {prerender}")
                                print(f"Pulling prerender data for {prerender_name}...")
                                self.current_test[problem]['prerender_data'][prerender_name] = data
                    except:
                        print(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender)
                        print("No prerender data found.")
                        self.current_test[problem]['prerender_data'] = {}
                    try:
                        for prerender_name, prerender in self.imports[problem]['prerender_data_encoded_filenames'].items():
                            if prerender.endswith(('.txt', '.json', '.xml', '.jsonl')):
                                map_location= 'cpu' if torch.cuda.is_available() else 'cpu'
                                if (prerender.endswith('.jsonl') or prerender.endswith('.json')):
                                    # Load JSONL file
                                    data = []
                                    with urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender) as f:
                                        for line in f:
                                            data.append(json.loads(line.decode('utf-8')))
                                    data = torch.tensor(data, dtype=torch.float32).to(map_location)
                                elif prerender.endswith('.txt'):
                                    data = torch.from_numpy(np.loadtxt(urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender), dtype=np.float32)).to(map_location)
                                elif prerender.endswith('.xml'):
                                    # Load XML file (assuming it contains numerical data)
                                    import xml.etree.ElementTree as ET
                                    tree = ET.parse(urllib.request.urlopen(self.root+'/'+self.imports[problem]['prerender_data_folder']+'/'+prerender))
                                    root_xml = tree.getroot()
                                    data = []
                                    for elem in root_xml.findall('.//data'):
                                        data.append(float(elem.text))
                                    data = torch.tensor(data, dtype=torch.float32).to(map_location)
                                else:
                                    raise ValueError(f"Unsupported file format for prerender data: {prerender}")
                                print(f"Pulling prerender encoded data for {prerender_name}...")
                                self.current_test[problem]['prerender_data_encoded'][prerender_name] = data
                    except:
                        print("No prerender encoded data found.")
                        self.current_test[problem]['prerender_data_encoded'] = {}
                else:
                    print("No prerender data found.")
                    self.current_test[problem]['prerender_data'] = {}
                    self.current_test[problem]['prerender_data_encoded'] = {}
                # ------------------------------------------------
                # Pull prerender images
                self.current_test[problem]['prerender_images'] = {}
                if self.imports[problem]['prerender_data_folder'] != '':
                    try:
                        for image_name, image in self.imports[problem]['prerender_image_filenames'].items():
                            if image.endswith(('.png', '.jpg', '.svg', '.gif')):
                                image_url = self.root + '/' + self.imports[problem]['prerender_data_folder'] + '/' + image
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
                # Pull instructions
                if self.imports[problem]['instruction_filenames'] != {}:
                    try:
                        self.current_test[problem]['instructions'] = {}
                        for instruction_name, instruction in self.imports[problem]['instruction_filenames'].items():
                            instruction_data = json.loads(urllib.request.urlopen(self.root+'/'+self.imports[problem]['instruction_folder']+'/'+instruction).read())
                            self.current_test[problem]['instructions'][instruction_name] = instruction_data
                            print(f"Pulling instruction data for {instruction_name}...")
                    except:
                        print("No instruction data found.")
                        self.current_test[problem]['instructions'] = {}
                else:
                    print("No instructions found.")
                    self.current_test[problem]['instructions'] = {}
                
                # -----------------------------------------------
                # Pull README files
                self.current_test[problem]['readme_files'] = {}
                try:
                    # Try to pull README.md from the root of the repository
                    readme_url = self.root + '/README.md'
                    readme_content = urllib.request.urlopen(readme_url).read().decode('utf-8')
                    self.current_test[problem]['readme_files']['README.md'] = readme_content
                    print("Pulling README.md...")
                except:
                    try:
                        # Try to pull README.txt from the root of the repository
                        readme_url = self.root + '/README.txt'
                        readme_content = urllib.request.urlopen(readme_url).read().decode('utf-8')
                        self.current_test[problem]['readme_files']['README.txt'] = readme_content
                        print("Pulling README.txt...")
                    except:
                        try:
                            # Try to pull README.rst from the root of the repository
                            readme_url = self.root + '/README.rst'
                            readme_content = urllib.request.urlopen(readme_url).read().decode('utf-8')
                            self.current_test[problem]['readme_files']['README.rst'] = readme_content
                            print("Pulling README.rst...")
                        except:
                            print("No README file found.")
                            self.current_test[problem]['readme_files'] = {}
                # -----------------------------------------------
            
            # Add cache metadata and save to cache
            if 'cache_metadata' not in self.current_test[problem]:
                cache_metadata = {
                    'commit_id': commit_id,
                    'source_hash': hashlib.md5(json.dumps(source_data, sort_keys=True).encode()).hexdigest(),
                    'timestamp': datetime.now().isoformat()
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

            