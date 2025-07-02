import sys
import os
import httpimport
import shutil
from datetime import datetime
import json
import threading
import queue
import uuid
from flask import Response # Added Response for SSE
import requests
from markdown import markdown
import torch
# App tools
from flask import Flask, render_template, request, jsonify, send_from_directory

# elsci methods
from elsciRL.instruction_following.elsciRL_GUI_search import elsciRLSearch as elsci_search
from elsciRL.instruction_following.elsciRL_instruction_following import elsciRLOptimize
from elsciRL.experiments.standard import Experiment as STANDARD_RL
# Get application data
from elsciRL.application_suite.import_data import Applications

# elsciRL LLM Instruction Following
from elsciRL.instruction_following.LLM_instr_planner.LLM_instr_generator import OllamaTaskBreakdown as LLMTaskBreakdown
from elsciRL.instruction_following.LLM_instr_planner.LLM_instr_validator import LLMInstructionValidator

# Analysis
import matplotlib
matplotlib.use('Agg')
from elsciRL.analysis.combined_variance_visual import combined_variance_analysis_graph as COMBINED_VARIANCE_ANALYSIS_GRAPH

# LLM API Setup
# - only import if user has selected to use LLM
#from elsciRL.GUI.LLM_utils import generate_application

dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, static_folder=os.path.join(dir_path, 'static'), 
            template_folder=os.path.join(dir_path, 'templates'))


class WebApp:
    def __init__(self, save_dir: str = './elsciRL-App-output', num_explor_epi: int = 1000):
        self.global_save_dir = save_dir
        self.num_explor_epi = num_explor_epi
        imports = Applications().data
        possible_applications = list(imports.keys())
        self.available_applications = possible_applications
        self.pull_app_data = None  # Will be set when load_data is called
        

        # Data used for LLM prompt
        with open(os.path.join(app.static_folder, 'app_setup.md'), "r") as f:
            self.app_setup_info = f.read()

        # Initialize LLM Instruction Planner
        self.LLM_INSTUCTION_PLANNER = False
        self.LLM_validation = None

        self.AGENT_PARAMETER_DEFINITIONS = {
            "Qlearntab": {
                "display_name": "Q-Learning",
                "params": {
                    "alpha": {"label": "Learning Rate (Alpha)", "type": "number", "min": 0, "max": 1, "step": 0.01, "default": 0.1},
                    "gamma": {"label": "Discount Factor (Gamma)", "type": "number", "min": 0, "max": 1, "step": 0.01, "default": 0.95},
                    "epsilon": {"label": "Exploration Rate (Epsilon)", "type": "number", "min": 0, "max": 1, "step": 0.01, "default": 0.2},
                    "epsilon_step": {"label": "Epsilon Step", "type": "number", "min": 0, "max": 1, "step": 0.001, "default": 0.01},
                }
            },
            "DQN": {
                "display_name": "Deep Q-Network",
                "params": {
                    "hidden_size": {"label": "Hidden Layer Size", "type": "number", "min": 32, "step": 32, "default": 128},
                    "learning_rate": {"label": "Learning Rate", "type": "number", "min": 0, "max": 1, "step": 0.001, "default": 0.001},
                    "gamma": {"label": "Discount Factor (Gamma)", "type": "number", "min": 0, "max": 1, "step": 0.01, "default": 0.99},
                    "epsilon": {"label": "Initial Exploration Rate", "type": "number", "min": 0, "max": 1, "step": 0.01, "default": 1.0},
                    "epsilon_min": {"label": "Minimum Exploration Rate", "type": "number", "min": 0, "max": 1, "step": 0.01, "default": 0},
                    "epsilon_decay": {"label": "Epsilon Decay Rate", "type": "number", "min": 0, "max": 1, "step": 0.001, "default": 0.995},
                    "memory_size": {"label": "Replay Memory Size", "type": "number", "min": 1000, "step": 1000, "default": 10000},
                    "batch_size": {"label": "Batch Size", "type": "number", "min": 1, "step": 1, "default": 64},
                    "target_update": {"label": "Target Network Update Frequency", "type": "number", "min": 1, "step": 1, "default": 10},
                    #"output_size": {"label": "Output Size", "type": "number", "min": 1, "step": 1, "default": 1},

                }
            },
            "LLM_Ollama": {
                "display_name": "LLM Ollama",
                "params": {
                    "epsilon": {"label": "Epsilon", "type": "number", "min": 0, "max": 1, "step": 0.01, "default": 0.2},
                    "model_name": {"label": "Model Name", "type": "text", "default": "qwen3:0.6b"},
                    "context_length":{"label": "Context Length", "type": "text", "default":"1000"},
                    "system_prompt": {"label": "System Prompt", "type": "textarea", "rows": 4, "placeholder": "Enter system prompt...", "default": ""},
                    "previous_state_action_history_length": {"label": "Previous State Action History Length", "type": "number", "min": 1, "step": 1, "default": 10},
                    "action_language_mapping": {"label": "Action Mapping", "type": "checkbox", "default": False}
                }
            },
            # Add other agents like DQN, SB3_DQN, SB3_PPO, SB3_A2C here if they are re-enabled
            # For example:
            # "DQN": {
            #     "display_name": "Deep Q-Network",
            #     "params": {
            #         "input_size": {"label": "Input Size", "type": "number", "min": 1, "default": 64},
            #         "sent_hidden_dim": {"label": "Sentence Hidden Dimension", "type": "number", "min": 1, "default": 32},
            #         # ... other DQN params
            #     }
            # },
        }

        self.active_jobs = {}  # Stores job_id: {'queue': Queue, 'thread': Thread, 'status': str}
        self.job_results = {} # Stores job_id: results payload

    def load_data(self):
        # Currently pulls all the available applications
        # - Moved to load_data so that it doesnt load on import
        if self.pull_app_data is None:
            from elsciRL.application_suite.import_tool import PullApplications
            self.application_data = PullApplications()
            self.pull_app_data = self.application_data.pull(problem_selection=self.available_applications)
            self.config = self.application_data.setup()

        # Init data here so it reset when page is reloaded
        self.global_input_count = 0
        self.user_feedback_count = 0
        self.instruction_results = {}
        self.instruction_results_validated = {}
        self.correct_instructions = []
        self.incorrect_instructions = []
        self.user_input = {}

        if not os.path.exists('./elsciRL-App-output'):
            os.mkdir('./elsciRL-App-output')
        if 'results' not in self.global_save_dir:
            time_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
            save_dir = './elsciRL-App-output/' + str('results') + '_' + time_str
            if not os.path.exists(save_dir):                
                os.mkdir(save_dir)
            self.global_save_dir = save_dir

        if not os.path.exists(self.global_save_dir+'/uploads'):                
            os.mkdir(self.global_save_dir+'/uploads')
        print("GLOBAL SAVE DIR: ", self.global_save_dir)
        self.uploads_dir = os.path.abspath(os.path.join(self.global_save_dir, 'uploads'))
        if not os.path.exists(self.uploads_dir):
            os.makedirs(self.uploads_dir, exist_ok=True)
        print(f"Uploads directory (absolute path): {self.uploads_dir}")

    def home(self):
        template_path = os.path.join(app.template_folder, 'index.html')
        print(f"Trying to get HTML file from: {template_path}")
        return render_template('index.html', agent_parameter_definitions=self.AGENT_PARAMETER_DEFINITIONS)

    def get_applications(self):
        return jsonify({
            'applications': self.available_applications
        })
    
    def get_adapters(self, selected_application: str = ''):
        if selected_application == '':
            return []
        
        try:
            adapters = list(self.pull_app_data[selected_application]['adapters'].keys())
            return adapters
        except:
            print("Error fetching adapters...")
            return []
    
    def get_available_instructions(self, selected_application: str = ''):
        if selected_application == '':
            return []
        
        try:
            instructions = list(self.pull_app_data[selected_application]['instructions'].keys())
            return instructions
        except:
            print("Error fetching instructions...")
            return []
    
    def get_instruction_data(self, selected_application: str = '', instruction_name: str = ''):
        if selected_application == '' or instruction_name == '':
            return None
        
        try:
            instruction_data = self.pull_app_data[selected_application]['instructions'][instruction_name]
            self.instruction_results_validated[selected_application] = instruction_data
            print(f"Instruction data fetched for {instruction_name}")
            print(f"Instruction data: {self.instruction_results_validated}")
            return instruction_data
        except:
            print(f"Error fetching instruction data for {instruction_name}...")
            return None
    
    def get_observed_states(self, selected_application):
        observed_states = list(self.pull_app_data[selected_application[0]]['prerender_data'].keys())
        return observed_states

    def get_local_configs(self, selected_application:str=''):
        if selected_application == '':
            return []
            
        try:
            local_configs = list(self.pull_app_data[selected_application]['local_configs'].keys())
            return local_configs
        except:
            print("Application data not found...")
            return []

    def get_plot_options(self, selected_application: str = ''):
        if selected_application == '':
            return []
        
        try:
            plot_options = list(self.pull_app_data[selected_application]['local_analysis'].keys())
            return plot_options
        except:
            print("Error fetching plot options...")
            return []

    def get_all_options(self):
        all_local_configs = {}
        all_observed_states = {}
        all_plot_options = {}
        all_experiment_configs = {}
        for app_iter in self.available_applications:
            all_observed_states[app_iter] = self.get_observed_states([app_iter])
            all_local_configs[app_iter] = self.get_local_configs(app_iter)
            all_plot_options[app_iter] = self.get_plot_options(app_iter)
            try:
                all_experiment_configs[app_iter] = list(self.pull_app_data[app_iter]['experiment_configs'].keys())
            except Exception as e:
                print(f"Error fetching experiment configs for {app_iter}: {e}")
                all_experiment_configs[app_iter] = None
        return jsonify({
            'localConfigs': all_local_configs,
            'observedStates': all_observed_states,
            'plotOptions': all_plot_options,
            'experimentConfigs': all_experiment_configs
        })
    
    def generate_application(self, user_input:str=''):
        # TODO: Use this in a new tab with user input to update application list
        # Load the app_setup.md content as part of the system prompt
        
        # Add requirement to system prompt for code chunk separation
        
        return jsonify({"reply": None})

    def process_input(self):
        print("Processing input...")
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided'}), 400

        user_input = data.get('userInput', '')
        application = data.get('selectedApps', [])[0]
        config_input = data.get('localConfigInput', '')
        observed_states_filename = data.get('observedStateInput', '')
        self.observed_states_filename = observed_states_filename
        enable_llm_planner = data.get('enableLLMPlanner', False)
        # Get user LLM settings from input
        llm_model_name = data.get('llmModelSelect', data.get('LLMModelName', 'llama3.2'))
        llm_context_length = int(data.get('LLMContextLength', 1000))
        llm_num_instructions = int(data.get('llmNumInstructions', 1))

        # --- Add Problem info to input prompt ---
        input_prompt = 'Reinforcement Learning experiment for application: ' + application + '\n'
        
        try:
            source = self.pull_app_data[application]['source']
            root_url = list(source.keys())[0]
            engine_folder = source[root_url]['engine_folder']
            engine_filename = source[root_url]['engine_filename']

            # Get the engine source code as text from the URL
            engine_url = f"{root_url.rstrip('/')}/{engine_folder.strip('/')}/{engine_filename}.py"
            response = requests.get(engine_url)
            engine_source_code = response.text

            input_prompt += f"\n Engine source code:\n{engine_source_code}...\n" 

            # Get the adapter source code
            adapter_folder = source[root_url]['local_adapter_folder']
            adapter_filename = source[root_url]['adapter_filenames']
            for adapter_name, adapter_file in adapter_filename.items():
                adapter_url = f"{root_url.rstrip('/')}/{adapter_folder.strip('/')}/{adapter_file}.py"
                response = requests.get(adapter_url)
                if response.status_code == 200:
                    input_prompt += f"\n Adapter source code ({adapter_name}):\n{response.text}...\n"
                else:
                    print(f"Error fetching adapter source code from {adapter_url}: {response.status_code}")

        except Exception as e:
            print(f"Error fetching engine source code from {root_url}: {e}")
        # ---
        instruction_reflection = True
        # Set LLM Instruction Planner state
        self.LLM_INSTUCTION_PLANNER = enable_llm_planner
        if self.LLM_INSTUCTION_PLANNER:
            try:
                if observed_states_filename == '':
                    observed_states_data = None
                else:
                    observed_states_data = self.pull_app_data[application]['prerender_data'][observed_states_filename]
                self.LLM_plan_generator = LLMTaskBreakdown(model_name=llm_model_name,
                                                           context_length=llm_context_length,
                                                           input_prompt = input_prompt,
                                                           observed_states=observed_states_data)
                self.LLM_validation = LLMInstructionValidator()
                if instruction_reflection:
                    reflection_prompt = f"""Update the instruction to increase the similarity to the environment's language structure.
The environment cannot be changed so you must only try to edit the instruction, you will be given an example of the environment's language to align to.
Here is information about the environment and the task: {input_prompt}
"""
                    self.LLM_reflection_plan_generator = LLMTaskBreakdown(model_name=llm_model_name,
                                                           context_length=llm_context_length,
                                                           input_prompt = reflection_prompt,
                                                           observed_states=observed_states_data)
            except Exception as e:
                print(f"Error initializing LLM validator: {e}")
                self.LLM_INSTUCTION_PLANNER = False
        

        if not application:
            return jsonify({'error': 'No application selected'}), 400

        instruction_descriptions = user_input.split('\n')
        instructions = [f'{i}' for i in range(0, len(instruction_descriptions))]

        # Use LLM plan generator if enabled
        if self.LLM_INSTUCTION_PLANNER and self.LLM_plan_generator is not None:
            try:
                # Use LLM to breakdown the user input into structured instructions
                llm_breakdown = self.LLM_plan_generator.break_down_task(user_input, max_subgoals=llm_num_instructions)
                if llm_breakdown and isinstance(llm_breakdown, list) and len(llm_breakdown) > 0:
                    instruction_descriptions = llm_breakdown
                    instructions = [f'{i}' for i in range(0, len(instruction_descriptions))]
                    print(f"LLM breakdown generated {len(instruction_descriptions)} instructions out of a possible {llm_num_instructions}")
                else:
                    print("LLM breakdown failed or returned empty, using original input")
            except Exception as e:
                print(f"Error using LLM plan generator: {e}")
                print("Falling back to original input splitting")

        results = {}
        console_output = ''
        match_plots = []

        # Update search number of episodes ONLY for instruction match
        training_episodes = data.get('trainingEpisodes', 1000)
        
        self.config.update({
            'problem_type': data.get('problemType', 'Default'),
            'number_training_episodes': int(training_episodes),
        })

        # Use default config for search agent
        self.ExperimentConfig = self.config.copy()

        engine = self.pull_app_data[application]['engine']
        local_config = self.pull_app_data[application]['local_configs'][config_input]
        adapters = self.pull_app_data[application]['adapters']
        # Instruction search uses the observed input for the the adapter
        self.ExperimentConfig["adapter_select"] = [observed_states_filename.split('-')[0]]

        if len(self.pull_app_data[application]['prerender_data'].keys()) > 0:
            print("Pre-rendered data found...")
            observed_states = self.pull_app_data[application]['prerender_data'][observed_states_filename]
            observed_states_encoded = self.pull_app_data[application]['prerender_data_encoded'][observed_states_filename] if observed_states_filename in self.pull_app_data[application]['prerender_data_encoded'] else None
            self.elsci_run = elsci_search(Config=self.ExperimentConfig,
                                          LocalConfig=local_config,
                                          Engine=engine, Adapters=adapters,
                                          save_dir=self.global_save_dir,
                                          number_exploration_episodes=self.num_explor_epi,
                                          match_sim_threshold=0.9,
                                          observed_states=observed_states,
                                          observed_states_encoded=observed_states_encoded,
                                          context_length=1000)
        else:
            print("No pre-rendered data found...")
            self.elsci_run = elsci_search(Config=self.ExperimentConfig,
                                          LocalConfig=local_config,
                                          Engine=engine, Adapters=adapters,
                                          save_dir=self.global_save_dir,
                                          number_exploration_episodes=self.num_explor_epi,
                                          match_sim_threshold=0.9,
                                          observed_states=None,
                                          observed_states_encoded=None,
                                          context_length=1000)
            observed_states = self.elsci_run.search()
            with open(os.path.join(self.uploads_dir, 'observed_states.txt'), 'w') as f:
                json.dump(observed_states, f)

        print("\n ====================================================================================================================")
        print("Matching instructions...")
        best_match_dict, instruction_results_data = self.elsci_run.match(
            instructions=instructions,
            instr_descriptions=instruction_descriptions
        )
        results[application] = best_match_dict.copy()
        # Match is completed then checked by LLM method
        # If LLM method not selected then will only complete the match and allow user to confirm
        if self.LLM_INSTUCTION_PLANNER:
            self.validated_LLM_instructions = [] 
            instruction_descriptions_reflection = []
            instruction_descriptions_current = instruction_descriptions
            feedback_counter = [0 for _ in range(0, len(instruction_descriptions_current))]
            feedback_limit = 10 # Currently applies all instr, not per instr limit
            while (len(self.validated_LLM_instructions) == 0) or (False in self.validated_LLM_instructions):            
                self.validated_LLM_instructions = [] # Reset validated instructions for next search
                print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                for n, instr in enumerate(list(results[application].keys())):
                    print("---")
                    print("\t - ",instruction_descriptions_current)
                    print("---")
                    # LLM Validation checks current instructions to confirm they are complete
                    validation_result = self.LLM_validation.validate_instruction_completion(
                        instruction_descriptions_current[n], results[application][instr]['sub_goal']
                        )
                    instr_complete = validation_result['is_complete']
                    confidence = validation_result['confidence']
                    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
                    print(f"Validation Result for instruction {n+1}:")
                    print(f"\t - {validation_result}")
                    print(f"INSTRUCTION: {instruction_descriptions_current[n]}")
                    print(f"|---> INSTRUCTION COMPLETE:  {instr_complete}")
                    print(f"|---> CONFIDENCE: {confidence}")
                    print(f"|---> REASONING: {validation_result['reasoning']}")
                    print(f"|---> BEST MATCH: {results[application][instr]['sub_goal']}")
                        
                    # TYPE CHECKING
                    if isinstance(instr_complete, str):
                        instr_complete = instr_complete.strip().lower() == 'true'
                    if isinstance(confidence, str):
                        try:
                            confidence = float(confidence)
                        except ValueError:
                            print(f"Invalid confidence value: {confidence}, defaulting to 0.75")
                            confidence = 0.75

                    # Check if instruction is complete
                    if instr_complete:
                        feedback_update = self.elsci_run.feedback(feedback_type='positive', feedback_increment=0.01, plot=False)
                        self.validated_LLM_instructions.append(True)
                    else:
                        feedback_update = self.elsci_run.feedback(feedback_type='negative', feedback_increment=0.001, plot=False)
                        self.validated_LLM_instructions.append(False)
              
                    

                    # Provide reflection prompt back to matching algorithm to let it try again
                    if instruction_reflection:
                        if not instr_complete:
                            instr_reflection = f"""Update the instruction {instruction_descriptions_current[n]} as a match could not be found in the environment.
The reason given was: {validation_result['reasoning']} 
Example of environment language structure: {results[application][instr]['sub_goal']}"""
                            print("\n --------------------------------")
                            print("LLM REFLECTION ENEABLED - UPDATING INSTRUCTIONS")
                            print(instr_reflection)
                            print("--------------------------------")
                            llm_breakdown = self.LLM_reflection_plan_generator.break_down_task(instr_reflection, max_subgoals=1)
                            if llm_breakdown and isinstance(llm_breakdown, list) and len(llm_breakdown) > 0:
                                instruction_descriptions_reflection.append(llm_breakdown[0]) # Only one instruction returned
                                print(f"LLM reflection enabled as instruction \n |---> {instruction_descriptions_current[n]} \nis not complete, updated instruction: \n |---> {llm_breakdown}")
                                print("--------------------------------")
                            else:
                                print("LLM breakdown failed or returned empty, using original input")
                                instruction_descriptions_reflection.append(instruction_descriptions_current[n])
                        else:
                            # Dont need to reflect if instruction is complete
                            instruction_descriptions_reflection.append(instruction_descriptions_current[n])
                    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
                    # Exit if feedback is complete or converged
                    if feedback_counter[n] >= feedback_limit:
                        print("LLM validation failed after 10 attempts, assuming instructions as valid")
                        self.validated_LLM_instructions.append(True)
                        break
                    feedback_counter[n] += 1 

                # Provide reflection back to matching algorithm to let it try again
                if len(instruction_descriptions_reflection) > 0:
                    # Check if the instruction is complete by matching the best match
                    best_match_dict, instruction_results_data = self.elsci_run.match(
                        instructions=instructions,
                        instr_descriptions=instruction_descriptions_reflection
                    )
                # Update results                    
                results[application] = best_match_dict.copy()
                
                if abs(feedback_update)<=0.01:
                    print("Feedback update has converged to small update, assuming instructions as valid")
                    self.validated_LLM_instructions.append(True)
                    break
                # Update current instructions to the reflection instructions for next validation
                instruction_descriptions_current = instruction_descriptions_reflection.copy()
                instruction_descriptions_reflection = [] # Reset reflection instructions for current search
                print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        
            for n, instr in enumerate(list(results[application].keys())):
                results[application][instr]['feedback_count'] = feedback_counter[n]
            
        # Store validated instructions
        if application not in self.instruction_results:
            self.instruction_results[application] = {}
        if self.LLM_INSTUCTION_PLANNER:
            self.instruction_results[application]['LLM_instr_' + str(self.global_input_count)] = instruction_results_data
            for instr_type in self.instruction_results[application].keys():
                for n,instr in enumerate(list(self.instruction_results[application][instr_type].keys())):
                    self.instruction_results[application][instr_type][instr]['feedback_count'] = feedback_counter[n]
            
            if len(instruction_descriptions_reflection) > 0:
                # If LLM reflection is enabled and instructions are not complete, use the reflection instructions
                instruction_descriptions = instruction_descriptions_reflection
                instructions = [f'{i}' for i in range(0, len(instruction_descriptions))]
                print(f"LLM reflection enabled as initial instructions do not complete, using updated reflection instructions: {instruction_descriptions}")
        else:
            self.instruction_results[application]['instr_' + str(self.global_input_count)] = instruction_results_data   
           
        # Add base user input to the results
        if application not in self.user_input:
            self.user_input[application] = {}
        for instr_type in self.instruction_results[application].keys():
            if instr_type not in self.user_input[application]:
                self.user_input[application][instr_type] = {}
            self.user_input[application][instr_type]['user_input'] = user_input

        console_output += f'<br><b>Results for {application}:</b><br>'
        for n, instr in enumerate(list(results[application].keys())):
            if results[application][instr] is None:
                console_output += '<b>' + str(n + 1) + ' - ' + instruction_descriptions[n] + ':</b> <i>No match found</i><br>'
            else:
                console_output += '<b>' + str(n + 1) + ' - ' + instruction_descriptions[n] + ':</b> <i>' + results[application][instr]['sub_goal'] + '</i><br>'

                engine_dummy = engine(local_config)
                engine_dummy.reset() # Reset required by gym environments
                
                instr_match_plot = engine_dummy.render(results[application][instr]['best_match'])
                instr_match_plot_filename = f'current_state_plot_{n}.png'
                instr_match_plot_path = os.path.abspath(os.path.join(self.uploads_dir, instr_match_plot_filename))
                instr_match_plot.savefig(instr_match_plot_path)
        
                if os.path.exists(instr_match_plot_path):
                    print(f"Current state plot created successfully at {instr_match_plot_path}")
                    print(f"File size: {os.path.getsize(instr_match_plot_path)} bytes")
                    match_plots.append(f'uploads/{instr_match_plot_filename}')
                else:
                    print(f"Error: Current state plot not created at {instr_match_plot_path}")

        prerender_image = self.get_prerender_image(application)

        response_data = {
            'console_output': console_output,
            'matchPlots': match_plots,
            'prerenderImage': prerender_image
        }
        
        # Include LLM validation result if LLM planner is enabled
        if self.LLM_INSTUCTION_PLANNER:
            response_data['llm_validation_result'] = self.validated_LLM_instructions

        return jsonify(response_data)

    def get_prerender_image(self, application):
        image_data = self.pull_app_data[application]['prerender_images']
        image_paths = []
        if image_data:
            for image_name, image_content in image_data.items():
                image_path = os.path.join(self.uploads_dir, image_name)
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_content)
                image_paths.append(f'uploads/{image_name}')
        return image_paths
        

    def _perform_training_async(self, job_id, data):
        job_queue = self.active_jobs[job_id]['queue']
        self.active_jobs[job_id]['status'] = 'running'
        figures_to_display = []

        try:
            application = data.get('selectedApps', [])[0]
            config_input = data.get('localConfigInput', '')
            selected_plot = data.get('selectedPlot', '')
            
            job_queue.put(f"EVENT: Starting training for application: {application}")

            if len(self.instruction_results_validated) > 0:
                if application not in self.instruction_results_validated:
                    error_msg = f"No instruction results found for {application}"
                    job_queue.put(f"ERROR: {error_msg}")
                    self.job_results[job_id] = {'figures': [], 'error': error_msg, 'status': 'failed'}
                    self.active_jobs[job_id]['status'] = 'failed'
                    job_queue.put("EVENT: RENDER_PHASE_TITLE: Experiment Failed")
                    job_queue.put("EVENT: JOB_FAILED")
                    return
            
            engine_class = self.pull_app_data[application]['engine']
            local_config = self.pull_app_data[application]['local_configs'][config_input]
            adapters = self.pull_app_data[application]['adapters']

            ExperimentConfig = self.config.copy()
            experimentConfigSelect = data.get('experimentConfigSelect', '')
            if (experimentConfigSelect is not None) and (experimentConfigSelect != ''):
                current_app_for_config = application 
                if current_app_for_config in self.pull_app_data and \
                   'experiment_configs' in self.pull_app_data[current_app_for_config] and \
                   experimentConfigSelect in self.pull_app_data[current_app_for_config]['experiment_configs']:
                    selected_config_data = self.pull_app_data[current_app_for_config]['experiment_configs'][experimentConfigSelect]
                    if selected_config_data is not None:
                        ExperimentConfig = selected_config_data.copy()
                        job_queue.put(f"EVENT: Loaded experiment config: {experimentConfigSelect}")
                    else:
                        job_queue.put(f"WARNING: Experiment config '{experimentConfigSelect}' is None. Using default.")
                else:
                    job_queue.put(f"WARNING: Experiment config '{experimentConfigSelect}' not found for app '{current_app_for_config}'. Using default.")
            else:
                job_queue.put("EVENT: No specific experiment config selected. Using default.")

            selected_agents = data.get('selectedAgents', ['Qlearntab'])
            training_episodes = data.get('trainingEpisodes', 1000)
            training_repeats = data.get('trainingRepeats', 5)
            # training_seeds = data.get('trainingSeeds', 1)
            training_seeds = 1
            test_episodes = data.get('testEpisodes', 200)
            test_repeats = data.get('testRepeats', 10)

            ExperimentConfig.update({
                'problem_type': data.get('problemType', 'Default'),
                'number_training_episodes': int(training_episodes),
                'number_training_repeats': int(training_repeats),
                'number_training_seeds': int(training_seeds),
                'number_test_episodes': int(test_episodes),
                'number_test_repeats': int(test_repeats),
                'agent_select': selected_agents,
                'agent_parameters': {}
            })
            job_queue.put(f"EVENT: ExperimentConfig updated. Agents: {selected_agents}")

            for agent_id, agent_config_def in self.AGENT_PARAMETER_DEFINITIONS.items():
                if agent_id in selected_agents:
                    ExperimentConfig['agent_parameters'][agent_id] = {}
                    for param_key, param_config in agent_config_def['params'].items():
                        form_field_name = f"{agent_id}_{param_key}"
                        value = data.get(form_field_name)
                        if value is not None:
                            if param_config['type'] == 'number':
                                try:
                                    float_val = float(value)
                                    ExperimentConfig['agent_parameters'][agent_id][param_key] = int(float_val) if float_val.is_integer() else float_val
                                except ValueError:
                                    job_queue.put(f"WARNING: Could not convert {form_field_name} value '{value}' to number. Using default: {param_config['default']}")
                                    ExperimentConfig['agent_parameters'][agent_id][param_key] = param_config['default']
                            else:
                                ExperimentConfig['agent_parameters'][agent_id][param_key] = str(value)
                        else:
                            job_queue.put(f"WARNING: Param {form_field_name} not found. Using default: {param_config['default']}")
                            ExperimentConfig['agent_parameters'][agent_id][param_key] = param_config['default']
            
            # Add user selected LLM adapter model to be called by adapters
            adapter_LLM_model = data.get('llmAdapterModelSelect', 'qwen3:0.6b').lower()
            local_config['model_name'] = adapter_LLM_model

            if 'LLM_Ollama' in selected_agents:
                LLM_Ollama_action_mapping = data.get('LLM_Ollama_action_language_mapping', False)
                if 'LLM_Ollama' not in ExperimentConfig['agent_parameters']:
                    ExperimentConfig['agent_parameters']['LLM_Ollama'] = {}
                ExperimentConfig['agent_parameters']['LLM_Ollama'].update({
                    "epsilon": float(data.get('LLM_Ollama_epsilon', 0.2)),
                    'model_name': str(data.get('LLM_Ollama_model_name', 'llama3.2')).lower(),
                    'context_length': int(data.get('LLM_Ollama_context_length', '1000')),
                    'system_prompt': str(data.get('LLM_Ollama_system_prompt', '')),
                    'previous_state_action_history_length': int(data.get('LLM_Ollama_previous_state_action_history_length', 10)),
                    'action_language_mapping': LLM_Ollama_action_mapping
                })

                if LLM_Ollama_action_mapping:
                    # Add engine source code to Agent's system prompt
                    try:
                        source = self.pull_app_data[application]['source']
                        root_url = list(source.keys())[0]
                        engine_folder = source[root_url]['engine_folder']
                        engine_filename = source[root_url]['engine_filename']

                        # Get the engine source code as text from the URL
                        engine_url = f"{root_url.rstrip('/')}/{engine_folder.strip('/')}/{engine_filename}.py"
                        response = requests.get(engine_url)
                        engine_source_code = response.text
                        ExperimentConfig['agent_parameters']['LLM_Ollama']['system_prompt'] += f"\n Engine source code:\n{engine_source_code}...\n" 
                    except Exception as e:
                        print(f"Error fetching engine source code from {root_url}: {e}")

            # Adapters are now selected per agent and output into agent_adapter_dict, so if none then use all adapters
            agent_adapter_dict = data.get('agent_adapter_dict', {})
            if not agent_adapter_dict:
                # Fallback to all adapters if new format not provided
                all_adapters = list(adapters.keys())
                agent_adapter_dict = {agent_name: list(all_adapters) for agent_name in selected_agents} if selected_agents else {}            
            ExperimentConfig['adapter_input_dict'] = agent_adapter_dict
            print("\n --- AGENT ADAPTER SELECTION ---")
            print(agent_adapter_dict)
            print("-------------------------------\n")
            job_queue.put(f"EVENT: Adapter dictionary set up: {agent_adapter_dict}")

            instruction_results_map = self.instruction_results_validated.get(application, {})
            if not instruction_results_map:
                job_queue.put(f"INFO: No validated instructions for {application}. Standard RL run only.")
            
            app_save_dir = os.path.join(self.global_save_dir, application)
            if not os.path.exists(app_save_dir):
                os.makedirs(app_save_dir)
                job_queue.put(f"EVENT: Created app save directory: {app_save_dir}")
            
            if instruction_results_map:
                job_queue.put("EVENT: Training with instructions...")
                if 'feedback_plot' in instruction_results_map:
                    job_queue.put(f"EVENT: Feedback plot found for {application}. Adding to results.")
                    figures_to_display.append(self.instruction_results_validated[application]['feedback_plot'].split('/')[-1])
                    # Remove feedback plot before using rest of results for training process
                    self.instruction_results_validated[application].pop('feedback_plot', None)
                # ------------------------
                # Save Insruction Results
                # Feedback layer and sim score is always Tensor so cannot be saved as JSON
                instructions_results_save = self.instruction_results_validated[application].copy()
                ignore_data = ["user_input","instr_description", "feedback_count", "feedback_plot", "user_feedback_count"]
                for instr_key,instr_dict in instructions_results_save.items():
                    if instr_key not in ignore_data:
                        # Add user feedback count to instruction results
                        if 'user_feedback_count' not in instr_dict:   
                            instr_dict['user_feedback_count'] = self.user_feedback_count 
                        # ---
                        for sub_instr_key, sub_instr_dict in instr_dict.items():
                            if sub_instr_key not in ignore_data:
                                for a_a_key, instr_agent_adapter_dict in sub_instr_dict.items():
                                    if a_a_key not in ignore_data:
                                        # Convert data to list if it exists
                                        if 'feedback_layer' in instr_agent_adapter_dict and isinstance(instr_agent_adapter_dict['feedback_layer'], torch.Tensor):
                                            job_queue.put(f"EVENT: Converting feedback layer for {a_a_key} to list for JSON serialization.")
                                            instr_agent_adapter_dict['feedback_layer'] = instr_agent_adapter_dict['feedback_layer'].cpu().numpy().tolist()
                                        if 'sub_goal' in instr_agent_adapter_dict and isinstance(instr_agent_adapter_dict['sub_goal'], torch.Tensor):
                                            job_queue.put(f"EVENT: Converting sub_goal for {a_a_key} to list for JSON serialization.")
                                            instr_agent_adapter_dict['sub_goal'] = instr_agent_adapter_dict['sub_goal'].cpu().numpy().tolist()
                                        if 'sim_score' in instr_agent_adapter_dict and isinstance(instr_agent_adapter_dict['sim_score'], torch.Tensor):
                                            job_queue.put(f"EVENT: Converting sim_score for {a_a_key} to list for JSON serialization.")
                                            instr_agent_adapter_dict['sim_score'] = instr_agent_adapter_dict['sim_score'].cpu().numpy().tolist()                     
                    instr_dict['user_input'] = self.user_input[application][instr_key]['user_input'] if instr_key in self.user_input[application] else ''
                # ---         
                if not os.path.exists(self.uploads_dir):
                    os.makedirs(self.uploads_dir, exist_ok=True)
                instr_results_path = os.path.join(app_save_dir, f'instruction_results_{application}_{self.observed_states_filename}.json')  
                #print(f"Saving instruction results: {instructions_results_save}")
                with open(instr_results_path, 'w') as f:
                    json.dump(instructions_results_save, f, indent=4)                
                job_queue.put(f"EVENT: Saved instruction results to {instr_results_path}")
                # ------------------------
                # Instruction keys can skip numbers which miss-aligns lookup of correct instructions from list
                # TODO: Make self.correct_instructions a dictionary with keys as instr_key and values as instruction text
                instr_key_lookup = {}
                idx_fix = 0
                for instr_key, instr_data_path in instruction_results_map.items():
                    if 'instr_' in instr_key:
                        instr_key_lookup[instr_key] = idx_fix
                        idx_fix+=1
                # ---
                for instr_key, instr_data_path in instruction_results_map.items():
                    instr_text = "Instruction (details unavailable)"
                    try:
                        instr_idx = instr_key_lookup[instr_key]
                        if 0 <= instr_idx < len(self.correct_instructions):
                            instr_text = self.correct_instructions[instr_idx]
                            instr_text = (instr_text[:75] + '...') if len(instr_text) > 75 else instr_text
                    except ValueError:
                        job_queue.put(f"WARNING: Could not parse index from instr_key: {instr_key}")
                    
                    job_queue.put(f"EVENT: RENDER_PHASE_TITLE: Instruction: {instr_text}")
                    job_queue.put(f"EVENT: Processing instruction key: {instr_key}")
                    instr_save_dir = os.path.join(app_save_dir, instr_key)

                    for agent_name in list(agent_adapter_dict.keys()):
                        agent_select_sub = [agent_name] if agent_name in selected_agents else []
                        ExperimentConfig['agent_select'] = agent_select_sub
                        for adapter_name in list(agent_adapter_dict[agent_name]):
                            agent_adapter_dict_sub = {agent_name: [adapter_name]}
                            ExperimentConfig['adapter_input_dict'] = agent_adapter_dict_sub
                            job_queue.put(f"EVENT: Running standard train for {agent_name} with {adapter_name} adapter...")
                            
                            reinforced_experiment = elsciRLOptimize(
                                Config=ExperimentConfig, LocalConfig=local_config, Engine=engine_class, Adapters=adapters,
                                save_dir=instr_save_dir, show_figures='No', window_size=0.1,
                                instruction_path=instr_data_path, predicted_path=None, instruction_episode_ratio=0.1,
                                instruction_chain=True, instruction_chain_how='exact')
                            job_queue.put(f"EVENT: Starting train for {instr_key} ({instr_text})")
                            reinforced_experiment.train()

                            job_queue.put(f"EVENT: Test complete for {instr_key}. Rendering results.")
                            render_result = reinforced_experiment.render_results()
                            
                            # Send real-time figure update for instruction experiments
                            render_results_dir_instr = os.path.join(instr_save_dir, 'Instr_Experiment', 'render_results')
                            if os.path.exists(render_results_dir_instr):
                                for file_item in os.listdir(render_results_dir_instr):
                                    if file_item.endswith('.gif'):
                                        # Copy to uploads directory for web access
                                        dest_filename = f'{instr_key}_{file_item}'
                                        shutil.copyfile(os.path.join(render_results_dir_instr, file_item), os.path.join(self.uploads_dir, dest_filename))
                                        figures_to_display.append(f'uploads/{dest_filename}')
                                        
                                        # Send real-time figure update
                                        figure_event = {
                                            'figure_path': f'uploads/{dest_filename}',
                                            'experiment_type': f'Instruction: {instr_text}',
                                            'filename': dest_filename,
                                            'timestamp': datetime.now().isoformat()
                                        }
                                        job_queue.put(f"EVENT: RENDER_FIGURE: {json.dumps(figure_event)}")

                            job_queue.put(f"EVENT: Train complete for {instr_key}. Starting test.")
                            reinforced_experiment.test()
                    

                if selected_plot:
                    analysis_class = self.pull_app_data[application]['local_analysis'][selected_plot]
                    for instr_key_analysis in instruction_results_map:
                        analysis_instance = analysis_class(save_dir=os.path.join(app_save_dir, instr_key_analysis))
                        for func_name in [f_name for f_name in dir(analysis_instance) if callable(getattr(analysis_instance, f_name)) and not f_name.startswith("__")]:
                            fig_dict = getattr(analysis_instance, func_name)()
                            for fig_name, fig_obj in fig_dict.items():
                                if fig_obj:
                                    fig_filename = f'{application}_{func_name}_{instr_key_analysis}_{fig_name}.png'
                                    fig_obj.savefig(os.path.join(self.uploads_dir, fig_filename))
                                    figures_to_display.append(f'uploads/{fig_filename}')
        
            job_queue.put("EVENT: RENDER_PHASE_TITLE: No Instruction")
            job_queue.put("EVENT: Running standard (no-instruction) experiment...")
            no_instr_save_dir = os.path.join(app_save_dir, 'no-instr')

            for agent_name in list(agent_adapter_dict.keys()):
                agent_select_sub = [agent_name] if agent_name in selected_agents else []
                ExperimentConfig['agent_select'] = agent_select_sub
                for adapter_name in list(agent_adapter_dict[agent_name]):
                    agent_adapter_dict_sub = {agent_name: [adapter_name]}
                    ExperimentConfig['adapter_input_dict'] = agent_adapter_dict_sub
                    job_queue.put(f"EVENT: Running standard train for {agent_name} with {adapter_name} adapter...")
                    standard_experiment = STANDARD_RL(
                        Config=ExperimentConfig, ProblemConfig=local_config, Engine=engine_class, Adapters=adapters,
                        save_dir=no_instr_save_dir, show_figures='No', window_size=0.1)
                    job_queue.put("EVENT: Starting standard train.")
                    standard_experiment.train()

                    job_queue.put("EVENT: Standard test complete. Rendering results.")
                    render_result = standard_experiment.render_results()
                    
                    # Send real-time figure update for standard experiments
                    render_results_dir_std = os.path.join(no_instr_save_dir, 'Standard_Experiment', 'render_results')
                    if os.path.exists(render_results_dir_std):
                        for file_item_std in os.listdir(render_results_dir_std):
                            if file_item_std.endswith('.gif'):
                                # Copy to uploads directory for web access
                                dest_filename = f'no-instr_{file_item_std}'
                                shutil.copyfile(os.path.join(render_results_dir_std, file_item_std), os.path.join(self.uploads_dir, dest_filename))
                                figures_to_display.append(f'uploads/{dest_filename}')
                                
                                # Send real-time figure update
                                figure_event = {
                                    'figure_path': f'uploads/{dest_filename}',
                                    'experiment_type': 'Standard (No Instruction)',
                                    'filename': dest_filename,
                                    'timestamp': datetime.now().isoformat()
                                }
                                job_queue.put(f"EVENT: RENDER_FIGURE: {json.dumps(figure_event)}")

                    job_queue.put("EVENT: Standard train complete. Starting test.")
                    standard_experiment.test()
            
            
            if selected_plot:
                analysis_class_std = self.pull_app_data[application]['local_analysis'][selected_plot]
                analysis_instance_std = analysis_class_std(save_dir=no_instr_save_dir)
                for func_name_std in [f_name for f_name in dir(analysis_instance_std) if callable(getattr(analysis_instance_std, f_name)) and not f_name.startswith("__")]:
                    fig_dict_std = getattr(analysis_instance_std, func_name_std)()
                    for fig_name_std, fig_obj_std in fig_dict_std.items():
                        if fig_obj_std:
                            fig_filename_std = f'{application}_{func_name_std}_no-instr_{fig_name_std}.png'
                            fig_obj_std.savefig(os.path.join(self.uploads_dir, fig_filename_std))
                            figures_to_display.append(f'uploads/{fig_filename_std}')
                        
            job_queue.put("EVENT: Performing combined variance analysis...")
            for eval_type in ['training', 'testing']:
                COMBINED_VARIANCE_ANALYSIS_GRAPH(results_dir=app_save_dir, analysis_type=eval_type, results_to_show='simple')
                src_plot = os.path.join(app_save_dir, f"variance_comparison_{eval_type}.png")
                if os.path.exists(src_plot):
                    dest_filename = f'{application}_variance_analysis_{eval_type}.png'
                    shutil.copyfile(src_plot, os.path.join(self.uploads_dir, dest_filename))
                    figures_to_display.append(f'uploads/{dest_filename}')
        
            self.job_results[job_id] = {'figures': figures_to_display, 'status': 'completed'}
            self.active_jobs[job_id]['status'] = 'completed'
            job_queue.put("EVENT: RENDER_PHASE_TITLE: Experiment Ended, See Results Tab")
            job_queue.put("EVENT: JOB_COMPLETE")

            
        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            job_queue.put(f"ERROR: {error_msg}")
            self.job_results[job_id] = {'figures': figures_to_display, 'error': error_msg, 'status': 'failed'}
            if job_id in self.active_jobs:
                 self.active_jobs[job_id]['status'] = 'failed'
            job_queue.put("EVENT: RENDER_PHASE_TITLE: Experiment Failed")
            job_queue.put("EVENT: JOB_FAILED")

    def train_model(self):
        data = request.json
        job_id = str(uuid.uuid4())
        
        job_queue = queue.Queue()
        thread = threading.Thread(target=self._perform_training_async, args=(job_id, data))
        
        self.active_jobs[job_id] = {'queue': job_queue, 'thread': thread, 'status': 'initializing'}
        self.job_results.pop(job_id, None)
        
        thread.start()
        return jsonify({'job_id': job_id})

    def upload_file(self):
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file.save(os.path.join(app.config.get('UPLOAD_FOLDER', self.uploads_dir), filename))
            return 'File uploaded successfully'

    def new_instruction(self):
        self.global_input_count = len(self.correct_instructions)
        self.user_feedback_count = 0
        return jsonify({'status': 'success'})

    def confirm_result(self):
        data = request.json
        # Check if this is an LLM validation (automatic)
        enable_llm_planner = data.get('enableLLMPlanner', False)        
        is_llm_validation = data.get('isLLMValidation', False)
        
        # Use user input if provided, otherwise use LLM validation
        if enable_llm_planner and is_llm_validation:
            # User is manually confirming LLM validation result
            is_correct = data.get('isCorrect')
            print("++++++++++++++++++++++++++++++++++++++++++++")
            print(f"User manual confirmation of LLM validation: {is_correct}")
            print("++++++++++++++++++++++++++++++++++++++++++++")
        elif enable_llm_planner:
            # Automatic LLM validation (legacy case)
            print("++++++++++++++++++++++++++++++++++++++++++++")
            print(self.validated_LLM_instructions)
            print("++++++++++++++++++++++++++++++++++++++++++++")
            is_correct = self.validated_LLM_instructions
        else:
            # Standard manual validation
            is_correct = data.get('isCorrect')
            
        user_input = data.get('userInput')
        
        application = data.get('selectedApps', [])[0]
        if application not in self.instruction_results_validated:
            self.instruction_results_validated[application] = {}

        if is_correct:
            if application in self.instruction_results:
                self.correct_instructions.append(user_input)
                if 'LLM_instr_'+str(self.global_input_count) in self.instruction_results[application]:
                    self.instruction_results_validated[application]['LLM_instr_'+str(self.global_input_count)] = self.instruction_results[application]['LLM_instr_'+str(self.global_input_count)]
                    if enable_llm_planner and is_llm_validation:
                        message = "<br>User confirmed LLM validation: Training an agent with this guidance to complete the task... <br> See the results tab once training is complete."
                    else:
                        message = "<br>LLM validation confirmed: Training an agent with this guidance to complete the task... <br> See the results tab once training is complete."
                elif 'instr_'+str(self.global_input_count) in self.instruction_results[application]:
                    self.instruction_results_validated[application]['instr_'+str(self.global_input_count)] = self.instruction_results[application]['instr_'+str(self.global_input_count)]
                    message = "<br>Great! Training an agent with this as guidance to complete the task... <br> See the results tab once training is complete."
                else:
                    message = "<br>Error: Could not find instruction data for validation. Please try again."
                    print(f"Error: Could not find instruction data for app {application}, key instr_{self.global_input_count}")
                    return jsonify({'status': 'error', 'message': message}), 500
                # Add feedback to the instruction results
                feedback_plot = self.elsci_run.feedback(feedback_type='positive', feedback_increment=0.1, plot=True, plot_save_dir='uploads')
                self.instruction_results_validated[application]['feedback_plot'] = feedback_plot
            else:
                message = "<br>Error: Original instruction match data not found. Cannot validate."
                print(f"Error: Could not find instruction data for app {application}, key instr_{self.global_input_count}")
                return jsonify({'status': 'error', 'message': message}), 500
            self.global_input_count = len(self.correct_instructions)
        else:
            self.incorrect_instructions.append(user_input)
            if enable_llm_planner and is_llm_validation:
                message = "<br>User disagreed with LLM validation. The model will use this feedback to improve."
            elif enable_llm_planner:
                message = "<br>LLM validation indicates instructions may need refinement. The model will use this feedback to improve."
            else:
                message = "<br>Thanks for the feedback. The model will use this to improve."

            # Add feedback to the instruction results
            if application in self.instruction_results:
                feedback_plot = self.elsci_run.feedback(feedback_type='negative', feedback_increment=0.05, plot=True, plot_save_dir='uploads')
                self.instruction_results_validated[application]['feedback_plot'] = feedback_plot
                self.user_feedback_count += 1
            

        return jsonify({
            'status': 'received',
            'message': message
        })

    def get_correct_instructions(self):
        return jsonify({
            'correctInstructions': self.correct_instructions
        })

    def reset_all_instructions_data(self):
        # Reset the global save directory to a new timestamped directory
        time_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.global_save_dir = './elsciRL-App-output/' + str('results') + '_' + time_str
        if not os.path.exists(self.global_save_dir):                
            os.mkdir(self.global_save_dir)
        self.uploads_dir = os.path.join(self.global_save_dir, 'uploads')
        if not os.path.exists(self.uploads_dir):
            os.makedirs(self.uploads_dir, exist_ok=True)
        
        # Clear the instruction results and validated results
        self.global_input_count = 0
        self.user_feedback_count = 0
        self.instruction_results = {}
        self.instruction_results_validated = {}
        self.correct_instructions = []
        self.incorrect_instructions = []
        # Also clear any files that might have been generated based on these, if necessary.
        # For now, just resetting the state variables.
        print("All instruction-related data has been reset.")
        return jsonify({'status': 'success', 'message': 'All instruction data reset.'})

    def get_experiment_config(self, application, config_name):
        if not application or not config_name:
            return jsonify({'error': 'Missing application or config parameter'}), 400
        
        try:
            experiment_config = self.pull_app_data[application]['experiment_configs'][config_name]
            if experiment_config is None: 
                return jsonify({'error': f'Experiment config "{config_name}" for app "{application}" is null.'}), 404
            config_to_send = experiment_config.copy()
            config_to_send['_agent_definitions_'] = self.AGENT_PARAMETER_DEFINITIONS
            return jsonify({'config': config_to_send})
        except KeyError:
            return jsonify({'error': f'Config "{config_name}" not found for app "{application}".'}), 404
        except Exception as e:
            print(f"Error getting experiment config: {str(e)}")
            return jsonify({'error': 'Failed to get experiment config'}), 500

# ----------------------------------------------------------------

# Initialize WebApp
WebApp_instance = WebApp()

@app.route('/')
def home_route():
    WebApp_instance.load_data()
    app.config['UPLOAD_FOLDER'] = WebApp_instance.uploads_dir
    return WebApp_instance.home()

@app.route('/process_input', methods=['POST'])
def process_input_route():
    response = WebApp_instance.process_input()
    return response

@app.route('/confirm_result', methods=['POST'])
def confirm_result_route():
    return WebApp_instance.confirm_result()

@app.route('/train_model', methods=['POST'])
def train_model_route():
    # Set a new instruction call when training is started as default behavior
    WebApp_instance.new_instruction()
    return WebApp_instance.train_model()

@app.route('/results', methods=['POST'])
def search_route():
    save_dir = request.json.get('save_dir', '')
    return jsonify({'save_dir': save_dir})

@app.route('/upload', methods=['POST'])
def upload_file_route():
    return WebApp_instance.upload_file()

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    try:
        upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(upload_folder, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return f"File not found: {filename}", 404
            
        print(f"Serving file from: {file_path}")
        directory = os.path.dirname(file_path)
        base_filename = os.path.basename(file_path)
        return send_from_directory(directory, base_filename, as_attachment=False)
    except Exception as e:
        print(f"Error serving file {filename}: {str(e)}")
        return f"Error: {str(e)}", 404

@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        print(f"Error serving static file {filename}: {str(e)}")
        return f"Error: {str(e)}", 404

@app.route('/get_applications')
def get_applications_route():
    return WebApp_instance.get_applications()

@app.route('/get_observed_states', methods=['POST'])
def get_observed_states_route():
    data = request.get_json()
    selected_applications = data.get('applications', [])
    observed_states = WebApp_instance.get_observed_states(selected_applications)
    return jsonify({
        'observedStates': observed_states
    })

@app.route('/get_local_configs', methods=['POST'])
def get_local_configs_route():
    data = request.get_json()
    selected_application = data.get('application', [])
    local_configs = WebApp_instance.get_local_configs(selected_application)
    return jsonify({
        'localConfigs': local_configs
    })

@app.route('/get_plot_options', methods=['POST'])
def get_plot_options_route():
    data = request.get_json()
    selected_application = data.get('application', '')
    plot_options = WebApp_instance.get_plot_options(selected_application)
    return jsonify({
        'plotOptions': plot_options
    })

@app.route('/get_all_options')
def get_all_options_route():
    return WebApp_instance.get_all_options()

@app.route('/get_prerender_image', methods=['POST'])
def get_prerender_image_route():
    data = request.get_json()
    application = data.get('application', '')
    if not application:
        return jsonify({'error': 'No application selected'}), 400
    image_paths = WebApp_instance.get_prerender_image(application)
    if image_paths:
        return jsonify({'imagePaths': image_paths})
    return jsonify({'error': 'No prerender images found'}), 404

@app.route('/new_instruction', methods=['POST'])
def new_instruction_route():
    response = WebApp_instance.new_instruction()
    return response

@app.route('/get_correct_instructions')
def get_correct_instructions_route():
    return WebApp_instance.get_correct_instructions()

@app.route('/reset_all_instructions', methods=['POST'])
def reset_all_instructions_route():
    return WebApp_instance.reset_all_instructions_data()

@app.route('/load_data')
def load_data_route():
    WebApp_instance.global_save_dir = ''
    WebApp_instance.load_data()
    return jsonify({'status': 'success'})

@app.route('/get_adapters', methods=['POST'])
def get_adapters_route():
    data = request.get_json()
    selected_application = data.get('application', '')
    adapters = WebApp_instance.get_adapters(selected_application)
    return jsonify({
        'adapters': adapters
    })

@app.route('/get_available_instructions', methods=['POST'])
def get_available_instructions_route():
    data = request.get_json()
    selected_application = data.get('application', '')
    instructions = WebApp_instance.get_available_instructions(selected_application)
    return jsonify({
        'instructions': instructions
    })

@app.route('/get_instruction_data', methods=['POST'])
def get_instruction_data_route():
    data = request.get_json()
    selected_application = data.get('application', '')
    instruction_name = data.get('instruction_name', '')
    instruction_data = WebApp_instance.get_instruction_data(selected_application, instruction_name)
    return jsonify({
        'instruction_data': instruction_data
    })

@app.route('/get_experiment_config', methods=['POST'])
def get_experiment_config_route():
    data = request.get_json()
    application = data.get('application', '')
    config_name_req = data.get('config', '')
    
    return WebApp_instance.get_experiment_config(application, config_name_req)

@app.route('/stream_job_notifications/<job_id>')
def stream_job_notifications_route(job_id):
    if job_id not in WebApp_instance.active_jobs:
        WebApp_instance.job_completed_flag_for_image = True
        def empty_stream_for_old_job():
            yield f"data: ERROR: Job ID {job_id} not found or already cleaned up.\n\n"
            yield f"data: EVENT: JOB_FAILED\n\n"
        return Response(empty_stream_for_old_job(), mimetype='text/event-stream')

    job_queue_ref = WebApp_instance.active_jobs[job_id]['queue']

    def generate_notifications():
        keep_alive_count = 0
        MAX_KEEP_ALIVES_WITHOUT_MESSAGE = 300

        while True:
            try:
                message = job_queue_ref.get(timeout=1)
                yield f"data: {message}\n\n"
                if "JOB_COMPLETE" in message or "JOB_FAILED" in message:
                    if job_id in WebApp_instance.active_jobs:
                         WebApp_instance.active_jobs[job_id]['status'] = 'terminal_notified'
                    break
                keep_alive_count = 0
            except queue.Empty:
                keep_alive_count += 1
                yield ": keepalive\n\n"
                if keep_alive_count > MAX_KEEP_ALIVES_WITHOUT_MESSAGE:
                    if job_id in WebApp_instance.active_jobs and WebApp_instance.active_jobs[job_id]['status'] not in ['completed', 'failed', 'terminal_notified']:
                        WebApp_instance.active_jobs[job_id]['status'] = 'orphaned'
                    break
            except Exception as e:
                print(f"Error in SSE generator for job {job_id}: {e}")
                try:
                    yield f"data: SSE_ERROR: Stream error encountered: {str(e)}\n\n"
                except: pass
                if job_id in WebApp_instance.active_jobs:
                    WebApp_instance.active_jobs[job_id]['status'] = 'stream_error'
                break

    return Response(generate_notifications(), mimetype='text/event-stream')

@app.route('/get_job_status/<job_id>')
def get_job_status_route(job_id):
    if job_id in WebApp_instance.active_jobs:
        status = WebApp_instance.active_jobs[job_id].get('status', 'unknown')
        if job_id in WebApp_instance.job_results and WebApp_instance.job_results[job_id].get('status') in ['completed', 'failed']:
             status = WebApp_instance.job_results[job_id]['status']
        return jsonify({'job_id': job_id, 'status': status})
    elif job_id in WebApp_instance.job_results:
        return jsonify({'job_id': job_id, 'status': WebApp_instance.job_results[job_id].get('status', 'completed_unknown_state')})
    else:
        return jsonify({'job_id': job_id, 'status': 'not_found'}), 404

@app.route('/get_job_results/<job_id>', methods=['GET'])
def get_job_results_route(job_id):
    if job_id in WebApp_instance.job_results:
        return jsonify({'job_id': job_id, 'results': WebApp_instance.job_results[job_id]})
    elif job_id in WebApp_instance.active_jobs and WebApp_instance.active_jobs[job_id].get('status') == 'running':
        return jsonify({'job_id': job_id, 'status': 'running', 'message': 'Job is still processing.'}), 202
    else:
        return jsonify({'error': 'Job results not found or job may have failed without storing final results.'}), 404
    
@app.route('/get_interface_guide')
def get_interface_guide():
    try:
        # Fetch the raw markdown content from GitHub
        response = requests.get('https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/main/App%20Interface%20Guide.md')
        if response.status_code == 200:
            # Convert markdown to HTML
            html_content = markdown(response.text)
            return jsonify({'content': html_content})
        else:
            return jsonify({'error': 'Failed to fetch guide content'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_config', methods=['POST'])
def export_config():
    try:
        data = request.get_json()
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"config_export_{timestamp}.json"
        
        # Save to the output directory
        output_path = os.path.join(WebApp_instance.global_save_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        return jsonify({
            'status': 'success',
            'message': f'Configuration exported to {filename}',
            'filepath': output_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/import_config', methods=['POST'])
def import_config():
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
            
        if not file.filename.endswith('.json'):
            return jsonify({
                'status': 'error',
                'message': 'File must be a JSON file'
            }), 400
            
        # Read and parse the JSON file
        config_data = json.load(file)
        
        return jsonify({
            'status': 'success',
            'config': config_data
        })
    except json.JSONDecodeError:
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON file'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    if not os.path.exists(os.path.join(WebApp_instance.global_save_dir, 'uploads')):
        os.makedirs(os.path.join(WebApp_instance.global_save_dir, 'uploads'))
    
    app.run(debug=True)