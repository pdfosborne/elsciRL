import sys
import os
import shutil
from datetime import datetime

# App tools
from flask import Flask, render_template, request, jsonify, send_from_directory

# elsci methods
from elsciRL.GUI.elsciRL_demo_search import elsciRLSearch as elsci_search
from elsciRL.instruction_following.elsciRL_instruction_following import elsciRLOptimize
from elsciRL.experiments.standard import Experiment as STANDARD_RL
# Get application data
from elsciRL.benchmarking_suite.imports import Applications
from elsciRL.benchmarking_suite.benchmark_tool import PullApplications
# Analysis
import matplotlib
matplotlib.use('Agg')
from elsciRL.analysis.combined_variance_visual import combined_variance_analysis_graph as COMBINED_VARIANCE_ANALYSIS_GRAPH
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, static_folder=os.path.join(dir_path, 'static'), 
            template_folder=os.path.join(dir_path, 'templates'))

imports = Applications().data
possible_applications = list(imports.keys())
pull_data = PullApplications()

class WebApp:
    def __init__(self, save_dir: str = '', num_explor_epi: int = 1000):
        self.global_save_dir = save_dir
        self.num_explor_epi = num_explor_epi
        self.available_applications = possible_applications
        

        # Currently pulls all the available applications
        # - TODO: Make it so it pulls all the file names but not the data
        #    ---> Then once problem is selected it then pulls data
        self.pull_app_data = pull_data.pull(problem_selection=possible_applications)

    def load_data(self):
        # Init data here so it reset when page is reloaded
        self.global_input_count = 0
        self.instruction_results = {}
        self.instruction_results_validated = {}
        self.correct_instructions = []
        self.incorrect_instructions = []

        if not os.path.exists('./elsciRL-App-output'):
            os.mkdir('./elsciRL-App-output')
        if 'search' not in self.global_save_dir:
            time = datetime.now().strftime("%d-%m-%Y_%H-%M")
            save_dir = './elsciRL-App-output/' + str('search') + '_' + time
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

    def train_model(self):
        data = request.json
        application = data.get('selectedApps', [])[0]
        config_input = data.get('localConfigInput', '')
        selected_plot = data.get('selectedPlot', '')
        
        if application not in self.instruction_results_validated:
            print(f"No instruction results found for {application}")
            return jsonify({'error': 'No instruction results found for the selected application'}), 400
        
        engine = self.pull_app_data[application]['engine']
        local_config = self.pull_app_data[application]['local_configs'][config_input]
        adapters = self.pull_app_data[application]['adapters']
        # Use validated instructions for training
        instruction_results = self.instruction_results_validated[application]
        print(instruction_results.keys())
        
        if not os.path.exists(self.global_save_dir+'/'+application):
            os.mkdir(self.global_save_dir+'/'+application)  
        
        # Train for all correctly validated instructions
        flat_agent_run = False
        for instr_key, instr_results in instruction_results.items():
            reinforced_experiment = elsciRLOptimize(
                            Config=self.ExperimentConfig, 
                            LocalConfig=local_config, 
                            Engine=engine, Adapters=adapters,
                            save_dir=self.global_save_dir+'/'+application + '/'+instr_key, 
                            show_figures = 'No', window_size=0.1,
                            instruction_path=instr_results, predicted_path=None, 
                            instruction_episode_ratio=0.1,
                            instruction_chain=True, instruction_chain_how='exact' )
            reinforced_experiment.train()
            reinforced_experiment.test()
            
            # Baseline flat experiment
            # - only ran first time otherwise copied from prior input
            if not flat_agent_run:
                standard_experiment = STANDARD_RL(
                        Config=self.ExperimentConfig, 
                        ProblemConfig=local_config, 
                        Engine=engine, Adapters=adapters,
                        save_dir=self.global_save_dir+'/'+application + '/'+instr_key, 
                        show_figures = 'No', window_size=0.1)
                standard_experiment.train()
                standard_experiment.test()
                flat_agent_run = True
            # else:
            #     previous_input_dir = self.global_save_dir+'/'+application + '/instr_0/Standard_Experiment'
            #     current_input_dir = self.global_save_dir+'/'+application + '/'+instr_key+'/Standard_Experiment'
            #     if not os.path.exists(current_input_dir):
            #         os.makedirs(current_input_dir)
            #     for filename in os.listdir(previous_input_dir):
            #         if not os.path.isdir(os.path.join(previous_input_dir, filename)):
            #             shutil.copy(os.path.join(previous_input_dir, filename), os.path.join(current_input_dir, filename))

        # headers = ["","agent","num_repeats","episode","avg_R_mean","avg_R_se",
        #            "cum_R_mean","cum_R_se","time_mean"]
        
        # if not os.path.exists(self.global_save_dir+'/input_'+str(self.global_input_count)+'/Standard_Experiment'): 
        #     os.mkdir(self.global_save_dir+'/input_'+str(self.global_input_count)+'/Standard_Experiment')

        # with open(self.global_save_dir+'/input_'+str(self.global_input_count)
        #           +'/Standard_Experiment/training_variance_results_Qlearntab_Language.csv', 'w') as csv_file:  
        #     writer = csv.DictWriter(csv_file, fieldnames=headers)
        #     writer.writeheader()
        #     for key, value in training_data.data.items():
        #         writer.writerow(value)
        # with open(self.global_save_dir+'/input_'+str(self.global_input_count)
        #           +'/Standard_Experiment/testing_variance_results_Qlearntab_Language.csv', 'w') as csv_file:  
        #     writer = csv.DictWriter(csv_file, fieldnames=headers)
        #     writer.writeheader()
        #     for key, value in testing_data.data.items():
        #         writer.writerow(value) 


        #         # TODO RUN FOR ALL INSTR
    
        

        figures_to_display = []
        if selected_plot != '':
            analysis_class = self.pull_app_data[application]['local_analysis'][selected_plot]
            for instr_key, instr_results in instruction_results.items():
                analysis_instance = analysis_class(save_dir=self.global_save_dir+'/'+application+'/'+instr_key)
                analysis_functions = [func for func in dir(analysis_instance) if callable(getattr(analysis_instance, func)) and not func.startswith("__")]

                for i, func_name in enumerate(analysis_functions):
                    func = getattr(analysis_instance, func_name)
                    fig_dict = func()
                    for figure_names,figure in fig_dict.items():
                        if figure:
                            fig_filename = f'{application}_{func_name}_{instr_key}_{figure_names}.png'
                            fig_path = os.path.join(self.uploads_dir, fig_filename)
                            figure.savefig(fig_path)
                            figures_to_display.append(f'uploads/{fig_filename}')
            
        COMBINED_VARIANCE_ANALYSIS_GRAPH(
            results_dir=self.global_save_dir+'/'+application, 
            analysis_type='training', 
            results_to_show='simple'
        )
        variance_plot = self.global_save_dir+'/'+application+"/variance_comparison_training.png"
        variance_filename = f'{application}_variance_analysis.png'
        shutil.copy(variance_plot, os.path.join(self.uploads_dir, variance_filename))
        figures_to_display.append(f'uploads/{variance_filename}')

        return jsonify({
            'figures': figures_to_display
        })

    def home(self):
        template_path = os.path.join(app.template_folder, 'index.html')
        print(f"Trying to get HTML file from: {template_path}")
        return render_template('index.html')

    def process_input(self):
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided'}), 400


        user_input = data.get('userInput', '')
        application = data.get('selectedApps', [])[0]
        config_input = data.get('localConfigInput', '')
        observed_states_filename = data.get('observedStateInput', '')

        if not application:
            return jsonify({'error': 'No application selected'}), 400

        instruction_descriptions = user_input.split('\n')
        instructions = [f'{i}' for i in range(0, len(instruction_descriptions))]

        results = {}
        console_output = ''
        match_plots = []

        selected_agents = data.get('selectedAgents', ['Qlearntab'])
        training_episodes = data.get('trainingEpisodes', 1000)
        config = pull_data.setup()
        config.update({
            'problem_type': data.get('problemType', 'Default'),
            'number_training_episodes': int(training_episodes),
            'number_training_repeats': int(data.get('trainingRepeats', 5)),
            'number_training_seeds': int(data.get('trainingSeeds', 1)),
            'number_test_episodes': int(data.get('testEpisodes', 200)),
            'number_test_repeats': int(data.get('testRepeats', 10)),
            'agent_select': selected_agents,
            'agent_parameters': {}
        })

        # Always include Qlearntab for search agent
        config['agent_parameters']['Qlearntab'] = {
            'alpha': float(data.get('alpha', 0.1)),
            'gamma': float(data.get('gamma', 0.95)),
            'epsilon': float(data.get('epsilon', 0.2)),
            'epsilon_step': float(data.get('epsilonStep', 0.01))
        }

        if 'DQN' in selected_agents:
            dqn_params = data.get('dqnParams', {})
            config['agent_parameters']['DQN'] = {
                'input_type': 'lm',
                'input_size': int(dqn_params.get('input_size', 384)),
                'sent_hidden_dim': int(dqn_params.get('sent_hidden_dim', 10)),
                'hidden_dim': int(dqn_params.get('hidden_dim', 128)),
                'num_hidden': int(dqn_params.get('num_hidden', 2)),
                'sequence_size': int(dqn_params.get('sequence_size', 20)),
                'memory_size': int(dqn_params.get('memory_size', 2000)),
                'target_replace_iter': int(dqn_params.get('target_replace_iter', 100)),
                'learning_rate': float(dqn_params.get('learning_rate', 0.001)),
                'batch_size': int(dqn_params.get('batch_size', 1))
            }
     
        for n, agent in enumerate(selected_agents):
            adapter_select = self.pull_app_data[application]['adapters']
            adapter_list = ['DQN_language' if 'language' in adapter.lower() or 'lang' in adapter.lower() else agent for adapter in adapter_select]
        config['adapter_select'] = adapter_list
        
        self.ExperimentConfig = config

        engine = self.pull_app_data[application]['engine']
        local_config = self.pull_app_data[application]['local_configs'][config_input]
        adapters = self.pull_app_data[application]['adapters']
        if len(self.pull_app_data[application]['prerender_data'].keys())>0:
            print("Pre-rendered data found...")
            observed_states = self.pull_app_data[application]['prerender_data'][observed_states_filename]
            self.elsci_run = elsci_search(Config=self.ExperimentConfig,
                                        LocalConfig=local_config,
                                        Engine=engine, Adapters=adapters,
                                        save_dir=self.global_save_dir,
                                        number_exploration_episodes=self.num_explor_epi,
                                        match_sim_threshold=0.9,
                                        observed_states=observed_states)
        else:
            print("No pre-rendered data found...")
            self.elsci_run = elsci_search(Config=self.ExperimentConfig,
                                        LocalConfig=local_config,
                                        Engine=engine, Adapters=adapters,
                                        save_dir=self.global_save_dir,
                                        number_exploration_episodes=self.num_explor_epi,
                                        match_sim_threshold=0.9,
                                        observed_states=None)
            observed_states = self.elsci_run.search()
            with open(os.path.join(self.uploads_dir, 'observed_states.txt'), 'w') as f:
                json.dump(observed_states, f)

        best_match_dict, instruction_results = self.elsci_run.match(
            action_cap=5,
            instructions=instructions,
            instr_descriptions=instruction_descriptions
        )
        results[application] = best_match_dict
        if application not in self.instruction_results:
            self.instruction_results[application] = {}
        self.instruction_results[application]['instr_'+str(self.global_input_count)] = instruction_results

        try:
            console_output += f'<br><b>Results for {application}:</b><br>'
            for n, instr in enumerate(list(results[application].keys())):
                if results[application][instr] is None:
                    console_output+='<b>'+str(n+1)+' - '+instruction_descriptions[n]+':</b> <i>No match found</i><br>'
                else:
                    best_match = results[application][instr]['best_match']
                    console_output+='<b>'+str(n+1)+' - '+instruction_descriptions[n]+':</b> <i>'+results[application][instr]['sub_goal']+'</i><br>'
                    
                    plot_filename = f'match_plot_{n}.png'
                    plot_path = os.path.abspath(os.path.join(self.uploads_dir, plot_filename))
                    print(f"Saving plot to (absolute path): {plot_path}")
                    
                    engine_dummy = engine(local_config)
                    instr_match_plot = engine_dummy.render(best_match)
                    instr_match_plot_filename = f'current_state_plot_{n}.png'
                    instr_match_plot_path = os.path.abspath(os.path.join(self.uploads_dir, instr_match_plot_filename))
                    instr_match_plot.savefig(instr_match_plot_path)
                    
                    if os.path.exists(instr_match_plot_path):
                        print(f"Current state plot created successfully at {instr_match_plot_path}")
                        print(f"File size: {os.path.getsize(instr_match_plot_path)} bytes")
                        match_plots.append(f'uploads/{instr_match_plot_filename}')
                    else:
                        print(f"Error: Current state plot not created at {instr_match_plot_path}")

        except Exception as e:
            print(f"Error in process_input: {str(e)}")
            raise

        prerender_image = self.get_prerender_image(application)

        return jsonify({
            'console_output': console_output,
            'matchPlots': match_plots,
            'prerenderImage': prerender_image
        })

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

    def upload_file(self):
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'File uploaded successfully'

    def get_applications(self):
        return jsonify({
            'applications': self.available_applications
        })
    
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

        for app in self.available_applications:
            all_observed_states[app] = self.get_observed_states([app])
            all_local_configs[app] = self.get_local_configs(app)
            all_plot_options[app] = self.get_plot_options(app)

        return jsonify({
            'localConfigs': all_local_configs,
            'observedStates': all_observed_states,
            'plotOptions': all_plot_options
        })

    def new_instruction(self):
        self.global_input_count += 1
        return jsonify({'status': 'success'})

    def confirm_result(self):
        data = request.json

        is_correct = data.get('isCorrect')
        user_input = data.get('userInput')

        if is_correct is None or user_input is None:
            print("Error: Invalid data provided")
            print(f"Received data: {data}")
            return jsonify({'error': 'Invalid data provided'}), 400            
        
        print(f"Received confirmation: isCorrect={is_correct}, userInput={user_input}")
        
        # Filter instruction results if validated correct
        application = data.get('selectedApps', [])[0]
        if application not in self.instruction_results_validated:
            self.instruction_results_validated[application] = {}

        if is_correct:
            self.instruction_results_validated[application]['instr_'+str(self.global_input_count)] = self.instruction_results[application]['instr_'+str(self.global_input_count)]
            self.correct_instructions.append(user_input)
            message = "<br>Great! Training an agent with this as guidance to complete the task... <br> See the results tab once training is complete."
        else:
            self.incorrect_instructions.append(user_input)
            message = "<br>Thanks for the feedback. The model will use this to improve."

        # Instr count increased on new instr or when correct/incorrect
        self.global_input_count += 1
        
        return jsonify({
            'status': 'received',
            'message': message
        })

    def get_correct_instructions(self):
        return jsonify({
            'correctInstructions': self.correct_instructions
        })

if len(sys.argv)>1:
    if 'search' in sys.argv[1]:
        if 'output' in sys.argv[1]:
            print('Using pre-rendered data from '+sys.argv[1])
            input_save_dir= sys.argv[1]
        else:
            print('Using pre-rendered data from ./output/'+sys.argv[1])
            input_save_dir= './output/'+sys.argv[1]
        if len(sys.argv)>2:
            input_explor_epi = int(sys.argv[2])
        else:
            input_explor_epi = 1000
    else:
        input_save_dir=''
        if len(sys.argv)==2:
            input_explor_epi = int(sys.argv[1])
        else:
            input_explor_epi = 1000
else:
    input_save_dir = './output'
    input_explor_epi = 1000

WebApp = WebApp(save_dir=input_save_dir,
                num_explor_epi=input_explor_epi)

@app.route('/')
def home_route():
    WebApp.load_data()
    app.config['UPLOAD_FOLDER'] = WebApp.uploads_dir
    return WebApp.home()

@app.route('/process_input', methods=['POST'])
def process_input_route():
    response = WebApp.process_input()
    return response

@app.route('/confirm_result', methods=['POST'])
def confirm_result_route():
    return WebApp.confirm_result()

@app.route('/train_model', methods=['POST'])
def train_model_route():
    return WebApp.train_model()

@app.route('/search', methods=['POST'])
def search_route():
    save_dir = request.json.get('save_dir', '')
    return jsonify({'save_dir': save_dir})

@app.route('/upload', methods=['POST'])
def upload_file_route():
    return WebApp.upload_file()

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
        filename = os.path.basename(file_path)
        return send_from_directory(directory, filename, as_attachment=False)
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
    return WebApp.get_applications()

@app.route('/get_observed_states', methods=['POST'])
def get_observed_states_route():
    data = request.get_json()
    selected_applications = data.get('applications', [])
    observed_states = WebApp.get_observed_states(selected_applications)
    return jsonify({
        'observedStates': observed_states
    })

@app.route('/get_local_configs', methods=['POST'])
def get_local_configs_route():
    data = request.get_json()
    selected_application = data.get('application', [])
    local_configs = WebApp.get_local_configs(selected_application)
    return jsonify({
        'localConfigs': local_configs
    })

@app.route('/get_plot_options', methods=['POST'])
def get_plot_options_route():
    data = request.get_json()
    selected_application = data.get('application', '')
    plot_options = WebApp.get_plot_options(selected_application)
    return jsonify({
        'plotOptions': plot_options
    })

@app.route('/get_all_options')
def get_all_options_route():
    return WebApp.get_all_options()

@app.route('/get_prerender_image', methods=['POST'])
def get_prerender_image_route():
    data = request.get_json()
    application = data.get('application', '')
    if not application:
        return jsonify({'error': 'No application selected'}), 400
    image_paths = WebApp.get_prerender_image(application)
    if image_paths:
        return jsonify({'imagePaths': image_paths})
    return jsonify({'error': 'No prerender images found'}), 404

@app.route('/new_instruction', methods=['POST'])
def new_instruction_route():
    response = WebApp.new_instruction()
    return response

@app.route('/get_correct_instructions')
def get_correct_instructions_route():
    return WebApp.get_correct_instructions()

@app.route('/load_data')
def load_data_route():
    WebApp.global_save_dir = ''
    WebApp.load_data()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    if not os.path.exists(os.path.join(WebApp.global_save_dir, 'uploads')):
        os.makedirs(os.path.join(WebApp.global_save_dir, 'uploads'))
    app.run(debug=True)



