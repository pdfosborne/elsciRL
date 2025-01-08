import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
import matplotlib
matplotlib.use('Agg')
import json
import shutil
# Init Flask app and set local directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, static_folder=os.path.join(dir_path, 'static'), 
            template_folder=os.path.join(dir_path, 'templates'))
# ---
from datetime import datetime
# ====== elsciRL IMPORTS =========================================
# ------ Train/Test Function Imports ----------------------------
from elsciRL.examples.WebApp.elsciRL_demo_search import elsciRLSearch as elsci_search
from elsciRL.experiments.standard import Experiment as STANDARD_RL
from elsciRL.instruction_following.elsciRL_instruction_following import elsciRLOptimize
# ------ Visual Analysis -----------------------------------------------
from elsciRL.examples.WebApp.sailing_graphs import Analysis as SailingAnalysis
# ====== LOCAL IMPORTS ==========================================
# ------ Local Environment --------------------------------------
from elsciRL.examples.environments.elsciRL_sailing import Engine as Sailing
# ------ Local Adapters -----------------------------------------
from elsciRL.examples.adapters.elsciRL_sailing_default import DefaultAdapter as SailingDefault
from elsciRL.examples.adapters.elsciRL_sailing_language import LanguageAdapter as SailingLanguage
# ------ Benchmark Fixed Config -----------------------------------------------
# Meta parameters
from elsciRL.examples import experiment_config as ExperimentConfig
# Local Parameters
from elsciRL.examples.local_configs import sailing_config_local as SailingLocalConfig
# ------ Visual Analysis -----------------------------------------------
from elsciRL.analysis.combined_variance_visual import combined_variance_analysis_graph as COMBINED_VARIANCE_ANALYSIS_GRAPH
# ----------------------------------------------------------------------
# Pre-rendered data
from elsciRL.examples.WebApp.prerender import observed_states as observed_state_data
from elsciRL.examples.WebApp.prerender.Standard_Experiment import training_data
from elsciRL.examples.WebApp.prerender.Standard_Experiment import testing_data
import csv
# ----------------------------------------------------------------------

# Adapters for the Sailing Environment
# |--> override selection to only language
SAILING_ADAPTERS = {'Default':SailingDefault,'Language': SailingLanguage}
# |--> only use Qlearntab agent
ExperimentConfig.ExperimentConfigData['agent_select'] = ["Qlearntab"]
# |--> reduce training episodes/repeats for faster training
ExperimentConfig.ExperimentConfigData['number_training_episodes'] = 1000
ExperimentConfig.ExperimentConfigData['number_training_repeats'] = 5
ExperimentConfig.ExperimentConfigData['number_test_episodes'] = 50
ExperimentConfig.ExperimentConfigData['number_test_repeats'] = 5
# |--> reduce state space size
SailingLocalConfig.LocalConfigData['obs_precision'] = 1
# |--> override local config selection to only language
SailingLocalConfig.LocalConfigData['adapter_select'] = ["Language"]

        
# Used to allow multiple users input attempts without overriding results
#global_input_count = 0

class WebApp:
    def __init__(self, save_dir: str = '', num_explor_epi: int = 1000):
        self.global_input_count = 0
        self.global_save_dir = save_dir
        self.num_explor_epi = num_explor_epi
        
    def load_data(self):
        if self.num_explor_epi == 1000:
            # Load pre-rendered search data
            self.observed_states = observed_state_data.observed_data
            print("Pre-rendered search data loaded...")
        else:
            print("Rendering state space search data...")
            self.observed_states = self.elsci_run.search(action_cap=100)
        # --------------------
        # Setup save directory
        
        if not os.path.exists('./output'):
            os.mkdir('./output')
        if 'search' not in self.global_save_dir:
            time = datetime.now().strftime("%d-%m-%Y_%H-%M")
            save_dir = './output/' + str('search') + '_' + time
            if not os.path.exists(save_dir):                
                os.mkdir(save_dir)
            self.global_save_dir = save_dir

        # Directory for uploaded files and generated plots
        if not os.path.exists(self.global_save_dir+'/uploads'):                
            os.mkdir(self.global_save_dir+'/uploads')
        
        # --------------------
        # Save observed states to file
        with open(self.global_save_dir + '/observed_states.txt', 'w') as file:
            file.write(json.dumps(self.observed_states))

        self.elsci_run = elsci_search(Config=ExperimentConfig.ExperimentConfigData,
                                     LocalConfig=SailingLocalConfig.LocalConfigData,
                                     Engine=Sailing, Adapters=SAILING_ADAPTERS,
                                     save_dir=self.global_save_dir,
                                     number_exploration_episodes=self.num_explor_epi,
                                     match_sim_threshold=0.9,
                                     observed_states=self.observed_states)
        print("GLOBAL SAVE DIR: ", self.global_save_dir)
        # Make sure uploads directory exists and is accessible
        self.uploads_dir = os.path.abspath(os.path.join(self.global_save_dir, 'uploads'))
        if not os.path.exists(self.uploads_dir):
            os.makedirs(self.uploads_dir, exist_ok=True)
        print(f"Uploads directory (absolute path): {self.uploads_dir}")

    def train_model(self):
        user_input = request.json.get('userInput')
        training_episodes = request.json.get('trainingEpisodes', 1000)  # Default to 1000 if not provided
        
        # Update config with new episode count
        ExperimentConfig.ExperimentConfigData['number_training_episodes'] = training_episodes
        
        instruction_descriptions = user_input.split('\n')
        instructions = [f'{i}' for i in range(0, len(instruction_descriptions))]

        _,instruction_results = self.elsci_run.match(action_cap=5,
                                    instructions=instructions,
                                    instr_descriptions=instruction_descriptions)
        print("_________ INSTRUCTION RESULTS _________")
        print(instruction_results)
        # Take Instruction path now defined with reinforced+unsupervised sub-goal locations and train to these
        # Init experiment setup with sub-goal defined
        reinforced_experiment = elsciRLOptimize(
                        Config=ExperimentConfig.ExperimentConfigData, 
                        LocalConfig=SailingLocalConfig.LocalConfigData, 
                        Engine=Sailing, Adapters=SAILING_ADAPTERS,
                        save_dir=self.global_save_dir + '/input_'+str(self.global_input_count), 
                        show_figures = 'No', window_size=0.1,
                        instruction_path=instruction_results, predicted_path=None, 
                        instruction_episode_ratio=0.1,
                        instruction_chain=True, instruction_chain_how='exact' )
        reinforced_experiment.train()
        reinforced_experiment.test()
        #reinforced_experiment.render_results()
        # --------------------------------------------------------------------
        # Flat Baselines
        # |--> Preloaded data to save time
        if not os.path.exists(self.global_save_dir+'/input_'+str(self.global_input_count)+'/Standard_Experiment'): 
            os.mkdir(self.global_save_dir+'/input_'+str(self.global_input_count)+'/Standard_Experiment')
        # Training data
        headers = ["","agent","num_repeats","episode","avg_R_mean","avg_R_se",
                   "cum_R_mean","cum_R_se","time_mean"]
        with open(self.global_save_dir+'/input_'+str(self.global_input_count)
                  +'/Standard_Experiment/training_variance_results_Qlearntab_Language.csv', 'w') as csv_file:  
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            for key, value in training_data.data.items():
                writer.writerow(value)
        # Testing data
        with open(self.global_save_dir+'/input_'+str(self.global_input_count)
                  +'/Standard_Experiment/testing_variance_results_Qlearntab_Language.csv', 'w') as csv_file:  
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            for key, value in testing_data.data.items():
                writer.writerow(value)                
        # --------------------------------------------------------------------
        analysis = SailingAnalysis(save_dir=self.global_save_dir+'/input_'+str(self.global_input_count))
        analysis.trace_plot(experiments_names=['Experiment WITHOUT Instructions',
                                            'Experiment WITH Instructions'])
        
        # |--> CHANGED TO PRERENDER IMAGE 
        #no_instr_trace_plot = self.global_save_dir+'/input_'+str(self.global_input_count)+"/Standard_Experiment/trace_plot.png"
        #no_instr_trace_plot = 'numeric_trace_plot.png'
        
        instr_trace_plot = self.global_save_dir+'/input_'+str(self.global_input_count)+"/Supervised_Instr_Experiment/trace_plot.png"
        
        COMBINED_VARIANCE_ANALYSIS_GRAPH(results_dir=self.global_save_dir+'/input_'+str(self.global_input_count), analysis_type='training', 
            results_to_show='simple', 
            experiment_names=['WITHOUT','WITH INSTRUCTION'])
        graph_image = self.global_save_dir+'/input_'+str(self.global_input_count)+"/variance_comparison_training.png"
        
        shutil.copy(instr_trace_plot, os.path.join(self.global_save_dir, 'uploads', 'results_1.png'))
        #shutil.copy(no_instr_trace_plot, os.path.join(self.global_save_dir, 'uploads', 'results_2.png'))
        shutil.copy(graph_image, os.path.join(self.global_save_dir, 'uploads', 'results_3.png'))

        return jsonify({
            #'current_state_image': 'uploads/sailing_setup.png',
            'result_image': 'uploads/results_1.png',
            #'graph_image': 'uploads/numeric_trace_plot.png',
            'additional_image': 'uploads/results_3.png'
        })

    def home(self):
        template_path = os.path.join(app.template_folder, 'index.html')
        print(f"Trying to get HTML file from: {template_path}")
        return render_template('index.html')

    def process_input(self):
        user_input = request.form.get('command')
        self.global_input_count += 1

        instruction_descriptions = user_input.split('\n')
        instructions = [f'{i}' for i in range(0, len(instruction_descriptions))]

        best_match_dict,instruction_results = self.elsci_run.match(action_cap=5,
                                        instructions=instructions,
                                        instr_descriptions=instruction_descriptions)

        analysis = SailingAnalysis(save_dir=self.global_save_dir+'/input_'+str(self.global_input_count))

        console_output = ''
        current_state_image = None
        
        try:
            for n,instr in enumerate(list(best_match_dict.keys())):
                if best_match_dict[instr] is None:
                    console_output+='<b>'+str(n+1)+' - '+instruction_descriptions[n]+':</b> <i>No match found</i><br>'
                else:
                    best_match = best_match_dict[instr]['best_match']
                    console_output+='<b>'+str(n+1)+' - '+instruction_descriptions[n]+':</b> <i>'+best_match_dict[instr]['sub_goal']+'</i><br>'
                    
                    # Generate and save the match plot
                    plot_filename = f'match_plot_{n}.png'
                    plot_path = os.path.abspath(os.path.join(self.uploads_dir, plot_filename))
                    print(f"Saving plot to (absolute path): {plot_path}")
                    
                    instr_match_plot = analysis.render(best_match)
                    instr_match_plot.savefig(plot_path)
                    
                    if os.path.exists(plot_path):
                        print(f"Plot file created successfully at {plot_path}")
                        print(f"File size: {os.path.getsize(plot_path)} bytes")
                        current_state_image = f'uploads/{plot_filename}'
                    else:
                        print(f"Error: Plot file not created at {plot_path}")
        except Exception as e:
            print(f"Error in process_input: {str(e)}")
            raise

        return console_output, current_state_image
    
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


# --- TERMINAL INPUTS ---
# Accept inputs in terminal
if len(sys.argv)>1:
    if 'search' in sys.argv[1]:
        if 'output' in sys.argv[1]:
            print('Using pre-rendered data from '+sys.argv[1])
            input_save_dir= sys.argv[1] #'./output/search_02-01-2025_18-21'
        else:
            print('Using pre-rendered data from ./output/'+sys.argv[1])
            input_save_dir= './output/'+sys.argv[1] #'search_02-01-2025_18-21'
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

# ----------------------------------------------------------------------

@app.route('/')
def home_route():
    WebApp.load_data()
    app.config['UPLOAD_FOLDER'] = WebApp.uploads_dir  # Update upload folder to use uploads directory
    return WebApp.home()

@app.route('/process_input', methods=['POST'])
def process_input_route():
    console_output, current_state_image = WebApp.process_input()
    return jsonify({
        'console_output': console_output,
        'current_state_image': current_state_image
    }) 

@app.route('/confirm_result', methods=['POST'])
def confirm_result():
    data = request.json
    is_correct = data.get('isCorrect')
    
    if is_correct:
        message = "<br>Great! Training an agent with this as guidance to complete the task... <br> See the results tab once training is complete."
    else:
        message = "<br>Thanks for the feedback. The model will use this to improve."
    
    return jsonify({
        'status': 'received',
        'message': message
    })

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

# Add route for static files
@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        print(f"Error serving static file {filename}: {str(e)}")
        return f"Error: {str(e)}", 404

if __name__ == '__main__':
    # Ensure uploads folder exists
    if not os.path.exists(os.path.join(WebApp.global_save_dir, 'uploads')):
        os.makedirs(os.path.join(WebApp.global_save_dir, 'uploads'))
    app.run(debug=True)

