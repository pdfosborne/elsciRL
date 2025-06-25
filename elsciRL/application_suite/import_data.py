
#  Define data through class function so it can be called within package
# Instead of using a .json file which is hard to load from local install
# NOTE: MAKE SURE TO TRUST REPOSITORIES BEFORE RUNNING CODE
# - Can set branch to specific commit to ensure no changes are made without knowledge
#   |-----> changed to commit id which is tied to branch and more stable
# - Compatibility defined to a single engine file
#   |-----> Adapters must be compatible with the given engine
# - Experiment configs are defined in the experiment_configs folder
#   |-----> NOTE: AT LEAST TWO EXPERIMENT CONFIGS MUST BE DEFINED
#       |-----> This is so that it triggers the selection swap on the server side
class Applications:
    def __init__(self):
        self.data ={
            "Sailing":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-Sailing",
                "commit_id": "*",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"quick-test":"testing.json", 
                                                "Osborne-2024":"config.json"},
                "local_config_filenames": {"easy":"config_local.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"default":"default", "language":"language", "LLM":"LLM_adapter"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {"sailing_graphs":"sailing_graphs"},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states":"observed_states.txt", "LLM_observed_states": "observed_states_Sailing_easy_LLM_1000epi.txt"},
                "prerender_data_encoded_filenames":{"observed_states":"encoded_observed_states.txt"},
                "prerender_image_filenames": {"Setup":"sailing_setup.png"}
                },
            "Classroom":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-Classroom",
                "commit_id": "*",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"default":"config.json"},
                "local_config_filenames": {"classroom_A":"classroom_A.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"default":"default", "classroom_A_language":"classroom_A_language", "LLM":"LLM_adapter"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states":"observed_states.txt", "LLM_observed_states":"observed_states_Classroom_classroom_A_LLM_100epi.txt"},
                "prerender_data_encoded_filenames":{},
                "prerender_image_filenames": {"Classroom_A_Setup":"Classroom_A_Summary.png"}
                },
            "Gym-FrozenLake":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-GymFrozenLake",
                "commit_id": "*",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"quick_test":"fast_agent.json", "Osborne2024_agent":"Osborne2024_agent.json"},
                "local_config_filenames": {"Osborne2024_env":"Osborne2024_env.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"numeric_encoder":"numeric", "language":"language", "LLM":"LLM_adapter"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states":"observed_states.txt", "LLM_observed_states":"observed_states_Gym-FrozenLake_Osborne2024_env_LLM_100epi.txt"},
                "prerender_data_encoded_filenames": {},
                "prerender_image_filenames": {"FrozenLake_Setup":"FrozenLake_4x4.png"}
                },
            "Chess":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-Chess",
                "commit_id": "*",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"Osborne2024_agent":"config.json"},
                "local_config_filenames": {"Osborne2024_env":"config_local.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"numeric_board_mapping": "numeric_board",
                                      "numeric_piece_counter":"numeric_piece_counter", 
                                      "active_pieces_language":"language_active_pieces",
                                      "LLM":"LLM_adapter"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states_active_pieces_language":"observed_states_active_pieces_language_50000_29-05-2025_16-13.txt",
                                             "observed_states_LLM_generation": "observed_states_Chess_Osborne2024_env_LLM_Generator_1000.txt"},
                "prerender_data_encoded_filenames":{},
                "prerender_image_filenames": {"Board_Setup":"board_start.png"}
                },
            "TextWorldExpress":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-TextWorldExpress",
                "commit_id": "*",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"Osborne2024_agent":"config.json"},
                "local_config_filenames": {"twc-easy":"twc-easy.json", 
                                            "twc-medium":"twc-medium.json",
                                            "twc-hard":"twc-hard.json", 
                                            "cookingworld-easy":"cookingworld-easy.json",
                                            "cookingworld-medium":"cookingworld-medium.json",
                                            "cookingworld-hard":"cookingworld-hard.json", 
                                            "coincollector":"coincollector.json",
                                            "mapreader":"mapreader.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"language_default":"language", "LLM":"LLM_adapter"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {
                    "twc-easy_observed_states":
                        "observed_states_TextWorldExpress_twc-easy_language_default_10000_29-05-2025_15-53.txt",
                    "cookingworld-easy_observed_states":
                        "observed_states_TextWorldExpress_cookingworld-easy_language_default_10000_29-05-2025_16-02.txt",},
                "prerender_data_encoded_filenames":{},
                "prerender_image_filenames": {}
                },
            "Maze":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-Maze",
                "commit_id": "*",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"default_agent":"config.json"},
                "local_config_filenames": {"umaze":"umaze.json",
                                           "double_t_maze":"double_t_maze.json",
                                           "medium":"medium.json",
                                           "large":"large.json",
                                           "random":"random.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"language_default":"language", "LLM":"LLM_adapter"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {'observed_states_umaze_language':'observed_states_Maze_umaze_language_default_10000.txt',
                                             'observed_states_umaze_LLM':'observed_states_Maze_default_LLM_100.txt'},
                "prerender_data_encoded_filenames":{},
                "prerender_image_filenames": {}
                }
        }