
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
                "commit_id": "4259e94",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"quick-test":"testing.json", 
                                                "Osborne-2024":"config.json"},
                "local_config_filenames": {"easy":"config_local.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"default":"default", "language":"language"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {"sailing_graphs":"sailing_graphs"},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states":"observed_states.txt"},
                "prerender_image_filenames": {"Setup":"sailing_setup.png"}
                },
            "Classroom":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-Classroom",
                "commit_id": "192df85",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"default":"config.json"},
                "local_config_filenames": {"classroom_A":"classroom_A.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"default":"default", "classroom_A_language":"classroom_A_language"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states":"observed_states.txt"},
                "prerender_image_filenames": {"Classroom_A_Setup":"Classroom_A_Summary.png"}
                },
            "Gym-FrozenLake":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-GymFrozenLake",
                "commit_id": "cca3b57",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"quick_test":"fast_agent.json", "Osborne2024_agent":"Osborne2024_agent.json"},
                "local_config_filenames": {"Osborne2024_env":"Osborne2024_env.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"numeric_encoder":"numeric", "language":"language"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states":"observed_states.txt"},
                "prerender_image_filenames": {"FrozenLake_Setup":"FrozenLake_4x4.png"}
                },
            "Chess":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-Chess",
                "commit_id": "316fa18",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "config_folder": "configs",
                "experiment_config_filenames": {"Osborne2024_agent":"config.json"},
                "local_config_filenames": {"Osborne2024_env":"config_local.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"numeric_piece_counter":"numeric_piece_counter", 
                                      "active_pieces_language":"language_active_pieces"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states":"observed_states.txt"},
                "prerender_image_filenames": {"Board_Setup":"board_start.png"}
                },
            "TextWorldExpress":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-TextWorldExpress",
                "commit_id": "46cd028",
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
                "adapter_filenames": {"language_default":"language"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {},
                "prerender_image_filenames": {}
                }
        }