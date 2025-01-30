
#  Define data through class function so it can be called within package
# Instead of using a .json file which is hard to load from local install
# NOTE: MAKE SURE TO TRUST REPOSITORIES BEFORE RUNNING CODE
# - Can set branch to specific commit to ensure no changes are made without knowledge
#   |-----> changed to commit id which is tied to branch and more stable
# - Compatibility defined to a single engine file
#   |-----> Adapters must be compatible with the given engine
class Applications:
    def __init__(self):
        self.data ={
            "Sailing":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-App-Sailing",
                "commit_id": "aef5afd",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "local_config_folder": "configs",
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
                "commit_id": "7010ad6",
                "engine_folder": "environment",
                "engine_filename": "engine",
                "local_config_folder": "configs",
                "local_config_filenames": {"classroom_A":"classroom_A.json"},
                "local_adapter_folder": "adapters",
                "adapter_filenames": {"default":"default", "classroom_A_language":"classroom_A_language"},
                "local_analysis_folder": "analysis",
                "local_analysis_filenames": {},
                "prerender_data_folder": "prerender",
                "prerender_data_filenames": {"observed_states":"observed_states.txt"},
                "prerender_image_filenames": {"Classroom_A_Setup":"Classroom_A.png"}
                }
        }