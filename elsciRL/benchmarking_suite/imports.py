
#  Define data through class function so it can be called within package
# Instead of using a .json file which is hard to load from local install
# NOTE: MAKE SURE TO TRUST REPOSITORIES BEFORE RUNNING CODE
# - Can set branch to specific commit to ensure no changes are made without knowledge
#   |-----> changed to commit id which is tied to branch and more stable
class Applications:
    def __init__(self):
        self.data ={
            "Sailing":{
                "github_user": "pdfosborne",
                "repository": "elsciRL-Sailing",
                "commit_id": "e384424",
                "engine_filenames": "engine",
                "local_config_filenames": {"easy":"easy_river.json", "medium":"medium_river.json"},
                "adapter_filenames": {"default":"default","language":"language"}
                }
        }