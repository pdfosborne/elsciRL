import os
import logging
import json

class Config:
    def __init__(self, config_file_path: str):
        if (config_file_path):
            with open(config_file_path) as config_file:
                self.data = json.load(config_file)
                self.config_file_path = config_file_path
                
        else:
            self.data = dict()
            self.config_path = ""
            logging.info("No arguments given, using default configuration...")
        
    def __getitem__(self, key: str):
        item = None

        if (key in self.__dict__):
            item = self.__dict__[key]
        else:
            item = self.data[key]

        return item

class ExperimentConfig(Config):
    def __init__(self, config_path: str):
        super(ExperimentConfig, self).__init__(config_path)
        
        # Name setup
        self.name = self.data.get(
            "name", 
            os.path.split(self.config_file_path)[-1].replace(".json", "")
            )
        # Define Problem Type Choice
        self.problem_type = self.data.get("problem_type", "")
        # Specify local config choices to select agents of interest
        self.agent_select = self.data.get("agent_select", ["Qlearntab"])
        
        # ---> We then parse these three inputs to obtain the local config setup info
        # ---> Ideally input is a dict input: setups = { 'Setup1':{"Adapter":"Engine", "Encoder":"Yes", "Agent":"TabQ"},... }

        # Training repeated 
        self.experience_sample_batch_size = self.data.get(
            "experience_sample_batch_ratio", 0)

        # Training repeated 
        self.num_episodes = self.data.get("num_episodes", 1000)
        self.number_training_repeats = self.data.get(
            "number_training_repeats", 2)

        # Testing repeated
        self.number_test_episodes = self.data.get("number_test_episodes", 100)
        self.number_test_repeats = self.data.get("number_test_repeats", 5)
        self.test_agent_type = self.data.get("test_agent_type","best")

        # Tab Q Agent parameters
        self.alpha = self.data.get("alpha", [0.05])
        self.gamma = self.data.get("gamma", [0.95])
        self.epsilon = self.data.get("epsilon", [0.05])
        # Neural Agent Parameters
        self.input_type = "lm"
        self.input_size = self.data.get("input_size", [384])
        self.sent_hidden_dim = self.data.get("sent_hidden_dim", [10])
        self.hidden_dim = self.data.get("hidden_dim", [128])
        self.num_hidden = self.data.get("num_hidden", [2])
        self.sequence_size = self.data.get("sequence_size", [20])
        self.memory_size = self.data.get("memory_size", [2000])
        self.target_replace_iter = self.data.get("target_replace_iter", [100])
        self.learning_rate = self.data.get("learning_rate", [0.001])
        self.batch_size = self.data.get("batch_size", [1])
        
        self.number_test_episodes = self.data.get("number_test_episodes", 250)
        self.number_test_repeats = self.data.get("number_test_repeats", 5)
        
class TestingSetupConfig(Config):
    def __init__(self, config_dir: str):
        super(TestingSetupConfig, self).__init__(config_dir)
        self.state_configs = ExperimentConfig(os.path.join(config_dir))
                                              
        