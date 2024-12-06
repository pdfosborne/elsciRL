import os
import json
import logging
# Define Agent's parameters for problem
# Opponent is considered a 'local' specification as benchmarks vary between setting

class LocalConfig:
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

#TODO this is not universal at all !!!
class ProblemConfig(LocalConfig):
    """Local Config is used to define any problem specific parameters."""
    def __init__(self, config_path: str):
        super(ProblemConfig, self).__init__(config_path)
        # State form
        self.adapter_select = self.data.get("adapter_select", [""])
        # Enabled agent to be trained against multiple opponents in order provided
        self.training_opponent_agent = self.data.get(
            "training_opponent_agent", "")
        self.testing_opponent_agent = self.data.get(
            "testing_opponent_agent", "")
        
        self.training_setup = self.data.get("training_setup",'default')
        self.testing_setup = self.data.get("testing_setup",'default')
        
        self.training_action_cap = self.data.get("training_action_cap",1000) # Arbitrary number to ensure games dont last forever
        self.testing_action_cap = self.data.get("testing_action_cap",1000) # Arbitrary number to ensure games dont last forever
        # Reward Signal, should be consistent between all agent being compared
        self.reward_signal = self.data.get("reward_signal",[1,-0.1,0,0] )# [Value of winning, Value for draw, Value for each action, Value for reaching new state]
        # Sub-Goal Defined
        self.sub_goal = self.data.get("sub_goal",None)
        
class ConfigSetup(LocalConfig):
    def __init__(self, config_dir: str):
        super(ConfigSetup, self).__init__(config_dir)
        self.state_configs = ProblemConfig(os.path.join(config_dir))