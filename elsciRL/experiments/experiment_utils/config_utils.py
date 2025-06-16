import os
import json

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(config_path):
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        # Add more config formats if needed
        raise ValueError("Unsupported config file format.")


def merge_configs(config1, config2):
    # Simple dict merge, can be improved for deep merge
    merged = config1.copy()
    merged.update(config2)
    return merged
