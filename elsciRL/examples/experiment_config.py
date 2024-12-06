ExperimentConfigData = {
    "name": "Example Experiment",
    "problem_type": "Examples",
    
    "experience_sample_batch_ratio": 0,
    
    "number_training_episodes": 100,
    "number_training_repeats": 5,
    "number_training_seeds": 1,

    "test_agent_type":"all",
    "number_test_episodes": 25,
    "number_test_repeats": 5,

    "agent_select": ["Qlearntab", "Qlearntab"],
    "agent_parameters":{
        "Qlearntab":{
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon": 0.2,
            "epsilon_step":0.01
            }
        }
    }