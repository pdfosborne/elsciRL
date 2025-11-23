import os
import json
# ------ Interaction Protocol -----------------------------------
from elsciRL.interaction_loops.standard_gym import GymInteractionLoop
# ------ Experiment Import --------------------------------------
from elsciRL.evaluation.standard_report import Evaluation
# ------ Agent Imports -----------------------------------------
# Universal Agents
from elsciRL.agents.agent_abstract import Agent, QLearningAgent
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.DQN import DQNAgent
from elsciRL.agents.agent_abstract import Agent
# Gym based agents
from elsciRL.environment_setup.gym_translator import GymRegistration
from elsciRL.agents.stable_baselines.SB3_DQN import SB_DQN
from elsciRL.agents.stable_baselines.SB3_PPO import SB_PPO
from elsciRL.agents.stable_baselines.SB3_A2C import SB_A2C
from elsciRL.experiments.experiment_utils.env_manager import EnvManager
from elsciRL.experiments.experiment_utils.result_manager import ResultManager
from elsciRL.experiments.experiment_utils.agent_factory import AgentFactory
from elsciRL.experiments.training_procedures.default_exp_training import run_training_loop

# This is the main run functions for elsciRL to be imported
# Defines the train/test operators and imports all the required agents and experiment functions ready to be used
# The local main.py file defines the [adapters, configs, environment] to be input

# This should be where the environment is initialized and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead
# -> DONE: This means we have multiple main.py types (e.g. with/without convergence measure) so should create a directory and finalize naming for this

class GymExperiment:
    """This is the GYM Reinforcement Learning experiment setup for a flat of hierarchy agent. 
    - ONLY GYM AGENTS ARE SUPPORTED - To uses non-Gym agents use the StandardExperiment class
    - The agent is trained for a fixed number of episodes
    - Then learning is fixed to be applied during testing phase
    - Repeats (or seeds if environment start position changes) are used for statistical significant testing
    """
    def __init__(self, Config:dict, ProblemConfig:dict, Engine, Adapters:dict, save_dir:str, show_figures:str, window_size:float): 
        # Environment setup
        if isinstance(Engine, dict):
            print("\n Multiple Engines detected, will compare results across engines...")
            self.engine_comparison = True
            self.engine_list = Engine
        else:
            self.engine_comparison = False
            self.engine_list = {'DefaultEng':Engine}
        self.adapters = Adapters
        self.env_manager = EnvManager(GymInteractionLoop, Adapters)
        # ---
        # Configuration setup
        self.ExperimentConfig = Config
        self.LocalConfig = ProblemConfig
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self.show_figures = show_figures
        if (self.show_figures.lower() != 'y') and (self.show_figures.lower() != 'yes'):
            print("Figures will not be shown and only saved.")
        try:
            self.setup_info = self.ExperimentConfig['data'] | self.LocalConfig['data'] 
        except Exception:
            self.setup_info = self.ExperimentConfig | self.LocalConfig 
        self.training_setups: dict = {}
        self.trained_agents: dict = {}
        self.num_training_seeds = self.setup_info['number_training_seeds']
        self.test_agent_type = self.setup_info['test_agent_type']
        self.analysis = Evaluation(window_size=window_size)
        self.result_manager = ResultManager(self.analysis)
        self.agent_factory = AgentFactory(self.adapters, self.setup_info)
        self.reward_signal = None

    def add_agent(self, agent_name:str, agent):
        self.agent_factory.register_agent(agent_name, agent)
        print("\n Agent added to experiment, all available agents: ", self.agent_factory.agent_types)

    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        for engine_name, engine in self.engine_list.items():
            for agent_type in self.setup_info['agent_select']:
                for adapter in self.setup_info['adapter_input_dict'][agent_type]:
                    train_setup_info = self.setup_info.copy()
                    agent_parameters = train_setup_info['agent_parameters'][agent_type]
                    train_setup_info['agent_type'] = agent_type
                    train_setup_info['agent_name'] = f"{engine_name}{agent_type}_{adapter}_{agent_parameters}"
                    train_setup_info['adapter_select'] = adapter
                    train_setup_info['train'] = True
                    number_training_repeats = self.ExperimentConfig["number_training_repeats"]
                    # Induce reward signal into engine
                    engine.reward_signal = self.reward_signal
                    # Use run_training_loop utility
                    trained_agents, seed_results_connection, temp_agent_store, training_results_stored, observed_states_stored = run_training_loop(
                        self.env_manager,
                        self.agent_factory,
                        self.result_manager,
                        training_render=False,
                        training_render_save_dir=None,
                        save_dir=self.save_dir,
                        engine_name=engine_name,
                        engine=engine,
                        agent_type=agent_type,
                        adapter=adapter,
                        all_adapters=self.adapters,
                        train_setup_info=train_setup_info,
                        trained_agents=self.trained_agents,
                        num_training_seeds=self.num_training_seeds,
                        test_agent_type=self.test_agent_type,
                        show_figures=self.show_figures,
                        number_training_repeats=number_training_repeats,
                        gym_env=True  # Use gym_env flag if available
                    )
                    self.trained_agents.update(trained_agents)
                    self.training_setups[f'Training_Setup_{engine_name}_{agent_type}_{adapter}'] = train_setup_info
        if (self.ExperimentConfig["number_training_repeats"] > 1) or (self.num_training_seeds > 1):
            self.result_manager.training_variance_report(self.save_dir, self.show_figures)
        return self.training_setups

    def test(self, training_setups:str=None):
        if training_setups is None:
            training_setups = self.training_setups
        else:
            training_setups = json.load(training_setups)
        for training_key in list(training_setups.keys()):    
            test_setup_info = training_setups[training_key]
            test_setup_info['train'] = False # Testing Phase
            test_setup_info['training_results'] = False
            test_setup_info['observed_states'] = False
            print("----------")
            print("Testing results for trained agents in saved setup configuration:")
            print(test_setup_info['train_save_dir'])
            number_training_repeats = test_setup_info['number_test_repeats']
            agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']
            for engine_name, engine in self.engine_list.items():
                for testing_repeat in range(0, test_setup_info['number_test_repeats']):  
                    engine.reward_signal = None # clear instr reward signal
                    env = self.env_manager.create_gym_env(engine, test_setup_info['adapter_select'], test_setup_info)
                    start_obs = env.start_obs
                    goal = str(start_obs).split(".")[0] + "---GOAL"
                    print("Flat agent Goal: ", goal)
                    if goal in self.trained_agents.get(f"{engine_name}_{test_setup_info['agent_type']}_{test_setup_info['adapter_select']}", {}):
                        print("Trained agent available for testing.")
                        all_agents = self.trained_agents[str(engine_name) + '_' + test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
                        if not isinstance(all_agents, list):
                            all_agents = [all_agents]
                    else:
                        print("NO agent available for testing position.")
                    env.agent.epsilon = 0 # Remove random actions
                    for ag,agent in enumerate(all_agents):
                        env.results.reset() # Reset results table for each agent
                        env.start_obs = start_obs
                        env.agent = agent
                        env.agent.epsilon = 0 # Remove random actions
                        agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']
                        # ---
                        # Testing generally is the agents replaying on the testing ENV
                        testing_results = env.episode_loop() 
                        test_save_dir = (self.save_dir+'/'+ str(engine_name) + '_' + agent_adapter + '__testing_results_' + str(goal).split("/")[0]+"_"+"agent"+str(ag)+"-repeat"+str(testing_repeat))
                        if not os.path.exists(test_save_dir):
                            os.mkdir(test_save_dir)
                        # Produce training report with Analysis.py
                        Return = self.analysis.test_report(testing_results, test_save_dir, self.show_figures)
        if (self.ExperimentConfig["number_training_repeats"] > 1) or (self.test_agent_type.lower() == 'all'):
            self.result_manager.testing_variance_report(self.save_dir, self.show_figures)