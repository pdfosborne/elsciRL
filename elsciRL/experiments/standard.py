import os
import json
# ------ Interaction Protocol -----------------------------------
from elsciRL.interaction_loops.standard import StandardInteractionLoop
# ------ Experiment Import --------------------------------------
from elsciRL.evaluation.standard_report import Evaluation
# ------ Agent Imports -----------------------------------------
# Universal Agents
from elsciRL.agents.agent_abstract import Agent, QLearningAgent
from elsciRL.agents.table_q_agent import TableQLearningAgent
from elsciRL.agents.DQN import DQNAgent
from elsciRL.agents.agent_abstract import Agent
# Stable Baselines
from elsciRL.agents.stable_baselines.SB3_DQN import SB_DQN
from elsciRL.agents.stable_baselines.SB3_PPO import SB_PPO
from elsciRL.agents.stable_baselines.SB3_A2C import SB_A2C
# ------ Gym Experiment ----------------------------------------
from elsciRL.experiments.GymExperiment import GymExperiment
from elsciRL.experiments.experiment_utils.render_current_results import render_current_result
# ------ LLM Agents ---------------------------------------------
from elsciRL.agents.LLM_agents.ollama_agent import LLMAgent as OllamaAgent
# ---------------------------------------------------------------

from elsciRL.experiments.experiment_utils.agent_factory import AgentFactory
from elsciRL.experiments.experiment_utils.config_utils import ensure_dir, merge_configs
from elsciRL.experiments.experiment_utils.env_manager import EnvManager
from elsciRL.experiments.experiment_utils.result_manager import ResultManager

# This is the main run functions for elsciRL to be imported
# Defines the train/test operators and imports all the required agents and experiment functions ready to be used
# The local main.py file defines the [adapters, configs, environment] to be input

# This should be where the environment is initialised and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead
# -> DONE: This means we have multiple main.py types (e.g. with/without convergence measure) so should create a directory and finalize naming for this

class Experiment:
    """Standard RL experiment setup, modularized."""
    def __init__(self, Config:dict, ProblemConfig:dict, Engine, Adapters:dict, save_dir:str, show_figures:str, window_size:float, 
                 training_render:bool=False, training_render_save_dir:str=None):
        # Environment setup
        if isinstance(Engine, dict):
            print("\n Multiple Engines detected, will compare results across engines...")
            self.engine_comparison = True
            self.engine_list = Engine
        else:
            self.engine_comparison = False
            self.engine_list = {'DefaultEng':Engine}
            self.engine = Engine
        self.adapters = Adapters
        self.env_manager = EnvManager(StandardInteractionLoop, Adapters)
        # ---
        # Configuration setup
        self.ExperimentConfig = Config
        self.LocalConfig = ProblemConfig
        ensure_dir(save_dir)
        self.save_dir = os.path.join(save_dir, 'Standard_Experiment')
        self.show_figures = show_figures
        if (self.show_figures.lower() != 'y') and (self.show_figures.lower() != 'yes'):
            print("Figures will not be shown and only saved.")
        self.training_render = training_render
        self.training_render_save_dir = training_render_save_dir
        try:
            self.setup_info = self.ExperimentConfig['data'] | self.LocalConfig['data'] 
        except Exception:
            self.setup_info = merge_configs(self.ExperimentConfig, self.LocalConfig)
        # Adapter/agent matching
        if 'adapter_input_dict' in self.ExperimentConfig:
            self.setup_info['adapter_input_dict'] = self.ExperimentConfig['adapter_input_dict']
        else:
            selected_adapters = list(Adapters.keys())
            selected_agents = self.ExperimentConfig.get('agent_select', [])
            agent_adapter_dict = {agent_name: list(selected_adapters) for agent_name in selected_agents} if selected_agents else {}
            self.ExperimentConfig['adapter_input_dict'] = agent_adapter_dict
            self.setup_info['adapter_input_dict'] = agent_adapter_dict
        self.training_setups = {}
        self.trained_agents = {}
        self.num_training_seeds = self.setup_info.get('number_training_seeds', 1)
        self.test_agent_type = self.setup_info.get('test_agent_type', 'all')
        self.analysis = Evaluation(window_size=window_size)
        self.result_manager = ResultManager(self.analysis)
        # Gym setup
        self.is_gym_agent = {}
        for agent_type in self.setup_info.get('agent_select', []):
            if agent_type.split('_')[0] == "SB3":
                self.is_gym_agent[agent_type] = True 
                self.gym_exp = GymExperiment(Config=self.ExperimentConfig, ProblemConfig=self.LocalConfig, 
                        Engine=self.engine, Adapters=self.adapters,
                        save_dir=self.save_dir, show_figures = self.show_figures, window_size=0.1)
            else:
                self.is_gym_agent[agent_type] = False 
        self.agent_factory = AgentFactory(self.adapters)

    def add_agent(self, agent_name:str, agent):
        self.agent_factory.register_agent(agent_name, agent)
        print("\n Agent added to experiment, all available agents: ", self.agent_factory.agent_types)


    def train(self):
        ensure_dir(self.save_dir)
        for engine_name, engine in self.engine_list.items():
            for agent_type in self.setup_info['agent_select']:
                is_gym_agent = self.is_gym_agent[agent_type]
                agent_adapter = (
                    agent_type + '_' + self.setup_info['adapter_input_dict'][agent_type][0]
                )
                if is_gym_agent:
                    train_setup_info = self.setup_info.copy()
                    for adapter in train_setup_info['adapter_input_dict'][agent_type]:
                        self.gym_exp.setup_info['agent_select'] = [agent_type]
                        self.training_setups[agent_adapter] = self.gym_exp.train()
                else:
                    train_setup_info = self.setup_info.copy()
                    for adapter in train_setup_info['adapter_input_dict'][agent_type]:
                        agent_parameters = train_setup_info['agent_parameters'][agent_type]
                        train_setup_info['agent_type'] = agent_type
                        train_setup_info['agent_name'] = (
                            f"{engine_name}{agent_type}_{adapter}_{agent_parameters}"
                        )
                        train_setup_info['adapter_select'] = adapter
                        train_setup_info['train'] = True
                        # Neural agent input size
                        player = self.agent_factory.create(agent_type, agent_parameters, adapter)
                        train_setup_info['agent'] = player
                        train_setup_info['live_env'] = True
                        number_training_repeats = self.ExperimentConfig["number_training_repeats"]
                        if f"{engine_name}_{agent_type}_{adapter}" not in self.trained_agents:
                            self.trained_agents[f"{engine_name}_{agent_type}_{adapter}"] = {}
                        seed_recall = {}
                        seed_results_connection = {}
                        for seed_num in range(self.num_training_seeds):
                            if self.num_training_seeds > 1:
                                print("------\n- Seed Num: ", seed_num)
                            if seed_num == 0:
                                train_setup_info['training_results'] = False
                                train_setup_info['observed_states'] = False
                            else:
                                train_setup_info['training_results'] = False
                                train_setup_info['observed_states'] = observed_states_stored.copy()
                            setup_num = 0
                            temp_agent_store = {}
                            for training_repeat in range(1, number_training_repeats + 1):
                                if number_training_repeats > 1:
                                    print("------\n- Repeat Num: ", training_repeat)
                                setup_num += 1
                                player = self.agent_factory.create(agent_type, agent_parameters, adapter)
                                train_setup_info['agent'] = player
                                live_env = self.env_manager.create_env(engine, self.adapters, train_setup_info)
                                if training_repeat > 1:
                                    live_env.start_obs = env_start
                                env_start = live_env.start_obs
                                goal = str(env_start).split(".")[0] + "---GOAL"
                                print("Flat agent Goal: ", goal)
                                if goal in seed_recall:
                                    setup_num = seed_recall[goal]
                                else:
                                    seed_recall[goal] = 1
                                agent_save_dir = os.path.join(
                                    self.save_dir,
                                    f"{engine_name}_{agent_type}_{adapter}__training_results_{goal}_{setup_num}"
                                ) if self.num_training_seeds > 1 else os.path.join(
                                    self.save_dir,
                                    f"{engine_name}_{agent_type}_{adapter}__training_results_{setup_num}"
                                )
                                ensure_dir(agent_save_dir)
                                if goal in self.trained_agents[f"{engine_name}_{agent_type}_{adapter}"]:
                                    live_env.agent = self.trained_agents[f"{engine_name}_{agent_type}_{adapter}"][goal].clone()
                                    live_env.agent.exploration_parameter_reset()
                                if goal in seed_results_connection:
                                    live_env.results.load(seed_results_connection[goal])
                                training_results = live_env.episode_loop()
                                training_results['episode'] = training_results.index
                                training_results.insert(loc=0, column='Repeat', value=setup_num)
                                Return = self.result_manager.train_report(training_results, agent_save_dir, self.show_figures)
                                if goal not in temp_agent_store:
                                    temp_agent_store[goal] = {}
                                temp_agent_store[goal][setup_num] = {'Return': Return, 'agent': live_env.agent.clone()}
                                if training_repeat == 1:
                                    max_Return = Return
                                    best_agent = live_env.agent
                                    training_results_stored = live_env.results.copy()
                                    observed_states_stored = live_env.elsciRL.observed_states
                                if Return > max_Return:
                                    max_Return = Return
                                    best_agent = live_env.agent
                                    training_results_stored = live_env.results.copy()
                                    observed_states_stored = live_env.elsciRL.observed_states
                                seed_recall[goal] = seed_recall[goal] + 1
                                train_setup_info['train_save_dir'] = agent_save_dir
                                if self.training_render:
                                    current_render_save_dir = self.training_render_save_dir or agent_save_dir
                                    render_current_result(
                                        training_setup=train_setup_info,
                                        current_environment=live_env,
                                        current_agent=live_env.agent,
                                        local_save_dir=current_render_save_dir
                                    )
                            seed_results_connection[goal] = training_results_stored
                            # Save trained agent(s)
                            if self.test_agent_type.lower() == 'best':
                                self.trained_agents[f"{engine_name}_{agent_type}_{adapter}"][goal] = best_agent.clone()
                            elif self.test_agent_type.lower() == 'all':
                                start_repeat_num = list(temp_agent_store[goal].keys())[0]
                                end_repeat_num = list(temp_agent_store[goal].keys())[-1]
                                all_agents = [temp_agent_store[goal][repeat]['agent'] for repeat in range(start_repeat_num, end_repeat_num + 1)]
                                self.trained_agents[f"{engine_name}_{agent_type}_{adapter}"][goal] = all_agents
                        self.training_setups[f'Training_Setup_{engine_name}_{agent_type}_{adapter}'] = train_setup_info.copy()
        self.result_manager.training_variance_report(self.save_dir, self.show_figures)
        return self.training_setups

    # TESTING PLAY
    def test(self, training_setups:str=None):
        # Use saved training setups if not provided
        if training_setups is None:
            training_setups = self.training_setups
        else:
            training_setups = json.load(training_setups)
        for training_key in list(training_setups.keys()):
            test_setup_info = training_setups[training_key]
            test_setup_info['train'] = False
            test_setup_info['training_results'] = False
            test_setup_info['observed_states'] = False
            agent_type = test_setup_info['agent_type']
            print("----------\n" + training_key)
            print("Testing results for trained agents in saved setup configuration:")
            print("TESTING SETUP INFO")
            print(test_setup_info['agent_type'])
            print(test_setup_info['adapter_select'])
            print("----------")
            agent_adapter = agent_type + "_" + test_setup_info['adapter_select']
            if self.is_gym_agent[agent_type]:
                gym_test_exp = self.training_setups[agent_adapter]
                gym_test_exp.reward_signal = None
                gym_test_exp.test()
            else:
                if self.test_agent_type.lower() == 'best':
                    for engine_name, engine in self.engine_list.items():
                        for testing_repeat in range(0, test_setup_info['number_test_repeats']):
                            env = self.env_manager.create_env(engine, self.adapters, test_setup_info)
                            start_obs = env.start_obs
                            goal = str(start_obs).split(".")[0] + "---GOAL"
                            print("Flat agent Goal: ", goal)
                            agent_key = f"{engine_name}_{test_setup_info['agent_type']}_{test_setup_info['adapter_select']}"
                            if goal in self.trained_agents[agent_key]:
                                print("Trained agent available for testing.")
                                env.agent = self.trained_agents[agent_key][goal]
                            else:
                                print("NO agent available for testing position.")
                            try:
                                env.agent.epsilon = 0
                            except Exception:
                                print(self.trained_agents)
                                raise KeyError("Trained agents lookup not found for testing position.")
                            testing_results = env.episode_loop()
                            test_save_dir = os.path.join(
                                self.save_dir,
                                f"{engine_name}_{agent_adapter}__testing_results_{str(goal).split('/')[0]}_{testing_repeat}"
                            )
                            ensure_dir(test_save_dir)
                            Return = self.result_manager.test_report(testing_results, test_save_dir, self.show_figures)
                elif self.test_agent_type.lower() == 'all':
                    for engine_name, engine in self.engine_list.items():
                        for testing_repeat in range(0, test_setup_info['number_test_repeats']):
                            env = self.env_manager.create_env(engine, self.adapters, test_setup_info)
                            start_obs = env.start_obs
                            goal = str(start_obs).split(".")[0] + "---GOAL"
                            print("Flat agent Goal: ", goal)
                            agent_key = f"{engine_name}_{test_setup_info['agent_type']}_{test_setup_info['adapter_select']}"
                            if goal in self.trained_agents[agent_key]:
                                print("Trained agents available for testing.")
                                all_agents = self.trained_agents[agent_key][goal]
                            else:
                                print("NO agent available for testing position.")
                                all_agents = []
                            for ag, agent in enumerate(all_agents):
                                env.results.reset()
                                env.start_obs = start_obs
                                env.agent = agent
                                env.agent.epsilon = 0
                                agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']
                                testing_results = env.episode_loop()
                                test_save_dir = os.path.join(
                                    self.save_dir,
                                    f"{engine_name}_{agent_adapter}__testing_results_{str(goal).split('/')[0]}_agent{ag}-repeat{testing_repeat}"
                                )
                                ensure_dir(test_save_dir)
                                Return = self.result_manager.test_report(testing_results, test_save_dir, self.show_figures)
        self.result_manager.testing_variance_report(self.save_dir, self.show_figures)

        
    def render_results(self, training_setups:str=None):
        if training_setups is None:
            training_setups = self.training_setups
        else:
            training_setups = json.load(training_setups)
        for training_key in list(training_setups.keys()):
            test_setup_info = training_setups[training_key]
            test_setup_info['train'] = False
            test_setup_info['training_results'] = False
            test_setup_info['observed_states'] = False
            test_setup_info['num_test_episodes'] = 10
            print("----------\nRendering trained agent's policy:")
            agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']
            if self.test_agent_type.lower() == 'best':
                for engine_name, engine in self.engine_list.items():
                    env = self.env_manager.create_env(engine, self.adapters, test_setup_info)
                    start_obs = env.start_obs
                    goal = str(start_obs).split(".")[0] + "---GOAL"
                    print("Flat agent Goal: ", goal)
                    agent_key = f"{engine_name}_{test_setup_info['agent_type']}_{test_setup_info['adapter_select']}"
                    if goal in self.trained_agents[agent_key]:
                        print("Trained agent available for testing.")
                        env.agent = self.trained_agents[agent_key][goal]
                    else:
                        print("NO agent available for testing position.")
                    env.agent.epsilon = 0
                    render_save_dir = os.path.join(self.save_dir, 'render_results')
                    ensure_dir(render_save_dir)
                    render_results = env.episode_loop(render=True, render_save_dir=render_save_dir)
            else:
                for engine_name, engine in self.engine_list.items():
                    env = self.env_manager.create_env(engine, self.adapters, test_setup_info)
                    start_obs = env.start_obs
                    goal = str(start_obs).split(".")[0] + "---GOAL"
                    print("Flat agent Goal: ", goal)
                    agent_key = f"{engine_name}_{test_setup_info['agent_type']}_{test_setup_info['adapter_select']}"
                    if goal in self.trained_agents[agent_key]:
                        print("Trained agents available for testing.")
                        all_agents = self.trained_agents[agent_key][goal]
                    else:
                        print("NO agent available for testing position.")
                        all_agents = []
                    for ag, agent in enumerate(all_agents):
                        env.results.reset()
                        env.start_obs = start_obs
                        env.agent = agent
                        env.num_train_repeat = 1
                        env.num_test_repeat = 1
                        env.agent.epsilon = 0
                        render_save_dir = os.path.join(self.save_dir, 'render_results')
                        ensure_dir(render_save_dir)
                        render_results = env.episode_loop(render=True, render_save_dir=render_save_dir)