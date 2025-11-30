import os
import json
from typing import Callable, Dict, Optional

from elsciRL.environment_setup.results_table import ResultsTable
from elsciRL.experiments.experiment_utils.config_utils import ensure_dir, merge_configs
from elsciRL.experiments.experiment_utils.env_manager import EnvManager
from elsciRL.experiments.experiment_utils.result_manager import ResultManager
from elsciRL.experiments.experiment_utils.policy_agent_factory import PolicyAgentFactory
from elsciRL.evaluation.standard_report import Evaluation
from elsciRL.experiments.training_procedures.policy_gradient import run_policy_gradient_training_loop
from elsciRL.interaction_loops.policy_gradient import PolicyGradientInteractionLoop


class PolicyGradientExperiment:
    """Policy-gradient experiment helper focused on train/test/render workflows."""

    def __init__(
        self,
        Config: dict,
        ProblemConfig: dict,
        Engine,
        Adapters: dict,
        save_dir: str,
        show_figures: str,
        window_size: float,
        policy_agent_factory=None,
        create_subdirectory: bool = False,
    ):
        if isinstance(Engine, dict):
            self.engine_comparison = True
            self.engine_list = Engine
        else:
            self.engine_comparison = False
            self.engine_list = {"DefaultEng": Engine}
            self.engine = Engine

        self.adapters = Adapters
        self.env_manager = EnvManager(PolicyGradientInteractionLoop, Adapters)

        self.ExperimentConfig = Config
        self.LocalConfig = ProblemConfig

        ensure_dir(save_dir)
        # Allow control over subdirectory creation for when called from other experiments
        if create_subdirectory:
            self.save_dir = os.path.join(save_dir, "PolicyGradient_Experiment")
        else:
            self.save_dir = save_dir
        self.show_figures = show_figures

        try:
            self.setup_info = self.ExperimentConfig["data"] | self.LocalConfig["data"]
        except Exception:
            self.setup_info = merge_configs(self.ExperimentConfig, self.LocalConfig)

        self.training_setups = {}
        self.trained_agents = {}
        self.num_training_seeds = self.setup_info.get("number_training_seeds", 1)
        self.test_agent_type = self.setup_info.get("test_agent_type", "best")
        self.analysis = Evaluation(window_size=window_size)
        self.result_manager = ResultManager(self.analysis)
        self.policy_agent_factory = policy_agent_factory or PolicyAgentFactory()
        self.default_wrapper_builder: Optional[Callable[[Dict], Optional[list]]] = None

    @staticmethod
    def _resolve_wrappers(
        wrapper_builder: Optional[Callable[[Dict], Optional[list]]],
        setup_info: Dict,
    ):
        if wrapper_builder is None:
            return None
        wrappers = wrapper_builder(setup_info)
        if not wrappers:
            return None
        return wrappers

    def train(self, wrapper_builder: Optional[Callable[[Dict], Optional[list]]] = None):
        for engine_name, engine in self.engine_list.items():
            for agent_type in self.setup_info.get("agent_select", []):
                for adapter in self.setup_info["adapter_input_dict"][agent_type]:
                    train_setup_info = self.setup_info.copy()
                    train_setup_info["agent_type"] = agent_type
                    train_setup_info["adapter_select"] = adapter
                    train_setup_info["train"] = True
                    train_setup_info.setdefault("agent_name", f"{engine_name}_{agent_type}_{adapter}")
                    train_setup_info.setdefault("training_results", False)
                    train_setup_info.setdefault("observed_states", False)
                    train_setup_info.setdefault("engine_name", engine_name)

                    wrappers = self._resolve_wrappers(wrapper_builder, train_setup_info)

                    trained_agents, *_ = run_policy_gradient_training_loop(
                        self.env_manager,
                        self.policy_agent_factory,
                        self.result_manager,
                        training_render=False,
                        training_render_save_dir=None,
                        save_dir=self.save_dir,
                        engine_name=engine_name,
                        engine=engine,
                        agent_type=agent_type,
                        adapter=adapter,
                        train_setup_info=train_setup_info,
                        trained_agents=self.trained_agents,
                        num_training_seeds=self.num_training_seeds,
                        test_agent_type=self.test_agent_type,
                        show_figures=self.show_figures,
                        number_training_repeats=self.ExperimentConfig.get("number_training_repeats", 1),
                        wrappers=wrappers,
                    )
                    self.trained_agents.update(trained_agents)
                    setup_key = f"Training_Setup_{engine_name}_{agent_type}_{adapter}"
                    self.training_setups[setup_key] = train_setup_info.copy()

        self.result_manager.training_variance_report(
            self.save_dir,
            self.show_figures,
        )

        return self.training_setups

    def _load_training_setups(self, training_setups):
        if training_setups is None:
            return self.training_setups
        if isinstance(training_setups, dict):
            return training_setups
        if isinstance(training_setups, (str, os.PathLike)):
            with open(training_setups, "r", encoding="utf-8") as setup_file:
                return json.load(setup_file)
        return training_setups

    def _prepare_test_setup(self, setup_info: Dict):
        """Prepare test setup info to match instruction following patterns."""
        test_setup_info = setup_info.copy()
        test_setup_info["train"] = False
        test_setup_info["training_results"] = False
        test_setup_info["observed_states"] = False
        # Support both 'number_test_episodes' and 'num_test_episodes' for compatibility
        if "num_test_episodes" in test_setup_info:
            test_setup_info["number_test_episodes"] = test_setup_info["num_test_episodes"]
        else:
            test_setup_info.setdefault(
                "number_test_episodes", self.setup_info.get("number_test_episodes", 1)
            )
        return test_setup_info

    def _evaluate_agents(
        self,
        training_setups,
        render: bool = False,
        render_save_dir: Optional[str] = None,
        save_reports: bool = True,
        wrapper_builder: Optional[Callable[[Dict], Optional[list]]] = None,
    ):
        setups = self._load_training_setups(training_setups)
        evaluation_outputs: Dict[str, list] = {}
        if not setups:
            return evaluation_outputs

        for setup_key, setup in setups.items():
            test_setup_info = self._prepare_test_setup(setup)
            agent_type = test_setup_info["agent_type"]
            adapter = test_setup_info["adapter_select"]
            engine_name = test_setup_info.get("engine_name", next(iter(self.engine_list)))
            engine = self.engine_list[engine_name]

            wrappers = self._resolve_wrappers(wrapper_builder, test_setup_info)

            agent_key = f"{engine_name}_{agent_type}_{adapter}"
            agent_name = test_setup_info.get("agent_name", agent_key)
            stored_agents = self.trained_agents.get(agent_key, {})
            if agent_name not in stored_agents:
                continue

            agents = stored_agents[agent_name]
            if not isinstance(agents, list):
                agents = [agents]

            evaluation_outputs[setup_key] = []
            num_episodes = test_setup_info.get("number_test_episodes", 1)
            if render:
                num_episodes = 1

            for idx, agent in enumerate(agents, start=1):
                env = self.env_manager.create_gym_env(engine, adapter, test_setup_info, wrappers=wrappers)
                rollout_name = f"{agent_name}_eval{idx}"
                results_table = ResultsTable(test_setup_info)
                table_results = PolicyGradientInteractionLoop.policy_rollout(
                    agent,
                    env,
                    rollout_name,
                    num_episodes,
                    results_table,
                    render=render,
                    render_save_dir=render_save_dir,
                    action_limit=test_setup_info.get("testing_action_cap"),
                )

                if save_reports:
                    table_results["episode"] = table_results.index
                    table_results.insert(loc=0, column="Repeat", value=idx)
                    test_save_dir = os.path.join(
                        self.save_dir,
                        f"{engine_name}_{agent_type}_{adapter}__testing_results_{idx}",
                    )
                    ensure_dir(test_save_dir)
                    Return = self.result_manager.test_report(table_results, test_save_dir, self.show_figures)
                    evaluation_outputs[setup_key].append(Return)
                else:
                    evaluation_outputs[setup_key].append(table_results)

        return evaluation_outputs

    def test(
        self,
        training_setups=None,
        render: bool = False,
        render_save_dir: Optional[str] = None,
        wrapper_builder: Optional[Callable[[Dict], Optional[list]]] = None,
    ):
        """Evaluate trained policy-gradient agents on their Gym environments."""

        evaluation_outputs = self._evaluate_agents(
            training_setups,
            render=render,
            render_save_dir=render_save_dir,
            save_reports=True,
            wrapper_builder=wrapper_builder,
        )

        self.result_manager.testing_variance_report(self.save_dir, self.show_figures)

        return evaluation_outputs

    def render_results(
        self,
        training_setups=None,
        render_save_dir: Optional[str] = None,
        wrapper_builder: Optional[Callable[[Dict], Optional[list]]] = None,
    ): 
        """Render trained agents' policies with optional GIF output."""

        render_dir = render_save_dir or os.path.join(self.save_dir, "render_results")
        ensure_dir(render_dir)
        return self._evaluate_agents(
            training_setups,
            render=True,
            render_save_dir=render_dir,
            save_reports=False,
            wrapper_builder=wrapper_builder,
        )
