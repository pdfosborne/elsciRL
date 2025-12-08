"""
Run the following to test:
    pytest tests/test_policy_gradient_classroom.py -k Classroom
"""

import pytest

from elsciRL.application_suite.import_tool import PullApplications
from elsciRL.experiments.policy_gradient import PolicyGradientExperiment


@pytest.mark.integration
@pytest.mark.slow
def test_policy_gradient_runs_on_classroom(tmp_path):
    pytest.importorskip("stable_baselines3")

    puller = PullApplications()
    try:
        application_data = puller.pull(['Classroom'])
    except Exception as exc:
        pytest.skip(f"Classroom application unavailable: {exc}")

    classroom_data = application_data.get('Classroom')
    if not classroom_data:
        pytest.skip("Classroom application data missing")

    default_adapter = 'default'
    if default_adapter not in classroom_data['adapters']:
        default_adapter = list(classroom_data['adapters'].keys())[0]

    # Sanity-check the Classroom engine's API matches Gym expectations.
    engine_cls = classroom_data['engine']
    engine_instance = engine_cls(classroom_data['local_configs']['classroom_A'])
    initial_obs = engine_instance.reset()
    step_output = engine_instance.step(state=initial_obs, action=0)
    assert isinstance(step_output, tuple), "Engine.step should return a tuple"
    assert len(step_output) == 4, f"Engine.step must return 4 values, got {len(step_output)}"

    base_experiment_data = {
        "number_training_episodes": 100,
        "number_training_repeats": 1,
        "number_training_seeds": 1,
        "number_test_episodes": 100,
        "number_test_repeats": 1,
        "training_action_cap": 16,
        "testing_action_cap": 16,
        "test_agent_type": "best",
        "reward_signal": [1, 0, 0],
        "train": True,
        "live_env": True,
    }

    agent_configs = {
        "PPO": {
            "agent_parameters": {
                "learning_rate": 3e-4,
                "batch_size": 64,
                "minibatch_size": 16,
                "update_epochs": 2,
                "hidden_size": 64,
            }
        },
    }

    for agent_type, config in agent_configs.items():
        experiment_data = base_experiment_data.copy()
        experiment_data["agent_select"] = [agent_type]
        experiment_data["adapter_select"] = [default_adapter]
        experiment_data["adapter_input_dict"] = {agent_type: [default_adapter]}
        experiment_data["agent_parameters"] = {agent_type: config["agent_parameters"]}

        experiment_config = {"data": experiment_data}
        local_config = {"data": classroom_data['local_configs']['classroom_A']}

        agent_tmp_dir = tmp_path / agent_type
        agent_tmp_dir.mkdir(parents=True, exist_ok=True)

        experiment = PolicyGradientExperiment(
            Config=experiment_config,
            ProblemConfig=local_config,
            Engine=classroom_data['engine'],
            Adapters=classroom_data['adapters'],
            save_dir=str(agent_tmp_dir),
            show_figures='No',
            window_size=0.1,
        )
        
        print(f"Training {agent_type} agent...")
        training_setups = experiment.train()
        assert training_setups, f"Policy gradient training should generate setups for {agent_type}"

        print(f"Testing {agent_type} agent...")
        evaluation = experiment.test()
        assert evaluation is not None

        print(f"Rendering {agent_type} agent...")
        render_dir = agent_tmp_dir / "renders"
        render_outputs = experiment.render_results(render_save_dir=str(render_dir))
        assert render_outputs is not None
        print('Test complete')


if __name__ == "__main__":
    import argparse
    import tempfile
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run policy-gradient classroom test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where test artifacts should be saved. Uses a temp dir if omitted.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        test_policy_gradient_runs_on_classroom(args.output_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_policy_gradient_runs_on_classroom(Path(tmp_dir))
