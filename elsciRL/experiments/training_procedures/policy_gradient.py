import os

from elsciRL.environment_setup.results_table import ResultsTable
from elsciRL.experiments.experiment_utils.config_utils import ensure_dir
from elsciRL.interaction_loops.policy_gradient import PolicyGradientInteractionLoop


def run_policy_gradient_training_loop(
    env_manager,
    policy_agent_factory,
    result_manager,
    training_render,
    training_render_save_dir,
    save_dir,
    engine_name,
    engine,
    agent_type,
    adapter,
    train_setup_info,
    trained_agents,
    num_training_seeds,
    test_agent_type,
    show_figures,
    number_training_repeats,
    wrappers=None,
):
    """Specialized training loop for policy-gradient agents."""

    key = f"{engine_name}_{agent_type}_{adapter}"
    if key not in trained_agents:
        trained_agents[key] = {}

    seed_recall = {}
    seed_results_connection = {}
    observed_states_stored = {}
    training_results_stored = None

    for seed_num in range(num_training_seeds):
        if num_training_seeds > 1:
            print("------\n- Seed Num: ", seed_num)

        setup_num = 0
        temp_agent_store = {}

        for training_repeat in range(1, number_training_repeats + 1):
            setup_num += 1
            env = env_manager.create_gym_env(engine, adapter, train_setup_info, wrappers=wrappers)
            agent_parameters = train_setup_info['agent_parameters'][agent_type]
            agent = policy_agent_factory.create(agent_type, agent_parameters, env)

            total_steps = train_setup_info.get('training_action_cap', 100) * train_setup_info.get('number_training_episodes', 1)
            agent.learn(total_steps=total_steps)

            agent_name = train_setup_info.get('agent_name', f"{agent_type}_{adapter}")
            results_table = ResultsTable(train_setup_info)
            table_results = PolicyGradientInteractionLoop.policy_rollout(
                agent,
                env,
                agent_name,
                train_setup_info.get('number_training_episodes', 1),
                results_table,
                render=False,
                action_limit=train_setup_info.get('training_action_cap'),
            )

            table_results['episode'] = table_results.index
            table_results.insert(loc=0, column='Repeat', value=setup_num)

            agent_save_dir = os.path.join(
                save_dir,
                f"{engine_name}_{agent_type}_{adapter}__training_results_{setup_num}"
            )
            ensure_dir(agent_save_dir)
            Return = result_manager.train_report(table_results, agent_save_dir, show_figures)
            train_setup_info['train_save_dir'] = agent_save_dir

            if key not in temp_agent_store:
                temp_agent_store[key] = {}
            temp_agent_store[key][setup_num] = {'Return': Return, 'agent': agent, 'train_setup': train_setup_info.copy()}

            seed_recall[agent_name] = setup_num
            training_results_stored = table_results

        seed_results_connection[key] = training_results_stored

        def _select_training_setups():
            if test_agent_type.lower() == 'best':
                best_repeat = max(temp_agent_store[key], key=lambda r: temp_agent_store[key][r]['Return'])
                return [temp_agent_store[key][best_repeat]]
            if test_agent_type.lower() == 'all':
                return list(temp_agent_store[key].values())
            return [temp_agent_store[key][setup_num]]

        selected_setups = _select_training_setups()
        trained_agents[key][agent_name] = [entry['agent'] for entry in selected_setups] if len(selected_setups) > 1 else selected_setups[0]['agent']

        training_setups_for_key = {}
        for idx, entry in enumerate(selected_setups, start=1):
            training_setup = entry['train_setup']
            repeat_label = entry.get('train_setup', {}).get('Repeat', idx)
            training_setups_for_key[f"Training_Setup_{engine_name}_{agent_type}_{adapter}_{repeat_label}"] = training_setup

    return trained_agents, seed_results_connection, temp_agent_store, training_results_stored, observed_states_stored
