import os
from elsciRL.experiments.experiment_utils.config_utils import ensure_dir
from elsciRL.experiments.experiment_utils.render_current_results import render_current_result


def run_training_loop(
    env_manager,
    agent_factory,
    result_manager,
    training_render,
    training_render_save_dir,
    save_dir,
    engine_name,
    engine,
    agent_type,
    adapter,
    all_adapters,
    train_setup_info,
    trained_agents,
    num_training_seeds,
    test_agent_type,
    show_figures,
    number_training_repeats,
    gym_env:bool=False
):
    if f"{engine_name}_{agent_type}_{adapter}" not in trained_agents:
        trained_agents[f"{engine_name}_{agent_type}_{adapter}"] = {}
    seed_recall = {}
    seed_results_connection = {}
    for seed_num in range(num_training_seeds):
        if num_training_seeds > 1:
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
            agent = agent_factory.create(agent_type, train_setup_info['agent_parameters'][agent_type], engine, adapter)
            train_setup_info['agent'] = agent
            # Create the environment, use gym_env if specified
            if gym_env:
                live_env = env_manager.create_gym_env(engine, adapter, train_setup_info)
            else:
                live_env = env_manager.create_env(engine, all_adapters, train_setup_info)
            # ---
            if training_repeat > 1:
                live_env.start_obs = env_start
            env_start = live_env.start_obs
            goal = str(env_start).split(".")[0] + "---GOAL"
            print("Flat agent Goal: ", goal)
            if goal in seed_recall:
                setup_num = seed_recall[goal]
            else:
                seed_recall[goal] = 1
            agent_save_dir = os.path.join(save_dir,
                f"{engine_name}_{agent_type}_{adapter}__training_results_{goal}_{setup_num}"
                    ) if num_training_seeds > 1 else os.path.join(save_dir,
                            f"{engine_name}_{agent_type}_{adapter}__training_results_{setup_num}"
                        )
            ensure_dir(agent_save_dir)
            if goal in trained_agents[f"{engine_name}_{agent_type}_{adapter}"]:
                live_env.agent = trained_agents[f"{engine_name}_{agent_type}_{adapter}"][goal].clone()
                live_env.agent.exploration_parameter_reset()
            if goal in seed_results_connection:
                live_env.results.load(seed_results_connection[goal])
            training_results = live_env.episode_loop()
            training_results['episode'] = training_results.index
            training_results.insert(loc=0, column='Repeat', value=setup_num)
            Return = result_manager.train_report(training_results, agent_save_dir, show_figures)
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
            if training_render:
                current_render_save_dir = training_render_save_dir or agent_save_dir
                render_current_result(
                    training_setup=train_setup_info,
                    current_environment=live_env,
                    current_agent=live_env.agent,
                    local_save_dir=current_render_save_dir
                )
        seed_results_connection[goal] = training_results_stored
        # Save trained agent(s)
        if test_agent_type.lower() == 'best':
            trained_agents[f"{engine_name}_{agent_type}_{adapter}"][goal] = best_agent.clone()
        elif test_agent_type.lower() == 'all':
            start_repeat_num = list(temp_agent_store[goal].keys())[0]
            end_repeat_num = list(temp_agent_store[goal].keys())[-1]
            all_agents = [temp_agent_store[goal][repeat]['agent'] for repeat in range(start_repeat_num, end_repeat_num + 1)]
            trained_agents[f"{engine_name}_{agent_type}_{adapter}"][goal] = all_agents

    return trained_agents, seed_results_connection, temp_agent_store, training_results_stored, observed_states_stored
