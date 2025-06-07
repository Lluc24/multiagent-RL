import os
from tqdm import tqdm
from algorithms import JALGT
from environment import Environment, obs_to_state
from game_model import GameModel
import numpy as np
import shutil
import argparse
import json
from parameters import Parameters

def train_algorithms(parameters, env_manager, algorithms):
    td_error_per_episode = np.zeros(shape=parameters.get("episodes_per_epoch"))
    reward_per_episode = np.zeros(shape=parameters.get("episodes_per_epoch"))
    for ep in range(parameters.get("episodes_per_epoch")):
        env = env_manager.new_env(seed=ep % parameters.get("num_maps"))
        observations, infos = env.reset()
        terminated = [False for _ in range(parameters.get("num_agents"))]
        truncated = [False for _ in range(parameters.get("num_agents"))]
        states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]
        while not all(terminated) and not all(truncated):  # Hasta que acabe el episodio
            # Elegimos acciones
            actions = tuple(algorithms[i].select_action(states[i]) for i in range(parameters.get("num_agents")))
            # Ejecutamos acciones en el entorno
            observations, rewards, terminated, truncated, infos = env.step(actions)
            # Preparar siguiente iteración: convertir observaciones parciales en estados
            new_states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]
            for i in range(parameters.get("num_agents")):
                # Aprendemos: actualizamos valores Q
                algorithms[i].learn(actions, rewards, states[i], new_states[i])
                # Actualizamos métricas
                reward_per_episode[ep] += rewards[i]
            td_error_per_episode[ep] += algorithms[0].metrics["td_error"][-1]
            states = new_states
        # Actualizamos epsilon
        for i in range(parameters.get("num_agents")):
            algorithms[i].set_epsilon(parameters.get("epsilon_max") - parameters.get("epsilon_diff") * ep)
    return td_error_per_episode, reward_per_episode

def evaluate_algorithms(parameters, env_manager, algorithms, solution_concept_name, epoch):
    reward_per_episode = np.zeros(shape=parameters.get("episodes_per_epoch"))
    for ep in range(parameters.get("num_maps")):
        env = env_manager.new_env(seed=ep)  # Reaprovechamos mapas del entrenamiento
        observations, infos = env.reset()
        terminated = [False for _ in range(parameters.get("num_agents"))]
        truncated = [False for _ in range(parameters.get("num_agents"))]
        states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]
        while not all(terminated) and not all(truncated):  # Hasta que acabe el episodio
            actions = tuple(algorithms[i].select_action(states[i], train=False) for i in range(parameters.get("num_agents")))
            observations, rewards, terminated, truncated, infos = env.step(actions)
            states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]
            for i in range(parameters.get("num_agents")):
                reward_per_episode[ep] += rewards[i]
        # Guardamos animaciones
        env_manager.save_animations(solution_concept_name, ep, epoch)
    return reward_per_episode

def setup(wandb_config):
    parameters = Parameters(wandb_config)
    parameters.print()
    environment = Environment(
        num_agents=parameters.get("num_agents"),
        map_size=parameters.get("map_size"),
        obstacle_density=parameters.get("obstacle_density"),
        max_episode_steps=parameters.get("max_episode_steps"),
        obs_radius=parameters.get("observation_radius"),
        renders_directory=parameters.get("renders_dir"),
        save_every=parameters.get("save_every"),
    )

    # Crear directorio para almacenar los renders
    renders_dir = parameters.get("renders_dir")
    if os.path.exists(renders_dir):
        shutil.rmtree(renders_dir)
    os.makedirs(renders_dir)

    # Modelo de juego y algoritmos (uno para cada agente)
    game = GameModel(
        num_agents=parameters.get("num_agents"),
        num_states=parameters.get("num_states"),
        num_actions=parameters.get("num_actions"),
    )
    algorithms = [
        JALGT(
            agent_id=i,
            game=game,
            solution_concept=parameters.get("solution_concept_class")(),
            epsilon=parameters.get("epsilon_max"),
            gamma=parameters.get("gamma"),
            alpha=parameters.get("learning_rate"),
            seed=i
        )
        for i in range(game.num_agents)
    ]

    return parameters, environment, algorithms

def run(output_path, config=None):
    parameters, environment, algorithms = setup(config)
    solution_concept_name = parameters.get("solution_concept_class").__name__
    td_error_per_epoch = np.empty(shape=parameters.get("epochs"))
    train_reward_per_epoch = np.empty(shape=parameters.get("epochs"))
    evaluation_reward_per_epoch = np.empty(shape=parameters.get("epochs"))
    for epoch in tqdm(range(parameters.get("epochs"))):
        # Entrenamiento
        ###############
        td_error_per_episode, train_reward_per_episode = train_algorithms(parameters, environment, algorithms)
        td_error_per_epoch[epoch] = np.sum(td_error_per_episode)
        train_reward_per_epoch[epoch] = np.sum(train_reward_per_episode)

        # Evaluación
        ############
        evaluation_reward_per_episode = evaluate_algorithms(
            parameters, environment, algorithms, solution_concept_name, epoch
        )
        evaluation_reward_per_epoch[epoch] = np.sum(evaluation_reward_per_episode)

    with open(output_path, "w") as f:
        metrics = {
            "td_error_per_epoch": td_error_per_epoch.tolist(),
            "train_reward_per_epoch": train_reward_per_epoch.tolist(),
            "evaluation_reward_per_epoch": evaluation_reward_per_epoch.tolist(),
            "reward" : np.average(evaluation_reward_per_epoch).item(),
        }
        json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running an experiment.')
    parser.add_argument('--configuration', type=str, help='Path to the configuration file', required=False)
    parser.add_argument("--metrics", type=str, help="Path for the output metrics", required=True)
    args = parser.parse_args()
    # Load the configuration file as a dictionary (it is a JSON file)
    if args.configuration is None:
        run(args.metrics)
    else:
        with open(args.configuration, "r") as f:
            config = json.load(f)
        run(args.metrics, config)
