import os
from tqdm import tqdm
from algorithms import JALGT
from environment import Environment, obs_to_state
from game_model import GameModel
import shutil
import argparse
import json
from parameters import Parameters
from metrics import Metrics

def train_algorithms(parameters, env_manager, algorithms, metrics):
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
            states = new_states

            # métricas
            metrics.add_rewards(rewards)
            metrics.add_steps(terminated, truncated)
            metrics.add_td_errors([algorithms[i].metrics["td_error"][-1]for i in range(parameters.get("num_agents"))])
        metrics.add_to_success_rates(terminated, truncated)

def evaluate_algorithms(parameters, env_manager, algorithms, metrics, solution_concept_name, epoch):
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

            # métricas
            metrics.add_rewards(rewards, training=False)
            metrics.add_steps(terminated, truncated, training=False)
        metrics.add_to_success_rates(terminated, truncated, training=False)
        # Guardamos animaciones
        env_manager.save_animations(solution_concept_name, ep, epoch)

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
            alpha=parameters.get("alpha_max"),
            seed=i
        )
        for i in range(game.num_agents)
    ]

    metrics = Metrics(
        num_agents = parameters.get("num_agents"),
        num_epochs = parameters.get("epochs"),
        train_episodes = parameters.get("episodes_per_epoch"),
        evaluate_episodes = parameters.get("num_maps"),
    )

    return parameters, environment, algorithms, metrics

def run(output_path, config=None):
    parameters, environment, algorithms, metrics = setup(config)
    solution_concept_name = parameters.get("solution_concept_class").__name__
    for epoch in tqdm(range(parameters.get("epochs"))):
        train_algorithms(parameters, environment, algorithms, metrics)
        evaluate_algorithms(parameters, environment, algorithms, metrics, solution_concept_name, epoch)

        metrics.set_epsilon(algorithms[0].epsilon)
        metrics.set_alpha(algorithms[0].alpha)
        for i in range(parameters.get("num_agents")):
            algorithms[i].epsilon = max(parameters.get("epsilon_decay")*algorithms[i].epsilon, parameters.get("epsilon_min"))
            algorithms[i].alpha = max(parameters.get("alpha_decay")*algorithms[i].alpha, parameters.get("alpha_min"))

        metrics.incr_epoch()

        metrics.serialize_metrics(output_path)


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
