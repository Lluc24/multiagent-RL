import os
from tqdm import tqdm
from algorithms import JALGT
from environment import Environment, obs_to_state
from game_model import GameModel
import shutil
import argparse
import json
from paremeters import Parameters
from solution_concepts import ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept, MinimaxSolutionConcept

def create_metrics(num_agents):
    return {
        "num_agents": num_agents,
        "train_rewards": [0.0] * num_agents,
        "evaluate_rewards": [0.0] * num_agents,
        "train_success_rates": [0] * num_agents,
        "evaluate_success_rates": [0] * num_agents,
        "train_steps": [0] * num_agents,
        "evaluate_steps": [0] * num_agents,
    }
def add_success_rates(metrics, terminated, truncated, training=True):
    for i in range(metrics["num_agents"]):
        if terminated[i] and not truncated[i]:
            if training:
                metrics["train_success_rates"][i] += 1
            else:
                metrics["evaluate_success_rates"][i] += 1
def add_steps(metrics, terminated, truncated, training=True):
    for i in range(metrics["num_agents"]):
        if not terminated[i] and not truncated[i]:
            if training:
                metrics["train_steps"][i] += 1
            else:
                metrics["evaluate_steps"][i] += 1

# Función para guardar en JSON
def serialize_metrics(metrics, path):
    for i in range(metrics["num_agents"]):
        if metrics["evaluate_steps"][i] > 0:
            metrics["evaluate_rewards"][i] /= metrics["evaluate_steps"][i]
        else:
            metrics["evaluate_rewards"][i] = 0.0

        if metrics["train_steps"][i] > 0:
            metrics["train_rewards"][i] /= metrics["train_steps"][i]
        else:
            metrics["train_rewards"][i] = 0.0

    #with open(path, "w") as f:
    #    json.dump(metrics, f, indent=4)
    print(metrics)

def add_rewards(metrics, rewards, training=True):
    for i in range(metrics["num_agents"]):
        if training:
            metrics["train_rewards"][i] += rewards[i]
        else:
            metrics["evaluate_rewards"][i] += rewards[i]

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
            print(rewards)
        # Guardamos animaciones
        env_manager.save_animations(solution_concept_name, ep, epoch)

def evaluate_on_unseen_maps(parameters, env_manager, algorithms, metrics, solution_concept_name, output_path):
    reward_sum = 0
    succes_rate = 0
    num_unseen_maps = parameters.get("num_maps")  # o cualquier número que quieras
    offset_seed = 0  # Un offset suficientemente grande para evitar solapamiento con semillas de entrenamiento
    for ep in range(num_unseen_maps):
        env = env_manager.new_env(seed=offset_seed + ep)
        observations, infos = env.reset()
        terminated = [False for _ in range(parameters.get("num_agents"))]
        truncated = [False for _ in range(parameters.get("num_agents"))]
        states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]
        while not all(terminated) and not all(truncated):
            actions = tuple(algorithms[i].select_action(states[i], train=False) for i in range(parameters.get("num_agents")))
            observations, rewards, terminated, truncated, infos = env.step(actions)
            states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]
            reward_sum += sum(rewards)
        for i in range(parameters.get("num_agents")):
            if terminated[i] and not truncated[i]:
                succes_rate += 1
            #print("Rewards step:", rewards)
            # Si quieres llevar métricas separadas, deberías modificar aquí
            # metrics.add_unseen_rewards(...) (necesitarías adaptar la clase Metrics)
        # Si quieres guardar animaciones de mapas no vistos:
        env_manager.save_animations(solution_concept_name, ep, epoch=0)

    return reward_sum/num_unseen_maps/parameters.get("num_agents"), succes_rate/num_unseen_maps/parameters.get("num_agents")


def setup(wandb_config, solution_concept):
    if not isinstance(solution_concept, list):
        raise ValueError("Solution concept must be a list of strings.")
    parameters = Parameters(
        config=wandb_config,
        gamma=0.999,
        epsilon_min=0.3,
        epsilon_decay=0.7,
        alpha_min=0.01,
    )
    parameters.solution_concept_class = []
    for i in range(parameters.get("num_agents")):
        if i >= len(solution_concept):
            name = solution_concept[-1]
        else:
            name = solution_concept[i]
        if name == "Pareto":
            solution_concept_class = ParetoSolutionConcept
        elif name == "Nash":
            solution_concept_class = NashSolutionConcept
        elif name == "Welfare":
            solution_concept_class = WelfareSolutionConcept
        elif name == "Minimax":
            solution_concept_class = MinimaxSolutionConcept
        else:
            raise ValueError(f"Unknown solution concept: {solution_concept}")
        parameters.solution_concept_class.append(solution_concept_class)

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
            solution_concept=parameters.get("solution_concept_class")[i](),
            epsilon=parameters.get("epsilon_max"),
            gamma=parameters.get("gamma"),
            alpha=parameters.get("alpha_max"),
            seed=i
        )
        for i in range(game.num_agents)
    ]

    return parameters, environment, algorithms

def run(output_path, solution_concept, config=None):
    parameters, environment, algorithms = setup(config, solution_concept)
    metrics = create_metrics(parameters.get("num_agents"))
    solution_concept_class = parameters.get("solution_concept_class")
    for epoch in tqdm(range(parameters.get("epochs"))):
        train_algorithms(parameters, environment, algorithms, metrics)
        #evaluate_algorithms(parameters, environment, algorithms, metrics, solution_concept_class, epoch)
       
        
        for i in range(parameters.get("num_agents")):
            algorithms[i].epsilon = max(parameters.get("epsilon_decay")*algorithms[i].epsilon, parameters.get("epsilon_min"))
            algorithms[i].alpha = max(parameters.get("alpha_decay")*algorithms[i].alpha, parameters.get("alpha_min"))

    results = [
        evaluate_on_unseen_maps(parameters, environment, algorithms, metrics, solution_concept_class, output_path)
        for _ in range(parameters.get("episodes_per_epoch"))
    ]  

    avg_reward_unseen = sum(r for r, _ in results) / len(results)
    total_success_rate = sum(s for _, s in results) / len(results)
    print(avg_reward_unseen)
    print (total_success_rate)

    # Guardar JSON resumen
    summary = {
        "reward": avg_reward_unseen,
        "success_rate": total_success_rate
    }
    with open(os.path.join(output_path, "metric.json"), "w") as f:
        json.dump(summary, f, indent=4)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running an experiment.')
    parser.add_argument('--configuration', type=str, help='Path to the configuration file', required=False)
    parser.add_argument("--metrics", type=str, help="Path for the output metrics", required=True)
    parser.add_argument("--solution-concept", nargs='+', help="Solution concept to be used by each agent. Specify only one if all agents use the same", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.metrics):
        os.makedirs(args.metrics)
    # Load the configuration file as a dictionary (it is a JSON file)
    if args.configuration is None:
        output_file = os.path.join(args.metrics, "metrics.json")
        run(args.metrics, args.solution_concept)
    else:
        with open(args.configuration, "r") as f:
            config = json.load(f)
        output_path = os.path.join(args.metrics, f"eval.json")  
        run(output_path, args.solution_concept, config)
