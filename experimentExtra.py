import os
from tqdm import tqdm
from algorithms import ReinforceAgent
from environment import Environment, obs_to_state
from game_model import GameModel
import shutil
import argparse
import json
from parameters import Parameters
from metrics import Metrics

def train_algorithms(parameters, env_manager, algorithms, metrics):
    """Train algorithms over multiple episodes."""
    for ep in range(parameters.get("episodes_per_epoch")):
        # Initialize episode
        env, states, _, _ = initialize_episode(env_manager, parameters, ep)
        
        # Run episode and collect trajectories
        episode_states, episode_actions, episode_rewards, terminated, truncated = run_episode_loop(
            env, algorithms, metrics, parameters, states
        )
        
        # Learn from episode
        learn_from_episode(algorithms, episode_actions, episode_rewards, episode_states, metrics, parameters)
        
        # Update success rate metrics
        metrics.add_to_success_rates(terminated, truncated)

#===============Auxiliar Functions for train_algorithms===============
def initialize_episode(env_manager, parameters, ep):
    """Initialize a new episode with environment and agent states."""
    env = env_manager.new_env(seed=ep % parameters.get("num_maps"))
    observations, _ = env.reset()
    terminated = [False for _ in range(parameters.get("num_agents"))]
    truncated = [False for _ in range(parameters.get("num_agents"))]
    states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]
    return env, states, terminated, truncated

def initialize_trajectory_storage(num_agents):
    """Initialize storage for episode trajectories."""
    episode_states = [[] for _ in range(num_agents)]
    episode_actions = [[] for _ in range(num_agents)]
    episode_rewards = [[] for _ in range(num_agents)]
    return episode_states, episode_actions, episode_rewards

def store_step_data(episode_states, episode_actions, episode_rewards, states, actions, rewards, num_agents):
    """Store trajectory data for the current step."""
    for i in range(num_agents):
        episode_states[i].append(states[i])
        episode_actions[i].append(actions[i])
        episode_rewards[i].append(rewards[i])

def run_episode_loop(env, algorithms, metrics, parameters, initial_states):
    """Run the main episode loop and return trajectory data."""
    states = initial_states
    terminated = [False for _ in range(parameters.get("num_agents"))]
    truncated = [False for _ in range(parameters.get("num_agents"))]

    episode_states, episode_actions, episode_rewards = initialize_trajectory_storage(parameters.get("num_agents"))

    while not all(terminated) and not all(truncated):
        # Select actions
        actions = tuple(algorithms[i].select_action(states[i]) for i in range(parameters.get("num_agents")))

        # Execute actions in environment
        observations, rewards, terminated, truncated, _ = env.step(actions)

        # Convert observations to states for next iteration
        new_states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]

        # Store trajectory data
        store_step_data(episode_states, episode_actions, episode_rewards, states, actions, rewards, parameters.get("num_agents"))

        states = new_states

        # Update metrics
        metrics.add_rewards(rewards)
        metrics.add_steps(terminated, truncated)
    
    return episode_states, episode_actions, episode_rewards, terminated, truncated

def learn_from_episode(algorithms, episode_actions, episode_rewards, episode_states, metrics, parameters):
    """Update algorithms based on episode trajectories."""
    for i in range(parameters.get("num_agents")):
        algorithms[i].learn(episode_actions[i], episode_rewards[i], episode_states[i])
        
        # Update loss metrics if available
        if "loss" in algorithms[i].metrics and algorithms[i].metrics["loss"]:
            metrics.add_loss([algorithms[i].metrics["loss"][-1] for i in range(parameters.get("num_agents"))])

#============End of auxiliar Functions for train_algorithms============


def evaluate_algorithms(parameters, env_manager, algorithms, metrics, solution_concept_name, epoch):
    for ep in range(parameters.get("num_maps")):
        env = env_manager.new_env(seed=ep)  # Reaprovechamos mapas del entrenamiento
        observations, _ = env.reset() # Ignored return is infos
        terminated = [False for _ in range(parameters.get("num_agents"))]
        truncated = [False for _ in range(parameters.get("num_agents"))]
        states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]
        while not all(terminated) and not all(truncated):  # Hasta que acabe el episodio
            actions = tuple(
                            algorithms[i].select_action(states[i], train=False) for i in range(parameters.get("num_agents"))
                            )
            observations, rewards, terminated, truncated, _ = env.step(actions) # Ignored return is infos
            states = [obs_to_state(observations[i]) for i in range(parameters.get("num_agents"))]

            # no changes with respect to experiment1
            metrics.add_rewards(rewards, training=False)
            metrics.add_steps(terminated, truncated, training=False)
        metrics.add_to_success_rates(terminated, truncated, training=False)
        # Guardamos animaciones
        env_manager.save_animations(solution_concept_name, ep, epoch)

def setup(wandb_config):
    parameters = Parameters(wandb_config)
    parameters.print()

    # No changes with respect to experiment1
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

    # We use REINFORCE instead of using JAL-GT as in experiment 1
    algorithms = [
        ReinforceAgent(
            game = game,
            gamma = parameters.get("gamma"),
            learning_rate = parameters.get("alpha"),
            lr_decay = parameters.get("alpha_decay"),
            seed = i # The method for establishing the seed in experiment1 is maintained
        )
        for i in range(game.num_agents)
    ]

    # No changes were made here with respect to experiment1
    metrics = Metrics(
        num_agents = parameters.get("num_agents"),
        num_epochs = parameters.get("epochs"),
        train_episodes = parameters.get("episodes_per_epoch"),
        evaluate_episodes = parameters.get("num_maps"),
    )

    return parameters, environment, algorithms, metrics

def run(output_path, config=None):
    parameters, environment, algorithms, metrics = setup(config)
    for epoch in tqdm(range(parameters.get("epochs"))):
        train_algorithms(parameters, environment, algorithms, metrics)
        evaluate_algorithms(parameters, environment, algorithms, metrics, "REINFORCE", epoch) # "REINFORCE" is used where the name of the solution concept used to be placed in experiment1

        metrics.set_alpha(algorithms[0].get_alpha())
        # Alpha decay is applied by the function learn at the end of each episode, in contrast with JALGT, REINFORCE allows for this. It is done this way because it is more clean and transparent

        metrics.incr_epoch()

        metrics.serialize_metrics(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running an experiment.')
    parser.add_argument('--configuration', type=str, help='Path to the configuration file', required=False)
    parser.add_argument("--metrics", type=str, help="Path for the output metrics", required=True)
    args = parser.parse_args()
    # Load the configuration file as a dictionary (it is a JSON file)
    if args.configuration is None: # We are not using wandb
        run(args.metrics)
    else: # We are in wandb mode
        with open(args.configuration, "r") as f:
            config = json.load(f)
        run(args.metrics, config)
