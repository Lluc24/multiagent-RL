from datetime import datetime
import wandb
import json
import subprocess
import argparse
import os

class Vars:
    path_to_script = None
    soluction_concept = None

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        config_path = os.path.join("logs", "configurations", f"config-{today}.json")
        metrics_path = os.path.join("logs", "metrics", f"metrics-{today}.json")

        # Save the configuration to a file
        with open(config_path, "w") as f:
            json.dump(dict(config), f, indent=4)

        # Call pogema script (from its env)
        subprocess.run([
            os.path.join("pogema-env", "bin", "python"),  # path to pogema-env Python
            Vars.path_to_script,  # script to run
            "--configuration", config_path,
            "--metrics", metrics_path,
            "--solution-concept", Vars.soluction_concept
        ])

        # Read metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        reward = metrics.pop("reward")
        epochs = metrics.pop("epochs")
        num_agents = metrics.pop("num_agents")
        # Log to wandb
        wandb.log({"reward": reward}, step=0)
        for epoch in range(epochs):
            for key, value in metrics.items():
                wandb.log({key: value[epoch]}, step=epoch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for running a sweep with WandB.')
    parser.add_argument("--name", type=str, default="JAL-GT", help="Name of the wandb project")
    parser.add_argument("--script", type=str, help="Path to the script to run the experiment with", required=True)
    parser.add_argument('--sweep', type=str, help='Path to the sweep (e.g. sweeps/sweep1.json)', required=True)
    parser.add_argument("--count", type=int, default=30, help="Number of runs to perform in the sweep")
    parser.add_argument("--solution-concept", type=str, help="Solution concept to be used for all agents", default="Pareto")
    args = parser.parse_args()

    logs_dir = "logs"
    metrics_dir = os.path.join(logs_dir, "metrics")
    configurations_dir = os.path.join(logs_dir, "configurations")
    for path in (logs_dir, metrics_dir, configurations_dir):
        if not os.path.exists(path):
            os.makedirs(path)

    Vars.path_to_script = args.script
    Vars.soluction_concept = args.solution_concept

    with open(args.sweep, "r") as f:
        sweep_config = json.load(f)

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=args.name)
    wandb.agent(sweep_id, train, count=args.count)
