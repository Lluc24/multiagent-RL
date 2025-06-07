from datetime import datetime
import wandb
import json
import subprocess
import argparse
import os

class Files:
    path_to_script = None

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
            Files.path_to_script,  # script to run
            "--configuration", config_path,
            "--metrics", metrics_path,
        ])

        # Read metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Log to wandb
        wandb.log({"reward": metrics["reward"]})
        for epoch in range(len(metrics["td_error_per_epoch"])):
            wandb.log({
                "td_error_per_epoch": metrics["td_error_per_epoch"][epoch],
                "train_reward_per_epoch": metrics["train_reward_per_epoch"][epoch],
                "evaluation_reward_per_epoch": metrics["evaluation_reward_per_epoch"][epoch],
            })



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for running a sweep with WandB.')
    parser.add_argument("--name", type=str, default="JAL-GT", help="Name of the wandb project")
    parser.add_argument("--script", type=str, help="Path to the script to run the experiment with", required=True)
    parser.add_argument('--sweep', type=str, help='Path to the sweep (e.g. sweeps/sweep1.json)', required=True)
    parser.add_argument("--count", type=int, default=30, help="Number of runs to perform in the sweep")
    args = parser.parse_args()

    logs_dir = "logs"
    metrics_dir = os.path.join(logs_dir, "metrics")
    configurations_dir = os.path.join(logs_dir, "configurations")
    for path in (logs_dir, metrics_dir, configurations_dir):
        if not os.path.exists(path):
            os.makedirs(path)

    Files.path_to_script = args.script

    with open(args.sweep, "r") as f:
        sweep_config = json.load(f)

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=args.name)
    wandb.agent(sweep_id, train, count=args.count)
