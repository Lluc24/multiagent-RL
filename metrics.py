import numpy as np
import json

class Metrics:
    def __init__(self, num_agents, num_epochs, train_episodes, evaluate_episodes):
        self.epoch = 0
        self.num_agents = num_agents
        self.num_epochs = num_epochs
        self.train_episodes = train_episodes
        self.evaluate_episodes = evaluate_episodes
        self.agent_train_rewards = np.zeros((num_epochs, num_agents))
        self.agent_evaluate_rewards = np.zeros((num_epochs, num_agents))
        self.agent_td_errors = np.zeros((num_epochs, num_agents))
        self.agent_loss = np.zeros((num_epochs, num_agents))
        self.agent_train_success_rates = np.zeros((num_epochs, num_agents))
        self.agent_evaluate_success_rates = np.zeros((num_epochs, num_agents))
        self.agent_train_steps = np.zeros((num_epochs, num_agents))
        self.agent_evaluate_steps = np.zeros((num_epochs, num_agents))
        self.alpha = np.empty(num_epochs)
        self.epsilon = np.empty(num_epochs)

    def add_rewards(self, rewards, training=True):
        for i in range(self.num_agents):
            if training:
                self.agent_train_rewards[self.epoch, i] += rewards[i]
            else:
                self.agent_evaluate_rewards[self.epoch, i] += rewards[i]

    def add_td_errors(self, td_errors):
        for i in range(self.num_agents):
            self.agent_td_errors[self.epoch, i] += td_errors[i]
            
    def add_loss(self, loss):
        for i in range(self.num_agents):
            self.agent_loss[self.epoch, i] = loss[i] 
            #Only one loss per epoch as it is calculated at the end of epoch, therefore we don't add as in td_error, we assign

    def add_to_success_rates(self, terminated, truncated, training=True):
        for i in range(self.num_agents):
            if terminated[i] and not truncated[i]:
                if training:
                    self.agent_train_success_rates[self.epoch, i] += 1
                else:
                    self.agent_evaluate_success_rates[self.epoch, i] += 1

    def add_steps(self, terminated, truncated, training=True):
        for i in range(self.num_agents):
            if not truncated[i] and not terminated[i]:
                if training:
                    self.agent_train_steps[self.epoch, i] += 1
                else:
                    self.agent_evaluate_steps[self.epoch, i] += 1

    def set_epsilon(self, epsilon):
        self.epsilon[self.epoch] = epsilon

    def set_alpha(self, alpha):
        self.alpha[self.epoch] = alpha

    def incr_epoch(self):
        self.epoch += 1

    def serialize_metrics(self, path):
        """Serializes all metrics to a JSON file.
        
        Preconditions:
            - path is a valid file path with write permissions
            - All metrics have been properly tracked during training/evaluation
            - self.epoch > 0 (at least one epoch has been completed)
            - All metrics arrays have been initialized with the correct dimensions
        
        Postconditions:
            - A JSON file is created at the specified path
            - The JSON file contains all per-agent metrics normalized by episode count
            - Average metrics across all agents are included
            - The final evaluation reward is calculated and stored
            - All array data is converted to Python lists for JSON serialization
        
        Args:
            path (str): The file path where the JSON metrics will be saved
        """
        metrics = {}
        for i in range(self.num_agents):
            metrics[f"agent_{i}_train_rewards"] = (self.agent_train_rewards[:, i]/self.train_episodes).tolist()
            metrics[f"agent_{i}_evaluate_rewards"] = (self.agent_evaluate_rewards[:, i]/self.evaluate_episodes).tolist()
            metrics[f"agent_{i}_td_errors"] = (self.agent_td_errors[:, i]/self.train_episodes).tolist()
            metrics[f"agent_{i}_loss"] = (self.agent_loss[:, i]/self.train_episodes).tolist()
            metrics[f"agent_{i}_train_success_rates"] = (self.agent_train_success_rates[:, i]/self.train_episodes).tolist()
            metrics[f"agent_{i}_evaluate_success_rates"] = (self.agent_evaluate_success_rates[:, i]/self.evaluate_episodes).tolist()
            metrics[f"agent_{i}_train_steps"] = (self.agent_train_steps[:, i]/self.train_episodes).tolist()
            metrics[f"agent_{i}_evaluate_steps"] = (self.agent_evaluate_steps[:, i]/self.evaluate_episodes).tolist()

        metrics["alpha"] = self.alpha.tolist()
        metrics["epsilon"] = self.epsilon.tolist()

        # train rewards should be the average over all agents, and divided by the number of episodes
        metrics["average_train_rewards"] = (np.sum(self.agent_train_rewards, axis=1) / self.num_agents / self.train_episodes).tolist()
        metrics["average_evaluate_rewards"] = (np.sum(self.agent_evaluate_rewards, axis=1) / self.num_agents / self.evaluate_episodes).tolist()
        metrics["average_td_errors"] = (np.sum(self.agent_td_errors, axis=1) / self.num_agents / self.train_episodes).tolist()
        metrics["average_loss"] = (np.sum(self.agent_loss, axis=1) / self.num_agents / self.train_episodes).tolist()
        metrics["average_train_success_rates"] = (np.sum(self.agent_train_rewards, axis=1) / self.num_agents / self.train_episodes).tolist()
        metrics["average_evaluate_success_rates"] = (np.sum(self.agent_evaluate_success_rates, axis=1) / self.num_agents / self.evaluate_episodes).tolist()
        metrics["average_train_steps"] = (np.sum(self.agent_train_steps, axis=1) / self.num_agents / self.train_episodes).tolist()
        metrics["average_evaluate_steps"] = (np.sum(self.agent_evaluate_steps, axis=1) / self.num_agents / self.evaluate_episodes).tolist()

        metrics["reward"] = (np.sum(metrics["average_evaluate_rewards"]) / self.num_epochs).item()

        metrics["epochs"] = self.num_epochs

        metrics["num_agents"] = self.num_agents

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
