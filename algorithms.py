import random

import numpy as np

from solution_concepts import SolutionConcept
from game_model import GameModel
import abc


class MARLAlgorithm(abc.ABC):
    @abc.abstractmethod
    def learn(self, joint_action, rewards, next_state: int, observations):
        pass

    @abc.abstractmethod
    def explain(self):
        pass

    @abc.abstractmethod
    def select_action(self, state):
        pass


class JALGT(MARLAlgorithm):
    def __init__(self, agent_id, game: GameModel, solution_concept: SolutionConcept,
                 gamma=0.95, alpha=0.5, epsilon=0.2, seed=42):
        self.agent_id = agent_id
        self.game = game
        self.solution_concept = solution_concept
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        # Q: N x S x AS
        self.q_table = np.zeros((self.game.num_agents, self.game.num_states,
                                 len(self.game.action_space)))
        # Política conjunta por defecto: distribución uniforme respecto
        # de las acciones conjuntas, para cada acción (pi(a | s))
        self.joint_policy = np.ones((self.game.num_agents, self.game.num_states,
                                     self.game.num_actions)) / self.game.num_actions
        self.metrics = {"td_error": []}

    def value(self, agent_id, state):
        value = 0
        for idx, joint_action in enumerate(self.game.action_space):
            payoff = self.q_table[agent_id][state][idx]
            joint_probability = np.prod(
                [
                    self.joint_policy[i][state][joint_action[i]]
                    for i in range(self.game.num_agents)
                ]
            )
            value += payoff * joint_probability
        return value

    def update_policy(self, agent_id, state):
        self.joint_policy[agent_id][state] = self.solution_concept.solution_policy(agent_id, state, self.game,
                                                                                   self.q_table)

    def learn(self, joint_action, rewards, state, next_state):
        joint_action_index = self.game.action_space_index[joint_action]
        for agent_id in range(self.game.num_agents):
            agent_reward = rewards[agent_id]
            agent_game_value_next_state = self.value(agent_id, next_state)
            agent_q_value = self.q_table[agent_id][state][joint_action_index]
            td_target = agent_reward + self.gamma * agent_game_value_next_state - agent_q_value
            self.q_table[agent_id][state][joint_action_index] += self.alpha * td_target
            self.update_policy(agent_id, state)
            # Guardamos el error de diferencia temporal para estadísticas posteriores
            self.metrics['td_error'].append(td_target)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def solve(self, agent_id, state):
        return self.joint_policy[agent_id][state]

    def select_action(self, state, train=True):
        if train:
            if self.rng.random() < self.epsilon:
                return self.rng.choice(range(self.game.num_actions))
            else:
                probs = self.solve(self.agent_id, state)
                np.random.seed(self.rng.randint(0, 10000))
                return np.random.choice(range(self.game.num_actions), p=probs)
        else:
            return np.argmax(self.solve(self.agent_id, state))

    def explain(self, state=0):
        return self.solution_concept.debug(self.agent_id, state, self.game, self.q_table)


# Clase ReinforceAgent
class ReinforceAgent(MARLAlgorithm):
    def __init__(self, game: GameModel, gamma=0.95, learning_rate=0.99, lr_decay=1, seed=42):
        self.game = game
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        # Adaptación: tabla de política para un solo agente
        self.policy_table = np.ones((self.game.num_states, self.game.num_actions)) / self.game.num_actions
        np.random.seed(seed)
        self.metrics = {"loss": []}  # Encara haig de pensar com adaptar

    # Pre: recibe lista de estados, acciones y recompensas a lo largo de un episodio.
    # Post: actualiza la política del agente y devuelve su pérdida
    def update_policy(self, states, actions, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0

        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        loss = -np.sum(np.log(self.policy_table[states, actions]) * discounted_rewards) / len(states)

        policy_logits = np.log(np.maximum(self.policy_table, np.finfo(float).tiny))  # Evitar log(0)
        for t in range(len(states)):
            G_t = discounted_rewards[t]
            action_probs = np.exp(policy_logits[states[t]])
            action_probs /= np.sum(action_probs)
            policy_gradient = G_t * (1 - action_probs[actions[t]])
            policy_logits[states[t], actions[t]] += self.learning_rate * policy_gradient

        exp_logits = np.exp(policy_logits)
        self.policy_table = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return loss

    # Pre: recibe 3 listas con las acciones, recompensas y estados del agente a lo largo de un episodio.
    # Post: guarda la pérdida y el reward acumulado del episodio
    def learn(self, actions, rewards, states):
        loss = self.update_policy(states, actions, rewards)
        total_reward = sum(rewards)
        self.learning_rate *= self.lr_decay
        # Guardar métricas
        self.metrics["loss"].append(loss)

    # Pre: recibe el estado actual del agente.
    # Post: selecciona una acción basada en la política aprendida
    def select_action(self, state, train=True):
        action_probabilities = self.policy_table[state]
        if train:
            return np.random.choice(np.arange(self.game.num_actions), p=action_probabilities)
        else:
            return np.argmax(action_probabilities)

    # It is not clear to us what the specific intent of this method is in the superclass MARLAlgorithm, therefore we decided to return the policy_table as an implementation is needed since it is an abstract method
    def explain(self):
        return self.policy_table
    
    def get_alpha(self):
        return self.learning_rate