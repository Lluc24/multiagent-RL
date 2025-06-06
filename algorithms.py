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
    def __init__(self, game: GameModel, gamma=0.95, learning_rate=0.99, lr_decay=1, seed=42, t_max=200):
        self.game = game
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.t_max = t_max
        self.policy = np.ones((self.game.num_states, self.game.num_actions)) /self.game.num_states
        np.random.seed(seed)

    def select_action(self, state, training=True):
        action_probabilities = self.policy_table[state]
        if training:
            return np.random.choice(np.arange(self.game.num_actions), p=action_probabilities)
        else:
            return np.argmax(action_probabilities)

    def update_policy(self, episode):
        states, actions, rewards = episode
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0

        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        loss = -np.sum(np.log(self.policy_table[states, actions]) * discounted_rewards) / len(states)
        
        policy_logits =np.log(np.maximum(self.policy_table, np.finfo(float).tiny))# Evitar log(0), si una celda es 0, establecerla a epsilon
        for t in range(len(states)):
            G_t = discounted_rewards[t]
            action_probs = np.exp(policy_logits[states[t]])
            action_probs /= np.sum(action_probs)
            policy_gradient = G_t * (1 - action_probs[actions[t]])
            policy_logits[states[t], actions[t]] += self.learning_rate * policy_gradient

        exp_logits = np.exp(policy_logits)
        self.policy_table = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return loss

    def learn_from_episode(self):
        state, _ = self.env.reset()
        episode = []
        done = False
        step = 0
        total_reward = 0
        while not done and step < self.t_max:
            action = self.select_action(state)
            next_state, reward, done, terminated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
            step += 1
        loss = self.update_policy(zip(*episode))
        self.learning_rate *= self.lr_decay
        return total_reward, loss

    # Ejecuta un episodio de prueba usando la política aprendida
    # y devuelve la recompensa total obtenida en ese episodio
    def test_episode(self):
        state, _ = self.env.reset()
        episode = []
        done = False
        step = 0
        total_reward = 0
        test_t_max = 200
        while not done and step < test_t_max:
            action = self.select_action(state, training=False)
            next_state, reward_test, done, terminated, _ = self.env.step(action)
            episode.append((state, action, reward_test))
            state = next_state
            total_reward += reward_test
            step += 1
        return total_reward
    
    def policy(self):
        policy = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy, self.policy_table

    def policy_probabilities(self):
        policy_probabilities = []  # Nombre de variable corregido
        estado_actual = 0
        estados_terminales = {47}
        for actions_s in self.policy_table:
            if estado_actual not in estados_terminales:
                a = np.argmax(actions_s)
                a_prob = actions_s[a]
                state_prob = float(np.sum(actions_s))
                dominant_prob = float(a_prob / state_prob)
                policy_probabilities.append(round(dominant_prob, 3))
            else:
                policy_probabilities.append(-0.25)
            estado_actual += 1
        return policy_probabilities
