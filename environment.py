from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
import numpy as np
import os

ON_TARGET = "finish"
RENDER_MODE = None
STATIC = False
SHOW_AGENTS = True
EGOCENTRIC_IDX = None
SHOW_BORDERS = True
SHOW_LINES = True

def obs_to_state(obs):
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]

    # Representación del objetivo:
    #  Ocupa 2 bits
    #  0 si el objetivo está arriba, diagonal arriba-izquierda o diagonal arriba-derecha
    #  1 si el objetivo está abajo, diagonal abajo-izquierda o diagonal abajo-derecha
    #  2 si el objetivo está a la izquierda (no en diagonal)
    #  3 si el objetivo está a la derecha (no en diagonal)
    target = np.max(matrix_target[2]) * 1 + \
             matrix_target[1][0] * 2 + matrix_target[1][2] * 3

    # Representación de los obstáculos:
    #  Shift de 2^6, ocupando 4 bits
    #  2^9 si hay un obstáculo arriba (no diagonal)
    #  2^8 si hay un obstáculo a la izquierda (no diagonal)
    #  2^7 si hay un obstáculo a la derecha (no diagonal)
    #  2^6 si hay un obstáculo abajo (no diagonal)
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + \
                matrix_obstacles[1][0] * 2 ** 8 + \
                matrix_obstacles[1][2] * 2 ** 7 + \
                matrix_obstacles[2][1] * 2 ** 6

    # Representación de los otros agentes:
    #  Shift de 2^2, ocupando 4 bits
    #  2^5 si hay un agente arriba (no diagonal)
    #  2^4 si hay un agente a la izquierda (no diagonal)
    #  2^3 si hay un agente a la derecha (no diagonal)
    #  2^2 si hay un agente abajo (no diagonal)
    agents = matrix_agents[0][1] * 2 ** 5 + \
             matrix_agents[1][0] * 2 ** 4 + \
             matrix_agents[1][2] * 2 ** 3 + \
             matrix_agents[2][1] * 2 ** 2

    return int(obstacles + agents + target)

class Environment:
    def __init__(self, num_agents, map_size, obstacle_density, max_episode_steps, obs_radius, renders_directory, save_every):
        self.num_agents = num_agents
        self.map_size = map_size
        self.obstacle_density = obstacle_density
        self.max_episode_steps = max_episode_steps
        self.obs_radius = obs_radius
        self.renders_directory = renders_directory
        self.save_every = save_every
        self.env = None

    def new_env(self, seed):
        grid_config = GridConfig(
            num_agents=self.num_agents,
            size=self.map_size,
            density=self.obstacle_density,
            seed=seed,
            max_episode_steps=self.max_episode_steps,
            obs_radius=self.obs_radius,
            on_target=ON_TARGET,
            render_mode=RENDER_MODE
        )
        animation_config = AnimationConfig(
            directory=self.renders_directory,  # Dónde se guardarán las imágenes
            static=STATIC,
            show_agents=SHOW_AGENTS,
            egocentric_idx=EGOCENTRIC_IDX,  # Punto de vista
            save_every_idx_episode=self.save_every,  # Guardar cada save_every episodios
            show_border=SHOW_BORDERS,
            show_lines=SHOW_LINES
        )
        env = pogema_v0(grid_config)
        env = AnimationMonitor(env, animation_config=animation_config)
        wrapped_env = RewardWrapper(env)  # Añadimos nuestra función de recompensa
        self.env = wrapped_env
        return wrapped_env

    def save_animations(self, solution_concept_name, ep, epoch):
        for agent_i in range(self.num_agents):
            filename = f"{solution_concept_name}-map{ep}-agent{agent_i}-epoch{epoch}.svg"
            path = os.path.join(self.renders_directory, filename)
            self.env.save_animation(
                path,
                AnimationConfig(egocentric_idx=agent_i, show_border=SHOW_BORDERS, show_lines=SHOW_LINES)
            )

class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, joint_action):
        # En caso de que queráis utilizar las observaciones anteriores, utilizad este objeto:
        previous_observations = self.env.unwrapped._obs()

        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0:  # Penalización por tardar más en llegar
                    rewards[i] = rewards[i] - 0.01
        return observations, rewards, terminated, truncated, infos