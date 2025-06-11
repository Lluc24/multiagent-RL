class Parameters:
    def __init__(self, config=None, solution_concept_class=None):
        self.config = config

        # Parámetros que suelen venir en config (de wandb)
        self.gamma = 0.95  # gamma
        self.epsilon_max = 1  # epsilon inicial por epoch
        self.epsilon_min = 0.1  # cota mínima de epsilon
        self.epsilon_decay = 0.99
        self.alpha_max = 0.01  # alpha
        self.alpha_min = 0.001
        self.alpha_decay = 0.99

        # Parámetros que suelen fijarse por el usuario
        self.obstacle_density = 0.1  # Probabilidad de tener un obstáculo en el mapa
        self.num_agents = 2  # Número de agentes
        self.map_size = 4  # Tamaño del mapa (4x4)
        self.num_maps = 10  # Número de mapas a entrenar y evaluar (se repiten si episodios > mapas)
        self.epochs = 200  # Cada epoch es un entrenamiento de un número de episodios y una evaluación
        self.episodes_per_epoch = 10  # Número mínimo de episodios por epoch de entrenamiento
        self.max_episode_steps = 16  # Número máximo de pasos por episodio, se trunca si se excede
        self.observation_radius = 1

        # Parámetros necesariamente fijados por el usuario
        self.solution_concept_class = solution_concept_class  # Nombre del concepto de solución a utilizar
        self.renders_dir = "renders/"
        self.save_every = None  # Frecuencia con que se guarda el SVG con la animación de la ejecución

        # Parámetros fijados por el entorno
        self.num_actions = 5  # STAY, UP, DOWN, LEFT, RIGHT
        self.num_states = 16 * 16 * 4  # Número de estados posibles (4 bits para obstáculos, 4 bits para agentes, 2 bits para objetivo)

        # Parámetros calculados
        self.total_executions = self.epochs * self.episodes_per_epoch  # Total de episodios a ejecutar

    def get(self, attr):
        _, value = self._get_attr(attr)
        return value

    def _get_attr(self, attr):
        if self.config is not None and attr in self.config:
            value = self.config[attr]
            return "config", value
        elif hasattr(self, attr):
            value = getattr(self, attr)
            return "self", value
        else:
            raise AttributeError("No attribute '" + attr + "'")

    def print(self):
        for attr in vars(self):
            owner, value = self._get_attr(attr)
            print(f"{attr}: ({owner}) {value}")