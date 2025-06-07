from solution_concepts import ParetoSolutionConcept


class Parameters:
    def __init__(self, config=None):
        self.config = config

        # Default parameters if wandb_config is None
        self.renders_dir = "renders/"
        self.num_agents = 2  # Número de agentes
        self.map_size = 4  # Tamaño del mapa (4x4)
        self.num_maps = 10  # Número de mapas a entrenar y evaluar (se repiten si episodios > mapas)
        self.num_states = 16 * 16 * 4  # Número de estados posibles (4 bits para obstáculos, 4 bits para agentes, 2 bits para objetivo)
        self.epochs = 200  # Cada epoch es un entrenamiento de un número de episodios y una evaluación
        self.episodes_per_epoch = 10  # Número mínimo de episodios por epoch de entrenamiento
        self.max_episode_steps = 16  # Número máximo de pasos por episodio, se trunca si se excede
        self.obstacle_density = 0.1  # Probabilidad de tener un obstáculo en el mapa
        self.save_every = None # Frecuencia con que se guarda el SVG con la animación de la ejecución
        self.gamma = 0.95  # gamma
        self.learning_rate = 0.01  # alpha
        self.epsilon_max = 1  # epsilon inicial por epoch
        self.epsilon_min = 0.1  # cota mínima de epsilon
        self.epsilon_diff = (self.epsilon_max - self.epsilon_min) / self.episodes_per_epoch
        self.solution_concept_class = ParetoSolutionConcept
        self.num_actions = 5 # STAY, UP, DOWN, LEFT, RIGHT
        self.observation_radius = 1

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