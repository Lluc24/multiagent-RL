class Parameters:
    def __init__(self, config=None, solution_concept_class=None, **kwargs):
        self.config = config or {}

        # Parámetros que suelen venir en config (de wandb)
        self.gamma = 0.999
        self.epsilon_max = 1
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.7
        self.alpha_max = 0.01
        self.alpha_min = 0.01
        self.alpha_decay = 0.95

        # Parámetros que suelen fijarse por el usuario
        self.obstacle_density = 0.1
        self.num_agents = 2
        self.map_size = 4
        self.num_maps = 10
        self.epochs = 200
        self.episodes_per_epoch = 10
        self.max_episode_steps = 16
        self.observation_radius = 1

        # Parámetros necesariamente fijados por el usuario
        self.solution_concept_class = None
        self.renders_dir = "renders/"
        self.save_every = None

        # Parámetros fijados por el entorno
        self.num_actions = 5
        self.num_states = 16 * 16 * 4

        # Parámetros calculados
        self.total_executions = self.epochs * self.episodes_per_epoch

        # Primero aplicar config si existe
        for k, v in self.config.items():
            setattr(self, k, v)

        # Luego aplicar solución conceptual si la pasaron explícitamente
        if solution_concept_class is not None:
            self.solution_concept_class = solution_concept_class

        # Aplicar kwargs si los hubiera (opcional)
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Recalcular parámetros dependientes
        self.total_executions = self.epochs * self.episodes_per_epoch

    def get(self, attr):
        _, value = self._get_attr(attr)
        return value

    def _get_attr(self, attr):
        if attr in self.config:
            return "config", self.config[attr]
        elif hasattr(self, attr):
            return "self", getattr(self, attr)
        else:
            raise AttributeError(f"No attribute '{attr}'")

    def print(self):
        for attr in vars(self):
            if attr == 'config':
                continue
            owner, value = self._get_attr(attr)
            print(f"{attr}: ({owner}) {value}")
