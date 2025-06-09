# SID Laboratory: Multiagnet Reinforcement Learning

### Requirements
- Python 3.6+. This project has been made with Python 3.10.13

### Installation
For intalling the requirements, tweo virtual environments are recommended: one for the pogema and one for wandb. We
recommend working with two terminals, one for each virtual environment.

1. In one terminal, create a virtual environment for working with pogema, activate it, and install the requirements:
```bash
python3 -m venv pogema-env
source pogema-env/bin/activate
pip install -r requirements-pogema.txt
```

2. In another terminal, create a virtual environment for working with wandb, activate it, and install the requirements:
```bash
python3 -m venv wandb-env
source wandb-env/bin/activate
pip install wandb
```

## How to Replicate the Experiments

To reproduce the experiments described in the report, run the following commands:

### Experiment 1
On the second terminal (with wandb-env activated), run each commando below for experimenting
with the four different solution concepts:

#### 1. Pareto solution concept
```bash
python3 sweep_runner.py --script experiment1.py --sweep sweeps/sweep1.json --count=100 --solution-concept=Pareto
```

#### 2. Nash solution concept
```bash
python3 sweep_runner.py --script experiment1.py --sweep sweeps/sweep1.json --count=100 --solution-concept=Nash
```

#### 3. Welfare solution concept
```bash
python3 sweep_runner.py --script experiment1.py --sweep sweeps/sweep1.json --count=100 --solution-concept=Welfare
```

#### 4. Minimax solution concept
```bash
python3 sweep_runner.py --script experiment1.py --sweep sweeps/sweep1.json --count=100 --solution-concept=Minimax
```