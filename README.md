# SID Laboratory: Multiagnet Reinforcement Learning

### Requirements
- Python 3.6+. This project has been made with Python 3.10.13

### Installation
For installing the requirements, two virtual environments are recommended: one for the pogema and one for wandb. We
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
On the second terminal (with wandb-env activated), run each command below for experimenting
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

### Experiment 3
On the second terminal (with wandb-env activated), run each command below for experimenting
with the four different solution concepts:
```bash
 python3 sweep_runner.py --script experiment3.py --sweep sweeps/sweep1.json --count=100 --solution-concept=Pareto
 ```

 ### Experiment 4


To prove the generalization with a map 8x8 we execute the command bellow:
```bash
 python3 sweep_runner.py --script experiment4.py --sweep sweeps/sweep4.json --count=1 --solution-concept=Pareto
 ```

To prove the generalization with 3 agents we execute the command bellow:
```bash
 python3 sweep_runner.py --script experiment4.py --sweep sweeps/sweep5.json --count=1 --solution-concept=Pareto
 ```

 ### Experiment 5

 On the first terminal we exevcute this command, we can find the results at ./NewMaps/metrics.json

```bash
 python experiment5.py --metrics newMaps --solution-concept Pareto
```

### Experiment Extra: training with REINFORCE
To execute experiments with reinforce we use the sweep `sweepExtra_1.json` and script `experimentExtra.py`. It is executed in wandb this way:
```bash
python3 sweep_runner.py --script experimentExtra.py --sweep sweeps/sweepExtra_1.json --count=126
```
