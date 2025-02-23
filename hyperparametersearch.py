import os
import subprocess
from itertools import product

hyperparams = {
    'method': ['pmlp_gcn', 'pmlp_sgc', 'pmlp_appnp'], 
    'lr': [0.01, 0.1],
    'dropout': [0.2, 0.3, 0.5],
    'num_layers': [2, 4, 6],
    'hidden_channels': [32, 64, 128, 256],
    'weight_decay': [0.001, 0.01, 0.1],
}

combinations = list(product(*hyperparams.values()))

# Change dataset to run on a different one
dataset = 'wisconsin'
# Different datasets run on different protocols
protocol = 'supervised'
device = 0

# Change configuration to change between MLP [FFF], GNN [TTT], PMLP [FFT]
conv_tr = False
conv_va = False
conv_te = True

for comb in combinations:
    current_hyperparams = dict(zip(hyperparams.keys(), comb))

    command = f"python main.py --dataset {dataset} --method {current_hyperparams['method']} --protocol {protocol} --lr {current_hyperparams['lr']} " \
              f"--dropout {current_hyperparams['dropout']} --num_layers {current_hyperparams['num_layers']} " \
              f"--hidden_channels {current_hyperparams['hidden_channels']} --weight_decay {current_hyperparams['weight_decay']} " \
              f"--device {device}"

    if conv_tr:
        command += " --conv_tr"
    if conv_va:
        command += " --conv_va"
    if conv_te:
        command += " --conv_te"

    print(f"Running: {command}")

    subprocess.run(command, shell=True)