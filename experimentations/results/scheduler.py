import os
import numpy as np


def run_exp(n_exp, **params):
    exp_name = n_exp + f" --n_items {params['n_items']} --n_reps {params['n_reps']} --n_tiers {params['t']} --test_subsets {params['n_test_subsets']} --budget {params['budget']} --alpha {params['alpha']} --pr {params['pr']} --n_evals {params['n_evals']} --output {params['output']}"
    print("RUNNING:", exp_name)
    home_dir = os.system("python " + exp_name)

OUTPUT = "CARD"
for alpha in np.linspace(0.1,1,5):
    for p in np.linspace(0.1,0.9,5):
        for n_tiers in range(3,12,3):
            run_exp("exp2.py",n_items = 5, n_reps = 3, t = n_tiers, n_test_subsets = 20, budget = 30, alpha = alpha, pr = p, n_evals = 5, output = OUTPUT)
