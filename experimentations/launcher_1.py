import os
import numpy as np


def run_exp(n_exp, **params):
    exp_name = n_exp + f" --n_items {params['n_items']} --n_reps {params['n_reps']} --n_tiers {params['t']} --test_subsets {params['n_test_subsets']} --budget {params['budget']} --alpha {params['alpha']} --pr {params['pr']} --n_evals {params['n_evals']} --output {params['output']}"
    print("RUNNING:", exp_name)
    EXP_NAME = n_exp.split(".")[0]
    EXP_TITLE = EXP_NAME + f"_n{params['n_items']}_r{params['n_reps']}_t{params['t']}_ns{params['n_test_subsets']}_b{params['budget']}_a{params['alpha']}_p{params['pr']}f.csv"
    if not params['output'] in os.listdir("..") or not EXP_TITLE in os.listdir(params["output"]):
        home_dir = os.system("python " + exp_name)
    else:
        print("SKIPPING: ", exp_name)


OUTPUT = "vTHETA_03_03_9"

n_tiers = 9
alpha = 0.3
p = 0.3
run_exp("exp1.py",n_items = 4, n_reps = 10, t = n_tiers, n_test_subsets = 10, budget = 25, alpha = alpha, pr = p, n_evals = 5, output = OUTPUT)

OUTPUT = "vTHETA_01_01_9"

n_tiers = 9
alpha = 0.1
p = 0.1
run_exp("exp1.py",n_items = 4, n_reps = 10, t = n_tiers, n_test_subsets = 10, budget = 25, alpha = alpha, pr = p, n_evals = 5, output = OUTPUT)
