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


OUTPUT = "THETA_VS_NOTHETA_1"
for alpha in np.linspace(0.1,0.9,5):
    for p in np.linspace(0.1,0.9,5):
        for n_tiers in range(3,20,3):
            run_exp("exp3.py",n_items = 5, n_reps = 3, t = n_tiers, n_test_subsets = 10, budget = 25, alpha = alpha, pr = p, n_evals = 5, output = OUTPUT)
