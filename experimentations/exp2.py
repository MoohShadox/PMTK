from PMTK.experimentations.experiments_toolkit import *
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n','--n_items', type=int, default = 4)
parser.add_argument('-r','--n_reps', type=int, default = 5)
parser.add_argument('-t','--n_tiers', type=int, default = 6)
parser.add_argument('-ts','--test_subsets', type=int, default = 20)
parser.add_argument('-b','--budget', type=int, default = 30)
parser.add_argument('-a','--alpha', type=float, default = 0.2)
parser.add_argument('-p','--pr', type=float, default = 0.1)
parser.add_argument('-ne','--n_evals', type=int, default = 5)
parser.add_argument('-o','--output', type=str, default = "OUT")

args = parser.parse_args()


n_items = args.n_items
n_reps = args.n_reps
t = args.n_tiers
n_test_subsets = args.test_subsets
budget = args.budget
alpha = args.alpha
p = args.pr
n_evals = args.n_evals

EXP_NAME = "exp2"
EXP_TITLE = EXP_NAME + f"_n{n_items}_r{n_reps}_t{t}_ns{n_test_subsets}_b{budget}_a{alpha}_p{p}"
OUTPUT_DIR = args.output

if OUTPUT_DIR == "OUT":
    dirs = os.listdir(".")
    i = 0
    while f"EXP_SET_{i}" in dirs:
        i = i + 1
    OUTPUT_DIR = f"EXP_SET_{i}"


Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

items = np.arange(n_items)
data = {
      "budget":[],
      "model":[],
      "correct":[],
      "wrong":[],
      "total":[],
      "n_pref_input":[],
      "n_pref_output":[],
      "theta_size":[],
      "theta_additivity":[],
      "n_items":[],
      "n_test_subsets":[],
      "n_tiers":[],
      "alpha":[],
      "p":[],
}


for reps in range(n_reps):
    RTD = Random_Tierlist_Decider(items, p = p, n_theta= int((2**n_items)*alpha)+1, n_tiers = t)
    TOF = TL_Objective_Function(items, RTD)
    TOF(EMPTY_SET)
    thetas = None
    for i in range(1,budget):
        s = sample_subset(items)
        cpt = 0
        while s in TOF.saved:
            cpt = cpt + 1
            s = sample_subset(items)
            if cpt == 100000:
                break

        TOF(s)
        preferences = TOF.relation()
        if len(preferences) == 0:
            continue
        theta = list(RTD.utilities.keys())
        print(f"\n i ={i},  reps= {reps}, p ={p}, alpha = {alpha}, t = {t}, theta={theta}, preferences = {len(preferences)}")
        ## Estimated theta
        for _ in range(n_evals):
            test_subsets = sample_subsets(items, n_subsets = n_test_subsets)

            clf = train_clf(svm.SVC, preferences, theta, kernel = "linear")

            card = train_cardinal(items, theta, TOF.relation())

            p3_o = predict_ordinal(items, TOF.relation(), theta, test_subsets)
            if type(p3_o) == "NoneType" or type(p3_o) == bool:
                continue
            p3 = p3_o - preferences

            p1_o = (predict_clf(items, clf, theta, test_subsets))
            p1 = p1_o - preferences
            
            p6_o = predict_cardinal(card, test_subsets)
            p6 = p6_o - preferences

            c_clf, w_clf, t_clf = preference_evaluation(p1, RTD)
            c_uni, w_uni, t_uni = preference_evaluation(p3, RTD)
            c_card, w_card, t_card = preference_evaluation(p6, RTD)

            data["budget"].append(i)
            data["model"].append("CARD")
            data["correct"].append(c_card)
            data["wrong"].append(w_card)
            data["total"].append(t_card)
            data["n_pref_input"].append(len(preferences))
            data["n_pref_output"].append(len(p6_o))
            data["theta_size"].append(len(theta))
            data["theta_additivity"].append(additivity(theta))
            data["n_tiers"].append(t)
            data["n_items"].append(n_items)
            data["alpha"].append(alpha)
            data["p"].append(p)
            data["n_test_subsets"].append(len(test_subsets))

            data["budget"].append(i)
            data["model"].append("CLF_UNION")
            data["correct"].append(c_clf)
            data["wrong"].append(w_clf)
            data["total"].append(t_clf)
            data["n_pref_input"].append(len(preferences))
            data["n_pref_output"].append(len(p1_o))
            data["theta_size"].append(len(theta))
            data["theta_additivity"].append(additivity(theta))
            data["n_tiers"].append(t)
            data["n_items"].append(n_items)
            data["n_test_subsets"].append(len(test_subsets))
            data["alpha"].append(alpha)
            data["p"].append(p)
            
            data["budget"].append(i)
            data["model"].append("ORD-UNION")
            data["correct"].append(c_uni)
            data["wrong"].append(w_uni)
            data["total"].append(t_uni)
            data["n_pref_input"].append(len(preferences))
            data["n_pref_output"].append(len(p3_o))
            data["theta_size"].append(len(theta))
            data["theta_additivity"].append(additivity(theta))
            data["n_tiers"].append(t)
            data["n_items"].append(n_items)
            data["n_test_subsets"].append(len(test_subsets))
            data["alpha"].append(alpha)
            data["p"].append(p)

            df = pd.DataFrame(data)
            df.to_csv("model_comparison.csv")
            df.to_csv(os.path.join(OUTPUT_DIR, EXP_TITLE))
            print(f"\n Reps:{reps}, p:{p}, alpha:{alpha}, t:{t}, i={i}")
            summary = pd.DataFrame(df)
            summary = summary.groupby(["budget", "model"]).mean().reset_index()
            print(summary)
            print("%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%")        

df.to_csv(os.path.join(OUTPUT_DIR, EXP_TITLE + "f.csv"))

