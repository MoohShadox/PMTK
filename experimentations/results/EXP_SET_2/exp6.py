from PMTK.experimentations.experiments_toolkit import *
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-ni','--n_items', type=int, default = 4)
parser.add_argument('-b','--budget', type=int, default = 30)
parser.add_argument('-nt','--n_tiers', type=int, default = 15)

args = parser.parse_args()


n_items = args.n_items
n_tiers = args.n_tiers

EXP_TITLE = f"EXP6_{n_items}_{n_tiers}_2"

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
      "n_tiers":[],
      "alpha":[],
      "p":[],
      "n_items":[],
}

t = n_tiers
for alpha in np.linspace(0.1,0.6,5):
    for p in np.linspace(0.1,1,5):
      for reps in range(3):
        print("%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%")
        RTD = Random_Tierlist_Decider(items, p = p, n_theta= int((2**n_items)*alpha), n_tiers = t)
        theta = list(RTD.utilities.keys())
        TOF = TL_Objective_Function(items, RTD)
        for i in range(1, 2**n_items, 1):
          s = sample_subset(items)
          cpt = 0
          while s in TOF.saved:
            cpt = cpt + 1
            s = sample_subset(items)
            if cpt == 10000:
              break
          TOF(s)
          preferences = TOF.relation()
          if len(preferences) == 0:
            continue
          clf = train_clf(svm.SVC, preferences, theta, kernel = "linear")
          test_subsets = get_all_k_sets(items, len(items))
          p1_o = (predict_clf(items, clf, theta, test_subsets))
          p1 = p1_o - preferences
          p2_o = predict_ordinal(items, TOF.relation(), theta, test_subsets)
          if type(p2_o) == bool and not p2_o:
              break
          else:
              p2 = p2_o - preferences
          c_ord, w_ord, t_ord = preference_evaluation(p2, RTD)
          c_clf, w_clf, t_clf = preference_evaluation(p1, RTD)

          data["budget"].append(i)
          data["model"].append("CLF")
          data["correct"].append(c_clf)
          data["wrong"].append(w_clf)
          data["total"].append(t_clf)
          data["n_pref_input"].append(len(preferences))
          data["n_pref_output"].append(len(p1_o))
          data["theta_size"].append(len(theta))
          data["theta_additivity"].append(additivity(theta))
          data["n_tiers"].append(t)
          data["alpha"].append(alpha)
          data["p"].append(p)
          data["n_items"].append(n_items)

          data["budget"].append(i)
          data["model"].append("ORD")
          data["correct"].append(c_ord)
          data["wrong"].append(w_ord)
          data["total"].append(t_ord)
          data["n_pref_input"].append(len(preferences))
          data["n_pref_output"].append(len(p2_o))
          data["theta_size"].append(len(theta))
          data["theta_additivity"].append(additivity(theta))
          data["n_tiers"].append(t)
          data["alpha"].append(alpha)
          data["p"].append(p)
          data["n_items"].append(n_items)
          df = pd.DataFrame(data)
          df.to_csv(EXP_TITLE)
          print(f"\n Reps:{reps}, p:{p}, alpha:{alpha}, t:{t}, i={i}")
          summary = pd.DataFrame(df)
          summary = summary.groupby(["budget", "model"]).mean().reset_index()
          print(summary)
        print("%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%")

