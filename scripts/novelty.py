"""Can triage notice a failure type it was never trained on?

Leave-one-fault-out: train on every root cause but one, then score test
failures for novelty. Compares three scores:
  * classifier confidence (1 - max class probability)
  * Isolation Forest on the scaled features
  * distance to the 10 nearest training failures (kNN)
  * either: flag if confidence OR kNN exceeds its own 2.5% threshold
Thresholds are set on a held-out 20% of training failures at a 5%
false-alarm rate, then applied to the independent test campaign.

usage: python -m scripts.novelty TRAIN_DIR TEST_DIR OUT_DIR
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from ofdmtriage.triage import MLTriage
from ofdmtriage.validate import FAULTS
from scripts.evaluate import load


def scorers(Xfit, yfit):
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(Xfit)
    Z = qt.transform(Xfit)
    iso = IsolationForest(n_estimators=300, random_state=0).fit(Z)
    nn = NearestNeighbors(n_neighbors=10).fit(Z)
    clf = MLTriage().fit(Xfit, yfit)
    return {
        "confidence": lambda X: 1 - clf.predict_proba(X).max(axis=1),
        "isolation_forest": lambda X: -iso.score_samples(qt.transform(X)),
        "knn_distance": lambda X: nn.kneighbors(qt.transform(X))[0].mean(axis=1),
    }, clf


def main(train_dir, test_dir, out):
    Xtr, ytr, _, _ = load(train_dir)
    Xte, yte, _, _ = load(test_dir)
    Xtr, ytr = Xtr[ytr != "ambiguous"], ytr[ytr != "ambiguous"]
    Xte, yte = Xte[yte != "ambiguous"], yte[yte != "ambiguous"]
    res = {}
    for held in FAULTS:
        known = ytr != held
        Xfit, Xcal, yfit, _ = train_test_split(Xtr[known], ytr[known], test_size=0.2,
                                               random_state=0, stratify=ytr[known])
        fns, clf = scorers(Xfit, yfit)
        novel = yte == held
        r = {"n_novel": int(novel.sum()), "n_known": int((~novel).sum())}
        labels, counts = np.unique(clf.predict(Xte[novel]), return_counts=True)
        r["classifier_calls_it"] = labels[counts.argmax()]
        for name, fn in fns.items():
            t = np.quantile(fn(Xcal), 0.95)
            s = fn(Xte)
            r[name] = {"auroc": round(float(roc_auc_score(novel, s)), 3),
                       "detected": round(float((s[novel] > t).mean()), 3),
                       "false_alarm": round(float((s[~novel] > t).mean()), 3)}
        # Union of the two complementary scores, each given half the false-alarm budget.
        flag = np.zeros(len(yte), bool)
        for name in ["confidence", "knn_distance"]:
            flag |= fns[name](Xte) > np.quantile(fns[name](Xcal), 0.975)
        r["either"] = {"detected": round(float(flag[novel].mean()), 3),
                       "false_alarm": round(float(flag[~novel].mean()), 3)}
        res[held] = r
        print(held, r)
    summary = {name: {k: round(float(np.mean([res[h][name][k] for h in FAULTS])), 3)
                      for k in ["auroc", "detected", "false_alarm"]}
               for name in ["confidence", "isolation_forest", "knn_distance"]}
    summary["either"] = {k: round(float(np.mean([res[h]["either"][k] for h in FAULTS])), 3)
                         for k in ["detected", "false_alarm"]}
    res["mean_over_held_out_faults"] = summary
    print(summary)
    json.dump(res, open(os.path.join(out, "novelty.json"), "w"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:4])
