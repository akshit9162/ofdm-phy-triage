"""Evaluate adaptive test selection on fresh simulated builds.

usage: python -m scripts.test_selection TRAIN_DIR OUT_DIR [HISTORY_BUILDS] [EVAL_BUILDS]
"""
import json
import os
import sys

import numpy as np

from ofdmtriage.selection import HYPOTHESES, SUITE, diagnose, likelihood_table, simulate_build
from ofdmtriage.triage import MLTriage
from scripts.evaluate import load


def main(train_dir, out, history=60, per_hyp=50):
    os.makedirs(out, exist_ok=True)
    X, y, _, _ = load(train_dir)
    model = MLTriage().fit(X[y != "ambiguous"], y[y != "ambiguous"])

    L = likelihood_table(model, history, np.random.default_rng(100))      # history builds
    rng = np.random.default_rng(200)                                      # fresh builds
    builds = [(h, simulate_build(h, model, rng)) for h in HYPOTHESES for _ in range(per_hyp)]

    res = {"suite_size": len(SUITE), "history_builds_per_hypothesis": history,
           "eval_builds": len(builds), "stop_posterior": 0.99, "policies": {}}
    for policy in ["full", "random", "infogain"]:
        prng = np.random.default_rng(300)
        rows = [(h, *diagnose(o, L, policy, rng=prng)) for h, o in builds]
        truth = np.array([r[0] for r in rows])
        pred = np.array([r[1] for r in rows])
        n = np.array([r[2] for r in rows])
        bug = truth != "none"
        res["policies"][policy] = {
            "accuracy": round(float((truth == pred).mean()), 4),
            "mean_tests": round(float(n.mean()), 2),
            "median_tests": float(np.median(n)),
            "p90_tests": float(np.quantile(n, 0.9)),
            "missed_bug_rate": round(float((pred[bug] == "none").mean()), 4),
            "false_alarm_rate": round(float((pred[~bug] != "none").mean()), 4),
            "per_hypothesis": {h: {"accuracy": round(float((pred[truth == h] == h).mean()), 3),
                                   "mean_tests": round(float(n[truth == h].mean()), 1)}
                               for h in HYPOTHESES},
        }
    json.dump(res, open(os.path.join(out, "test_selection.json"), "w"), indent=2)
    for p, r in res["policies"].items():
        print(p, {k: v for k, v in r.items() if k != "per_hypothesis"})
        print("   ", r["per_hypothesis"])


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:4]))
