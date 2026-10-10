"""Train triage on one campaign, evaluate on an independent one.

usage: python -m scripts.evaluate TRAIN_DIR TEST_DIR OUT_DIR
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from ofdmtriage.triage import FEATURES, MLTriage, RuleTriage, feature_matrix, parse_log, split_logs
from ofdmtriage.validate import LABELS


def load(run_dir):
    gt = {r["tc_id"]: r for r in csv.DictReader(open(os.path.join(run_dir, "ground_truth.csv")))}
    recs, labels, mods, faults = [], [], [], []
    for tc, text in split_logs(open(os.path.join(run_dir, "campaign.log")).read()):
        rec = parse_log(text)
        if rec["verdict"] != "FAIL":
            continue
        recs.append(rec)
        labels.append(gt[tc]["label"])
        mods.append(rec["mod"])
        faults.append(gt[tc]["fault"] or "none")
    return feature_matrix(recs), np.array(labels), np.array(mods), np.array(faults)


def scores(y, p):
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, labels=LABELS, average="macro")), 4),
        "recall": {c: round(float(r), 3)
                   for c, r in zip(LABELS, recall_score(y, p, labels=LABELS, average=None, zero_division=0))},
    }


def plot_confusion(y, p, path, title):
    cm = confusion_matrix(y, p, labels=LABELS)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    short = [l.replace("_", " ") for l in LABELS]
    ax.set_xticks(range(len(LABELS)), short, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(LABELS)), short, fontsize=8.5)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "#222")
    ax.set_xlabel("predicted root cause")
    ax.set_ylabel("true root cause")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(train_dir, test_dir, out):
    os.makedirs(out, exist_ok=True)
    Xtr_all, ytr_all, mtr, _ = load(train_dir)
    Xte_all, yte_all, mte, fte = load(test_dir)
    keep_tr = ytr_all != "ambiguous"
    keep_te = yte_all != "ambiguous"
    Xtr, ytr = Xtr_all[keep_tr], ytr_all[keep_tr]
    Xte, yte = Xte_all[keep_te], yte_all[keep_te]

    rules = RuleTriage().fit(Xtr, ytr)
    ml = MLTriage().fit(Xtr, ytr)
    p_rules, p_ml = rules.predict(Xte), ml.predict(Xte)

    res = {
        "train_failures": int(len(ytr_all)), "train_ambiguous": int((~keep_tr).sum()),
        "test_failures": int(len(yte_all)), "test_ambiguous": int((~keep_te).sum()),
        "rules": scores(yte, p_rules),
        "ml": scores(yte, p_ml),
        "rule_thresholds": {f"{f} {d} -> {l}": round(t, 4)
                            for (f, d, l), t in zip(__import__("ofdmtriage.triage", fromlist=["RULES"]).RULES,
                                                    rules.thresholds)},
    }

    # Generalization: never see 64QAM in training, test only on 64QAM.
    tr = keep_tr & (mtr != "64QAM")
    te = keep_te & (mte == "64QAM")
    ml_h = MLTriage().fit(Xtr_all[tr], ytr_all[tr])
    rules_h = RuleTriage().fit(Xtr_all[tr], ytr_all[tr])
    res["heldout_64qam"] = {"ml": scores(yte_all[te], ml_h.predict(Xte_all[te])),
                            "rules": scores(yte_all[te], rules_h.predict(Xte_all[te]))}

    # Abstention: send the least confident failures to a human.
    conf = ml.predict_proba(Xte).max(axis=1)
    correct = p_ml == yte
    curve = []
    for cov in [1.0, 0.95, 0.9, 0.85, 0.8]:
        t = np.quantile(conf, 1 - cov)
        sel = conf >= t
        curve.append({"coverage": cov, "threshold": round(float(t), 3),
                      "accuracy": round(float(correct[sel].mean()), 4)})
    res["abstention"] = curve

    # What the model says about failures that have no single root cause.
    # These fail with or without the injected fault (fault + too little SNR).
    amb = Xte_all[~keep_te]
    if len(amb):
        p = ml.predict_proba(amb)
        pred = ml.model.classes_[p.argmax(axis=1)]
        res["ambiguous"] = {
            "n": int(len(amb)),
            "mean_confidence": round(float(p.max(axis=1).mean()), 3),
            "clean_mean_confidence": round(float(conf.mean()), 3),
            "share_confidence_ge_0.9": round(float((p.max(axis=1) >= 0.9).mean()), 3),
            "predicted_injected_fault": round(float((pred == fte[~keep_te]).mean()), 3),
            "predicted_low_snr": round(float((pred == "low_snr").mean()), 3),
        }
        # Realistic queue: all failures mixed; defer the least-confident 10%.
        c_all = ml.predict_proba(Xte_all).max(axis=1)
        deferred = c_all < np.quantile(c_all, 0.10)
        res["ambiguous"]["defer_10pct_share_ambiguous"] = round(float((~keep_te)[deferred].mean()), 3)
        res["ambiguous"]["defer_10pct_ambiguous_caught"] = round(float(deferred[~keep_te].mean()), 3)

    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    plot_confusion(yte, p_ml, os.path.join(out, "confusion_ml.png"),
                   f"ML triage on an independent campaign (macro-F1 {res['ml']['macro_f1']:.3f})")
    plot_confusion(yte, p_rules, os.path.join(out, "confusion_rules.png"),
                   f"Rule triage on an independent campaign (macro-F1 {res['rules']['macro_f1']:.3f})")
    print(json.dumps({k: res[k] for k in ["rules", "ml", "heldout_64qam", "abstention"]}, indent=1))
    print("ambiguous", res.get("ambiguous"))
    print(res["rule_thresholds"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
