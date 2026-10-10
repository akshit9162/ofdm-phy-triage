"""Run a validation campaign: N random test cases -> one log file + ground truth CSV.

usage: python -m scripts.campaign N_CASES OUT_DIR [SEED]
"""
import csv
import os
import sys

import numpy as np

from ofdmtriage.validate import case_record, run_case, sample_case


def main(n, out, seed=0):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    with open(os.path.join(out, "campaign.log"), "w") as logf:
        for i in range(n):
            tc = sample_case(i, rng)
            log, m, label = run_case(tc)
            logf.write(log)
            rows.append(case_record(tc, m, label))
    with open(os.path.join(out, "ground_truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    failed = sum(not r["passed"] for r in rows)
    print(f"{n} cases, {failed} failed -> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
