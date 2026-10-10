"""Auto-triage a campaign's failures into a consolidated markdown report.

Trains on TRAIN_DIR's labelled failures, triages every failure in RUN_DIR
from its log alone, and routes low-confidence ones to human review.

usage: python -m scripts.triage_report TRAIN_DIR RUN_DIR OUT.md [REVIEW_BELOW]
"""
import collections
import os
import sys

import numpy as np

from ofdmtriage.triage import MLTriage, feature_matrix, parse_log, split_logs
from scripts.evaluate import load

HINTS = {
    "low_snr": "Expected failure: SNR too low for this MCS on this channel. Check the test config, not the DUT.",
    "cfo_not_corrected": "Common phase ramps linearly across symbols: CFO is estimated but not removed.",
    "stale_channel_estimate": "Equalizer error follows the channel's fades: estimate not refreshed per packet.",
    "timing_late": "Channel impulse response wraps to negative delay: FFT window starts past the CP.",
    "pa_clipping": "Constellation compressed (gain < 1, outer points shrink): PA driven into saturation.",
    "iq_imbalance": "Error correlates with the mirror subcarrier: IQ gain/phase imbalance in the modulator.",
    "phase_noise": "Common phase random-walks between symbols: LO phase noise.",
    "delay_spread_exceeds_cp": "Channel taps beyond the 16-sample CP: ISI. Check the channel emulator profile.",
}


def main(train_dir, run_dir, out, review_below=0.9):
    X, y, _, _ = load(train_dir)
    model = MLTriage().fit(X[y != "ambiguous"], y[y != "ambiguous"])

    ids, recs, total = [], [], 0
    for tc, text in split_logs(open(os.path.join(run_dir, "campaign.log")).read()):
        total += 1
        rec = parse_log(text)
        if rec["verdict"] == "FAIL":
            ids.append(tc)
            recs.append(rec)
    p = model.predict_proba(feature_matrix(recs))
    pred = model.model.classes_[p.argmax(axis=1)]
    conf = p.max(axis=1)

    groups = collections.defaultdict(list)
    review = []
    for tc, rec, c, lab in zip(ids, recs, conf, pred):
        (groups[lab] if c >= review_below else review).append((tc, rec, c, lab))

    lines = [f"# Triage report: `{run_dir}`", "",
             f"{total} test cases, **{len(ids)} failed**. "
             f"{len(ids) - len(review)} triaged automatically, "
             f"**{len(review)} sent to human review** (confidence < {review_below}).", "",
             "| Root cause | Failures | Mean confidence | Mean EVM miss (dB) |",
             "|---|---:|---:|---:|"]
    for lab, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        miss = np.mean([r["evm_db"] - r["limit_db"] for _, r, _, _ in rows])
        lines.append(f"| `{lab}` | {len(rows)} | {np.mean([c for *_, c, _ in rows]):.3f} | {miss:+.1f} |")
    for lab, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines += ["", f"## `{lab}` ({len(rows)})", "", HINTS[lab], ""]
        by_cfg = collections.Counter(f"{r['mod']} / {r['channel']}" for _, r, _, _ in rows)
        lines.append("Most affected configs: " + ", ".join(f"{k} ({v})" for k, v in by_cfg.most_common(3)))
        lines.append("")
        lines.append("Examples: " + ", ".join(f"`{tc}`" for tc, *_ in rows[:6]))
    if review:
        lines += ["", f"## Needs human review ({len(review)})", "",
                  "| Test | Config | Best guess | Confidence |", "|---|---|---|---:|"]
        for tc, r, c, lab in sorted(review, key=lambda t: t[2])[:25]:
            lines.append(f"| `{tc}` | {r['mod']} @ {r['snr_db']:.1f} dB, {r['channel']} | `{lab}` | {c:.2f} |")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{len(ids)} failures -> {out} ({len(review)} for review)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.9)
