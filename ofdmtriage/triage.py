"""Failure triage from test logs: parse -> features -> root cause.

Two triagers share one feature extractor:
  * RuleTriage: an ordered decision list written from the physics of each
    impairment (what a validation engineer would check by hand).
  * MLTriage: gradient-boosted trees on the same features, with an
    abstain option that sends low-confidence failures to a human.
"""
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .validate import EVM_LIMIT_DB, N_CP

_KV = re.compile(r"(\w+)=([^\s]+)")


def parse_log(text):
    """Turn one test log into a dict of scalars and arrays."""
    rec = {}
    for line in text.splitlines():
        if not line or line.startswith("==="):
            continue
        tag, _, rest = line.partition(" ")
        kv = dict(_KV.findall(rest))
        if kv:
            for k, v in kv.items():
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
        else:
            name, _, values = rest.partition(" ")
            rec[f"{tag.lower()}_{name}"] = np.array(values.split(), dtype=float)
    return rec


def split_logs(text):
    """Split a campaign log file into (tc_id, log_text) pairs."""
    parts = re.split(r"(?m)^(?==== TEST )", text)
    return [(re.match(r"=== TEST (\S+)", p).group(1), p) for p in parts if p.strip()]


def _lin(db):
    return 10 ** (np.asarray(db) / 10)


FEATURES = [
    "bits_per_symbol", "evm_margin_db", "excess_db", "excess_sc_std_db",
    "snr_gap_db", "abs_cfo_est",
    "pilot_slope_deg", "pilot_resid_std_deg", "pilot_step_std_deg",
    "evm_sym_slope_db", "evm_sym_std_db",
    "cir_main_frac", "cir_beyond_cp_frac", "cir_precursor_frac",
    "gain", "radial_slope", "image_corr",
]


def features(rec):
    """Scalar features, each tied to an impairment signature."""
    sc = _lin(rec["evm_per_subcarrier_db"])
    exp_sc = _lin(rec["evm_expected_per_subcarrier_db"])
    sym = rec["evm_per_symbol_db"]
    ph = rec["pilot_phase_deg"]
    cir = _lin(rec["chest_cir_db"])
    t = np.arange(len(ph))
    slope, icpt = np.polyfit(t, ph, 1)
    bits = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}[rec["mod"]]
    return {
        "bits_per_symbol": bits,
        # how far the test missed its limit
        "evm_margin_db": rec["evm_db"] - EVM_LIMIT_DB[rec["mod"]],
        # measured error vs what the LTF noise estimate predicts: ~0 dB means noise-limited
        "excess_db": 10 * np.log10(sc.mean() / exp_sc.mean()),
        "excess_sc_std_db": float(np.std(10 * np.log10(sc / exp_sc))),
        "snr_gap_db": rec["snr_db"] - rec["snr_est_db"],
        "abs_cfo_est": abs(rec["cfo_est"]),
        # uncorrected CFO -> linear common-phase ramp; phase noise -> random walk
        "pilot_slope_deg": abs(slope),
        "pilot_resid_std_deg": float(np.std(ph - (slope * t + icpt))),
        "pilot_step_std_deg": float(np.std(np.diff(ph))),
        "evm_sym_slope_db": float(np.polyfit(t, sym, 1)[0]),
        "evm_sym_std_db": float(np.std(sym)),
        # channel impulse response from the LTF: within CP, beyond CP, or wrapped (early taps)
        "cir_main_frac": cir[:N_CP].sum() / cir.sum(),
        "cir_beyond_cp_frac": cir[N_CP:40].sum() / cir.sum(),
        "cir_precursor_frac": cir[52:].sum() / cir.sum(),
        "gain": rec["gain"],
        "radial_slope": rec["radial_slope"],
        "image_corr": rec["image_corr"],
    }


def feature_matrix(recs):
    return np.array([[features(r)[f] for f in FEATURES] for r in recs])


# Hand-ordered checks: (feature, direction, root cause). Only the
# thresholds are learned, each as a one-vs-rest stump on the training split.
RULES = [
    ("pilot_slope_deg", ">", "cfo_not_corrected"),  # linear common-phase ramp
    ("cir_precursor_frac", ">", "timing_late"),     # taps wrapped to negative delay
    ("cir_beyond_cp_frac", ">", "delay_spread_exceeds_cp"),
    ("image_corr", ">", "iq_imbalance"),            # error correlated with mirror subcarrier
    ("excess_sc_std_db", ">", "stale_channel_estimate"),  # error tracks the old channel's fades
    ("radial_slope", "<", "pa_clipping"),           # large symbols compressed (QAM)
    ("gain", "<", "pa_clipping"),                   # whole constellation shrunk (BPSK/QPSK)
    ("pilot_resid_std_deg", ">", "phase_noise"),    # random-walk common phase
    ("excess_db", "<", "low_snr"),                  # error explained by measured noise
]


def _best_threshold(x, positive, direction):
    """Threshold maximizing one-vs-rest F1 for `x direction t`."""
    best_t, best_f1 = None, -1.0
    for t in np.unique(np.quantile(x, np.linspace(0.01, 0.99, 197))):
        hit = x < t if direction == "<" else x > t
        tp = np.sum(hit & positive)
        f1 = 2 * tp / (hit.sum() + positive.sum() + 1e-12)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


class RuleTriage:
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        remaining = np.ones(len(y), bool)
        self.thresholds = []
        for feat, direction, label in RULES:
            col = X[:, FEATURES.index(feat)]
            t = _best_threshold(col[remaining], y[remaining] == label, direction)
            self.thresholds.append(t)
            remaining &= ~((col < t) if direction == "<" else (col > t))
        labels, counts = np.unique(y[remaining], return_counts=True)
        self.fallback = labels[counts.argmax()] if len(labels) else "low_snr"
        return self

    def predict_one(self, f):
        for (feat, direction, label), t in zip(RULES, self.thresholds):
            if (f[feat] < t) if direction == "<" else (f[feat] > t):
                return label
        return self.fallback

    def predict(self, X):
        return np.array([self.predict_one(dict(zip(FEATURES, row))) for row in np.asarray(X)])


class MLTriage:
    def __init__(self, seed=0):
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                    random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X, abstain_below=None):
        p = self.predict_proba(X)
        pred = self.model.classes_[p.argmax(axis=1)].astype(object)
        if abstain_below is not None:
            pred[p.max(axis=1) < abstain_below] = "needs_review"
        return pred
