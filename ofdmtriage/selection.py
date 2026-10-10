"""Adaptive regression-test selection for a new build with an unknown bug.

A build either is clean or carries one DUT bug at an unknown severity.
The regression suite is a fixed grid of 48 configs. Instead of running all
of them, pick the next test that is expected to tell us the most about
which bug (if any) the build has, update a posterior from the outcome
(PASS, or the triage model's root cause for a FAIL), and stop once one
hypothesis is confident enough.

Likelihoods P(outcome | hypothesis, config) come from simulated builds
(the "history"), with the triage model's own confusions baked in.
"""
import numpy as np

from .triage import feature_matrix, parse_log
from .validate import EVM_LIMIT_DB, LABELS, TestCase, measure, run_link, sample_severity, write_log

# delay_spread_exceeds_cp is a channel-emulator setting, not a DUT bug.
HYPOTHESES = ["none", "cfo_not_corrected", "stale_channel_estimate", "timing_late",
              "pa_clipping", "iq_imbalance", "phase_noise"]
OUTCOMES = ["PASS"] + LABELS
SUITE = [(mod, profile, margin)
         for mod in EVM_LIMIT_DB
         for profile in ["flat", "short", "medium"]
         for margin in [2, 8, 14, 20]]          # SNR above what the modulation needs, dB


def suite_case(cfg, hyp, sev, rng):
    mod, profile, margin = cfg
    return TestCase("sel", mod, -EVM_LIMIT_DB[mod] + margin, profile,
                    float(rng.uniform(-0.3, 0.3)), None if hyp == "none" else hyp, sev,
                    seed=int(rng.integers(2**31)))


def run_outcomes(cases, model):
    """Run test cases; return outcome indices (PASS, or the triaged root cause)."""
    out = np.zeros(len(cases), dtype=int)
    failed, recs = [], []
    for i, tc in enumerate(cases):
        f, rx = run_link(tc)
        m = measure(tc, f, rx)
        if not m.passed:
            failed.append(i)
            recs.append(parse_log(write_log(tc, m)))
    if recs:
        pred = model.predict(feature_matrix(recs))
        out[failed] = [OUTCOMES.index(p) for p in pred]
    return out


def simulate_build(hyp, model, rng):
    """One build: a hypothesis at one severity, run through every suite config."""
    sev = sample_severity(None if hyp == "none" else hyp, rng)
    return run_outcomes([suite_case(c, hyp, sev, rng) for c in SUITE], model)


def likelihood_table(model, builds_per_hypothesis, rng, alpha=0.5):
    """P(outcome | hypothesis, config), estimated from simulated history builds."""
    counts = np.full((len(HYPOTHESES), len(SUITE), len(OUTCOMES)), alpha)
    for h, hyp in enumerate(HYPOTHESES):
        for _ in range(builds_per_hypothesis):
            counts[h, np.arange(len(SUITE)), simulate_build(hyp, model, rng)] += 1
    return counts / counts.sum(axis=2, keepdims=True)


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def diagnose(outcomes, L, policy="infogain", stop=0.99, rng=None):
    """Run tests from a build's outcome vector under a selection policy.

    Returns (diagnosed hypothesis, number of tests run).
    """
    logpost = np.zeros(len(HYPOTHESES))
    remaining = list(range(len(SUITE)))
    if policy == "random":
        rng.shuffle(remaining)
    n = 0
    while remaining:
        post = np.exp(logpost - logpost.max())
        post /= post.sum()
        if policy != "full" and post.max() >= stop:
            break
        if policy == "infogain":
            pred = np.einsum("h,hco->co", post, L[:, remaining])        # P(outcome | config)
            cond = post[:, None, None] * L[:, remaining]                 # joint, then normalize per outcome
            cond = cond / cond.sum(axis=0, keepdims=True)
            exp_h = np.array([sum(pred[i, o] * _entropy(cond[:, i, o]) for o in range(len(OUTCOMES)))
                              for i in range(len(remaining))])
            c = remaining.pop(int(np.argmin(exp_h)))
        else:
            c = remaining.pop(0)
        logpost += np.log(L[:, c, outcomes[c]])
        n += 1
    return HYPOTHESES[int(np.argmax(logpost))], n
