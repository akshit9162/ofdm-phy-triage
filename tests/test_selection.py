import numpy as np

from ofdmtriage.selection import HYPOTHESES, OUTCOMES, SUITE, diagnose


def test_suite_grid():
    assert len(SUITE) == 48 and len(set(SUITE)) == 48


def test_infogain_picks_the_discriminating_test():
    """Config 7 alone reveals the hypothesis; every other config is pure noise."""
    H, C, O = len(HYPOTHESES), len(SUITE), len(OUTCOMES)
    L = np.full((H, C, O), 1 / O)
    L[:, 7, :] = 1e-4
    for h in range(H):
        L[h, 7, h] = 1.0
    L /= L.sum(axis=2, keepdims=True)
    outcomes = np.zeros(C, dtype=int)
    outcomes[7] = HYPOTHESES.index("timing_late")
    assert diagnose(outcomes, L, "infogain") == ("timing_late", 1)
    assert diagnose(outcomes, L, "full") == ("timing_late", C)
