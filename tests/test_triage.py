import numpy as np
import pytest

from ofdmtriage.triage import FEATURES, RuleTriage, features, parse_log, split_logs
from ofdmtriage.validate import FAULTS, TestCase, measure, run_case, run_link, write_log


def case(fault=None, sev=0.0, mod="16QAM", snr=30.0, profile="short", seed=7):
    return TestCase("tc_t", mod, snr, profile, 0.12, fault, sev, seed=seed)


def test_log_roundtrip():
    tc = case()
    f, rx = run_link(tc)
    m = measure(tc, f, rx)
    rec = parse_log(write_log(tc, m))
    assert rec["mod"] == "16QAM" and rec["verdict"] == "PASS"
    assert np.isclose(rec["evm_db"], m.evm_db, atol=0.01)
    assert len(rec["evm_per_subcarrier_db"]) == 48
    assert len(rec["pilot_phase_deg"]) == tc.n_sym
    assert len(rec["chest_cir_db"]) == 64
    assert set(FEATURES) <= set(features(rec))


def test_log_hides_ground_truth():
    log, _, _ = run_case(case("pa_clipping", -1.0))
    assert "pa_clipping" not in log and "fault" not in log and "severity" not in log


def test_split_logs():
    text = "".join(run_case(case(seed=s))[0] for s in range(3))
    assert [tc for tc, _ in split_logs(text)] == ["tc_t"] * 3


def test_counterfactual_differs_only_by_fault():
    tc = case()
    f1, rx1 = run_link(tc)
    f2, rx2 = run_link(tc, with_fault=False)
    assert np.array_equal(f1.bits, f2.bits) and np.allclose(rx1.eq_data, rx2.eq_data)


@pytest.mark.parametrize("fault,sev", [
    ("cfo_not_corrected", 0.1), ("stale_channel_estimate", 0.2), ("timing_late", 6),
    ("pa_clipping", -1.5), ("iq_imbalance", 3.0), ("phase_noise", 3.0),
    ("delay_spread_exceeds_cp", 0.0)])
def test_each_fault_breaks_a_clean_link(fault, sev):
    assert fault in FAULTS
    _, clean, _ = run_case(case(profile="long" if fault == "delay_spread_exceeds_cp" else "short"))
    _, m, label = run_case(case(fault, sev, profile="long" if fault == "delay_spread_exceeds_cp" else "short"))
    assert clean.passed or fault == "delay_spread_exceeds_cp"
    assert not m.passed and label == fault


def test_signatures_point_at_the_fault():
    """The physics each rule relies on, checked on a single high-SNR run."""
    f = {k: features(parse_log(run_case(case(k, s))[0]))
         for k, s in [("cfo_not_corrected", 0.1), ("timing_late", 6), ("iq_imbalance", 3.0),
                      ("pa_clipping", -1.5), (None, 0.0)]}
    clean = f[None]
    assert f["cfo_not_corrected"]["pilot_slope_deg"] > 30 > clean["pilot_slope_deg"]
    assert f["timing_late"]["cir_precursor_frac"] > 0.5 > clean["cir_precursor_frac"]
    assert f["iq_imbalance"]["image_corr"] > 0.4 > clean["image_corr"]
    assert f["pa_clipping"]["radial_slope"] < -0.05 < clean["radial_slope"]


def test_rule_triage_fits_and_predicts():
    X = np.zeros((4, len(FEATURES)))
    X[:, FEATURES.index("excess_db")] = [0.5, 0.7, 12, 14]
    X[:, FEATURES.index("pilot_slope_deg")] = [1, 1, 40, 45]
    y = np.array(["low_snr", "low_snr", "cfo_not_corrected", "cfo_not_corrected"])
    assert list(RuleTriage().fit(X, y).predict(X)) == list(y)
