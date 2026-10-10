"""Validation test cases: run a link, measure it, judge pass/fail, write a log.

A test case is a modulation, an SNR, a channel profile and (optionally) one
injected fault. The log is plain text, like a lab automation log; triage
works only from these logs, never from the simulator's internal state.
"""
from dataclasses import asdict, dataclass

import numpy as np

from . import channel as ch
from .phy import (DATA_K, N_CP, N_FFT, USED_K, RxConfig, build_frame,
                  receive)

# Transmit modulation-accuracy limits (dB), after the 802.11a EVM table.
EVM_LIMIT_DB = {"BPSK": -5.0, "QPSK": -10.0, "16QAM": -16.0, "64QAM": -22.0}

FAULTS = ["cfo_not_corrected", "stale_channel_estimate", "timing_late",
          "pa_clipping", "iq_imbalance", "phase_noise", "delay_spread_exceeds_cp"]
LABELS = ["low_snr"] + FAULTS


@dataclass
class TestCase:
    __test__ = False  # not a pytest class
    tc_id: str
    mod: str
    snr_db: float
    profile: str
    cfo: float
    fault: str | None = None
    severity: float = 0.0      # fault parameter, units depend on the fault
    n_sym: int = 40
    seed: int = 0


def sample_severity(fault, rng):
    """Draw a fault's parameter; units depend on the fault."""
    if fault == "cfo_not_corrected":
        return float(rng.uniform(0.01, 0.15) * rng.choice([-1, 1]))  # uncorrected CFO, subcarrier spacings
    if fault == "stale_channel_estimate":
        return float(rng.uniform(0.0, 0.95))                          # correlation with the true channel
    if fault == "timing_late":
        return float(rng.integers(1, 9))                              # samples past the CP
    if fault == "pa_clipping":
        return float(rng.uniform(-2.0, 4.0))                          # clip level above RMS, dB
    if fault == "iq_imbalance":
        return float(rng.uniform(1.0, 4.0))                           # gain imbalance dB (phase = 2.5x deg)
    if fault == "phase_noise":
        return float(rng.uniform(0.5, 3.5))                           # deg rms per sample
    return 0.0


def sample_case(i, rng):
    """Random test case for a campaign: ~30% fault-free, otherwise one fault."""
    mod = rng.choice(list(EVM_LIMIT_DB))
    need = -EVM_LIMIT_DB[mod]
    snr = float(np.round(rng.uniform(need - 2, need + 24), 1))
    profile = str(rng.choice(["flat", "short", "medium"]))
    fault = None if rng.random() < 0.3 else str(rng.choice(FAULTS))
    sev = sample_severity(fault, rng)
    if fault == "stale_channel_estimate":
        profile = str(rng.choice(["short", "medium"]))
    elif fault == "delay_spread_exceeds_cp":
        profile = "long"
    return TestCase(f"tc_{i:05d}", str(mod), snr, profile,
                    float(np.round(rng.uniform(-0.3, 0.3), 4)), fault, sev, seed=int(rng.integers(2**31)))


def run_link(tc, with_fault=True):
    """Simulate one packet. Independent RNG streams keep the counterfactual
    (same case, fault removed) identical except for the fault itself."""
    s_data, s_chan, s_noise, s_fault = np.random.SeedSequence(tc.seed).spawn(4)
    r_data, r_chan, r_noise, r_fault = map(np.random.default_rng, (s_data, s_chan, s_noise, s_fault))
    fault = tc.fault if with_fault else None
    profile = tc.profile
    if not with_fault and tc.fault == "delay_spread_exceeds_cp":
        profile = "medium"

    f = build_frame(tc.mod, tc.n_sym, r_data)
    x = np.concatenate([f.samples, np.zeros(32)])   # capture runs past the packet end
    if fault == "pa_clipping":
        x = ch.clip(x, tc.severity)
    if fault == "iq_imbalance":
        x = ch.iq_imbalance(x, tc.severity, 2.5 * tc.severity)   # transmit modulator imbalance
    taps = ch.multipath_taps(profile, r_chan)
    y = ch.apply_taps(x, taps)
    y = ch.apply_cfo(y, tc.severity if fault == "cfo_not_corrected" else tc.cfo)
    if fault == "phase_noise":
        y = ch.phase_noise(y, tc.severity, r_fault)
    y = ch.awgn(y, tc.snr_db, r_noise)

    cfg = RxConfig()
    if fault == "cfo_not_corrected":
        # The bug: the offset is estimated (and logged) but never removed.
        cfg = RxConfig(correct_cfo=False)
    if fault == "timing_late":
        cfg = RxConfig(timing_offset=int(tc.severity))
    if fault == "stale_channel_estimate":
        rho = tc.severity
        old = ch.multipath_taps(profile, r_fault)
        stale_taps = rho * taps + np.sqrt(1 - rho ** 2) * old
        cfg = RxConfig(genie_channel=ch.freq_response(stale_taps))
    return f, receive(y, tc.mod, tc.n_sym, cfg)


@dataclass
class Measurement:
    evm_db: float
    ber: float
    passed: bool
    cfo_est: float
    snr_est_db: float
    evm_per_symbol_db: np.ndarray
    pilot_phase_deg: np.ndarray
    evm_per_subcarrier_db: np.ndarray
    expected_evm_per_subcarrier_db: np.ndarray
    cir_db: np.ndarray            # LTF channel impulse response energy, taps 0..63 (dB re peak)
    gain: float                   # mean constellation gain Re<Z, X>/|X|^2
    radial_slope: float           # amplitude error vs reference amplitude
    image_corr: float             # mean |corr(err[k], X[-k])| over mirror pairs


def _db(x):
    return 10 * np.log10(np.maximum(x, 1e-12))


def measure(tc, f, rx):
    X, Z = f.data_syms, rx.eq_data
    err = np.abs(Z - X) ** 2
    evm = float(np.mean(err))
    ber = float(np.mean(rx.bits != f.bits))
    data_pos = np.searchsorted(USED_K, DATA_K)
    H = rx.H_est[data_pos]
    expected = rx.noise_var_est / np.maximum(np.abs(H) ** 2, 1e-12)

    Hfull = np.zeros(N_FFT, dtype=complex)
    Hfull[USED_K % N_FFT] = rx.H_est
    cir = np.abs(np.fft.ifft(Hfull)) ** 2
    cir_db = _db(cir / cir.max())

    gain = float(np.real(np.vdot(X, Z)) / np.vdot(X, X).real)
    amp = np.abs(X).ravel()
    radial = (np.abs(Z) - np.abs(X)).ravel()
    radial_slope = float(np.polyfit(amp, radial, 1)[0]) if np.ptp(amp) > 1e-9 else 0.0

    E = Z - X
    mirror = {k: i for i, k in enumerate(DATA_K)}
    corr = []
    for i, k in enumerate(DATA_K):
        if k > 0:
            j = mirror[-k]
            for a, b in ((i, j), (j, i)):
                num = abs(np.sum(E[:, a] * X[:, b]))
                den = np.sqrt(np.sum(np.abs(E[:, a]) ** 2) * np.sum(np.abs(X[:, b]) ** 2)) + 1e-12
                corr.append(num / den)

    return Measurement(
        evm_db=float(_db(evm)), ber=ber, passed=bool(_db(evm) <= EVM_LIMIT_DB[tc.mod]),
        cfo_est=rx.cfo_est, snr_est_db=float(_db(np.mean(np.abs(rx.H_est) ** 2) / rx.noise_var_est)),
        evm_per_symbol_db=_db(err.mean(axis=1)),
        pilot_phase_deg=np.rad2deg(np.unwrap(rx.pilot_phase)),
        evm_per_subcarrier_db=_db(err.mean(axis=0)),
        expected_evm_per_subcarrier_db=_db(expected),
        cir_db=cir_db, gain=gain, radial_slope=radial_slope, image_corr=float(np.mean(corr)),
    )


def _fmt(a, p=1):
    return " ".join(f"{v:.{p}f}" for v in a)


def write_log(tc, m):
    """Plain-text log. Ground truth (fault, severity) is deliberately NOT in it."""
    lines = [
        f"=== TEST {tc.tc_id} ===",
        f"CONFIG mod={tc.mod} snr_db={tc.snr_db:.1f} channel={tc.profile} n_sym={tc.n_sym} "
        f"fft={N_FFT} cp={N_CP}",
        f"SYNC cfo_est={m.cfo_est:+.4f}",
        f"CHEST snr_est_db={m.snr_est_db:.2f}",
        f"CHEST cir_db {_fmt(m.cir_db)}",
        f"EQ gain={m.gain:.4f} radial_slope={m.radial_slope:+.4f} image_corr={m.image_corr:.4f}",
        f"PILOT phase_deg {_fmt(m.pilot_phase_deg)}",
        f"EVM per_symbol_db {_fmt(m.evm_per_symbol_db)}",
        f"EVM per_subcarrier_db {_fmt(m.evm_per_subcarrier_db)}",
        f"EVM expected_per_subcarrier_db {_fmt(m.expected_evm_per_subcarrier_db)}",
        f"RESULT evm_db={m.evm_db:.2f} limit_db={EVM_LIMIT_DB[tc.mod]:.1f} ber={m.ber:.3e} "
        f"verdict={'PASS' if m.passed else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"


def run_case(tc):
    """Run a case and its fault-free counterfactual.

    Returns (log_text, measurement, label). label is the root cause of a
    failure: the injected fault if removing it makes the test pass,
    "low_snr" for a fault-free failure, "ambiguous" if the test fails with
    or without the fault, and None if the test passed.
    """
    f, rx = run_link(tc)
    m = measure(tc, f, rx)
    label = None
    if not m.passed:
        if tc.fault is None:
            label = "low_snr"
        else:
            m0 = measure(tc, *run_link(tc, with_fault=False))
            label = tc.fault if m0.passed else "ambiguous"
    return write_log(tc, m), m, label


def case_record(tc, m, label):
    d = asdict(tc)
    d.update(evm_db=m.evm_db, ber=m.ber, passed=m.passed, label=label)
    return d
