import numpy as np
import pytest
from scipy.special import erfc

from ofdmtriage import channel as ch
from ofdmtriage.phy import (BITS_PER_SYMBOL, RxConfig, build_frame, constellation,
                            demodulate, modulate, receive)


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


def theory_ber(mod, snr_db):
    """Gray-coded uncoded BER on AWGN at per-subcarrier Es/N0 (nearest-neighbour approx for QAM)."""
    es = 10 ** (snr_db / 10)
    if mod == "BPSK":
        return qfunc(np.sqrt(2 * es))
    m = 2 ** BITS_PER_SYMBOL[mod]
    k = np.log2(m)
    return 4 / k * (1 - 1 / np.sqrt(m)) * qfunc(np.sqrt(3 * es / (m - 1)))


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "16QAM", "64QAM"])
def test_constellation_unit_energy_and_gray(mod):
    pts = constellation(mod)
    assert np.isclose(np.mean(np.abs(pts) ** 2), 1.0)
    # Gray property: every nearest neighbour differs in exactly one bit.
    dmin = np.min(np.abs(pts[:, None] - pts[None, :]) + np.eye(len(pts)) * 9)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and np.isclose(abs(pts[i] - pts[j]), dmin):
                assert bin(i ^ j).count("1") == 1


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "16QAM", "64QAM"])
def test_modulate_demodulate_roundtrip(mod):
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 600 * BITS_PER_SYMBOL[mod], dtype=np.uint8)
    assert np.array_equal(demodulate(modulate(bits, mod), mod), bits)


@pytest.mark.parametrize("profile", ["flat", "short", "medium"])
def test_noiseless_loopback_with_multipath_and_cfo(profile):
    rng = np.random.default_rng(1)
    f = build_frame("64QAM", 20, rng)
    r = ch.apply_cfo(ch.apply_taps(f.samples, ch.multipath_taps(profile, rng)), 0.27)
    rx = receive(r, "64QAM", 20)
    assert abs(rx.cfo_est - 0.27) < 1e-9
    assert np.array_equal(rx.bits, f.bits)


@pytest.mark.parametrize("mod,snr_db", [("BPSK", 4), ("QPSK", 8), ("16QAM", 14), ("64QAM", 20)])
def test_awgn_ber_matches_theory(mod, snr_db):
    rng = np.random.default_rng(2)
    errs = total = 0
    while errs < 400:
        f = build_frame(mod, 40, rng)
        r = ch.awgn(f.samples, snr_db, rng)
        rx = receive(r, mod, 40, RxConfig(correct_cfo=False, track_pilots=False, genie_channel=np.ones(52)))
        errs += int(np.sum(rx.bits != f.bits))
        total += f.bits.size
    ber, th = errs / total, theory_ber(mod, snr_db)
    assert abs(np.log10(ber) - np.log10(th)) < 0.15, (ber, th)


def test_ls_estimation_loss_is_bounded():
    """Two-LTF LS estimation costs ~1.8 dB in theory; check the measured EVM loss."""
    rng = np.random.default_rng(3)
    snr = 20.0
    evm_genie, evm_ls = [], []
    for _ in range(30):
        f = build_frame("16QAM", 40, rng)
        r = ch.awgn(f.samples, snr, rng)
        g = receive(r, "16QAM", 40, RxConfig(correct_cfo=False, track_pilots=False, genie_channel=np.ones(52)))
        e = receive(r, "16QAM", 40, RxConfig(track_pilots=False, correct_cfo=False))
        evm_genie.append(np.mean(np.abs(g.eq_data - f.data_syms) ** 2))
        evm_ls.append(np.mean(np.abs(e.eq_data - f.data_syms) ** 2))
    loss_db = 10 * np.log10(np.mean(evm_ls) / np.mean(evm_genie))
    assert 1.2 < loss_db < 2.4, loss_db


def test_delay_spread_beyond_cp_creates_error_floor():
    rng = np.random.default_rng(4)
    floor = {}
    for profile in ["medium", "long"]:
        evm = []
        for _ in range(20):
            f = build_frame("16QAM", 20, rng)
            r = ch.apply_taps(f.samples, ch.multipath_taps(profile, rng))
            rx = receive(r, "16QAM", 20)
            evm.append(np.mean(np.abs(rx.eq_data - f.data_syms) ** 2))
        floor[profile] = 10 * np.log10(np.median(evm))
    assert floor["medium"] < -60       # within the CP: no ISI at all
    assert floor["long"] > -35         # beyond the CP: ISI floor without any noise


def test_noise_variance_estimate():
    rng = np.random.default_rng(5)
    est = [receive(ch.awgn(build_frame("QPSK", 4, rng).samples, 15, rng), "QPSK", 4).noise_var_est
           for _ in range(200)]
    assert abs(10 * np.log10(np.mean(est)) + 15) < 0.3
