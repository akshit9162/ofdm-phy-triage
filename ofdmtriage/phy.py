"""802.11a-style OFDM physical layer: Gray QAM, framing, and a receiver.

Numerology follows 802.11a (64-point FFT, 16-sample cyclic prefix,
48 data + 4 pilot subcarriers, two-repetition long training field).
FFTs use norm="ortho", so a unit-energy symbol on a subcarrier stays
unit-energy and noise of variance s2 per time sample is s2 per subcarrier.
"""
from dataclasses import dataclass, field

import numpy as np

N_FFT = 64
N_CP = 16
SYM_LEN = N_FFT + N_CP

PILOT_K = np.array([-21, -7, 7, 21])
USED_K = np.array([k for k in range(-26, 27) if k != 0])
DATA_K = np.array([k for k in USED_K if k not in PILOT_K])
PILOT_VALUES = np.array([1, 1, 1, -1], dtype=complex)

# 802.11a long training sequence on subcarriers -26..26 (DC = 0).
_LTF = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
                 0,
                 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1])
LTF_USED = _LTF[USED_K + 26].astype(complex)

# Preamble: 32-sample guard interval followed by two identical LTF symbols.
PREAMBLE_LEN = 32 + 2 * N_FFT

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}


def bins(k):
    """Subcarrier index (-32..31) to FFT bin (0..63)."""
    return np.asarray(k) % N_FFT


def _gray(n):
    return n ^ (n >> 1)


def _pam(bits_per_axis):
    """Gray-coded PAM: levels[g] is the amplitude whose Gray label is g."""
    n = 1 << bits_per_axis
    levels = np.empty(n)
    for i in range(n):
        levels[_gray(i)] = 2 * i - (n - 1)
    return levels


def constellation(mod):
    """Unit-average-energy constellation indexed by the symbol's bit value (MSB first)."""
    b = BITS_PER_SYMBOL[mod]
    if mod == "BPSK":
        return np.array([-1.0, 1.0], dtype=complex)
    half = b // 2
    pam = _pam(half)
    v = np.arange(1 << b)
    pts = pam[v >> half] + 1j * pam[v & ((1 << half) - 1)]
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def bits_to_ints(bits, b):
    bits = np.asarray(bits).reshape(-1, b)
    return bits @ (1 << np.arange(b - 1, -1, -1))


def ints_to_bits(ints, b):
    ints = np.asarray(ints)[:, None]
    return ((ints >> np.arange(b - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def modulate(bits, mod):
    return constellation(mod)[bits_to_ints(bits, BITS_PER_SYMBOL[mod])]


def demodulate(symbols, mod):
    """Hard-decision nearest-point demapping."""
    pts = constellation(mod)
    idx = np.argmin(np.abs(np.asarray(symbols).reshape(-1, 1) - pts[None, :]), axis=1)
    return ints_to_bits(idx, BITS_PER_SYMBOL[mod])


def ofdm_symbol(used_values):
    """Map 52 used-subcarrier values to one time-domain symbol (no CP)."""
    X = np.zeros(N_FFT, dtype=complex)
    X[bins(USED_K)] = used_values
    return np.fft.ifft(X, norm="ortho")


def preamble():
    ltf = ofdm_symbol(LTF_USED)
    return np.concatenate([ltf[-32:], ltf, ltf])


@dataclass
class Frame:
    mod: str
    bits: np.ndarray        # transmitted payload bits
    data_syms: np.ndarray   # (n_sym, 48) reference data symbols
    samples: np.ndarray     # time-domain baseband samples


def build_frame(mod, n_sym, rng):
    b = BITS_PER_SYMBOL[mod]
    bits = rng.integers(0, 2, size=n_sym * len(DATA_K) * b, dtype=np.uint8)
    data = modulate(bits, mod).reshape(n_sym, len(DATA_K))
    parts = [preamble()]
    used = np.zeros(len(USED_K), dtype=complex)
    data_pos = np.searchsorted(USED_K, DATA_K)
    pilot_pos = np.searchsorted(USED_K, PILOT_K)
    for m in range(n_sym):
        used[data_pos] = data[m]
        used[pilot_pos] = PILOT_VALUES
        t = ofdm_symbol(used)
        parts.append(np.concatenate([t[-N_CP:], t]))
    return Frame(mod, bits, data, np.concatenate(parts))


@dataclass
class RxConfig:
    correct_cfo: bool = True
    track_pilots: bool = True
    timing_offset: int = 0                 # FFT window shift in samples (>0 = late)
    genie_channel: np.ndarray | None = None  # equalize with this H (52 used bins) instead of the LTF estimate


@dataclass
class RxResult:
    bits: np.ndarray
    eq_data: np.ndarray       # (n_sym, 48) equalized data subcarriers
    cfo_est: float            # in subcarrier spacings
    H_est: np.ndarray         # 52 used subcarriers, from the LTF
    noise_var_est: float      # per-subcarrier noise variance, from the LTF difference
    pilot_phase: np.ndarray   # raw common phase per symbol (rad), before tracking
    extras: dict = field(default_factory=dict)


def _fft_used(x):
    return np.fft.fft(x, norm="ortho")[bins(USED_K)]


def receive(r, mod, n_sym, cfg=RxConfig()):
    """Synchronize, estimate the channel from the LTF, equalize and demap."""
    d = cfg.timing_offset
    # CFO from the phase advance between the two LTF repetitions (|eps| < 0.5).
    l1 = r[32 + d:32 + N_FFT + d]
    l2 = r[32 + N_FFT + d:32 + 2 * N_FFT + d]
    cfo_est = np.angle(np.vdot(l1, l2)) / (2 * np.pi)
    if cfg.correct_cfo:
        r = r * np.exp(-2j * np.pi * cfo_est * np.arange(len(r)) / N_FFT)
        l1 = r[32 + d:32 + N_FFT + d]
        l2 = r[32 + N_FFT + d:32 + 2 * N_FFT + d]

    Y1, Y2 = _fft_used(l1), _fft_used(l2)
    H_est = (Y1 + Y2) / 2 / LTF_USED
    noise_var_est = float(np.mean(np.abs(Y1 - Y2) ** 2) / 2)
    H_eq = H_est if cfg.genie_channel is None else cfg.genie_channel

    data_pos = np.searchsorted(USED_K, DATA_K)
    pilot_pos = np.searchsorted(USED_K, PILOT_K)
    eq = np.empty((n_sym, len(DATA_K)), dtype=complex)
    phase = np.empty(n_sym)
    for m in range(n_sym):
        start = PREAMBLE_LEN + m * SYM_LEN + N_CP + d
        Z = _fft_used(r[start:start + N_FFT]) / H_eq
        phase[m] = np.angle(np.vdot(PILOT_VALUES, Z[pilot_pos]))
        if cfg.track_pilots:
            Z = Z * np.exp(-1j * phase[m])
        eq[m] = Z[data_pos]
    bits = demodulate(eq.ravel(), mod)
    return RxResult(bits, eq, float(cfo_est), H_est, noise_var_est, phase)
