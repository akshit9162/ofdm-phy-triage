"""Propagation channel and RF / receiver impairments.

Each impairment is a plain function of the sample stream so a test case
can compose them; the fault injector in validate.py picks one per run.
"""
import numpy as np

from .phy import N_FFT, USED_K, bins

# name -> (max excess delay in samples, rms delay spread in samples).
# "long" exceeds the 16-sample cyclic prefix and causes inter-symbol interference.
PROFILES = {
    "flat": (0, 0.0),
    "short": (6, 1.5),
    "medium": (14, 3.0),
    "long": (28, 9.0),
}


def multipath_taps(profile, rng):
    """Rayleigh taps with an exponential power-delay profile, unit total power."""
    max_delay, rms = PROFILES[profile]
    if max_delay == 0:
        return np.array([np.exp(2j * np.pi * rng.random())])
    n = np.arange(max_delay + 1)
    pdp = np.exp(-n / rms)
    pdp /= pdp.sum()
    g = (rng.standard_normal(len(n)) + 1j * rng.standard_normal(len(n))) / np.sqrt(2)
    return g * np.sqrt(pdp)


def freq_response(taps):
    """Channel frequency response on the 52 used subcarriers."""
    return np.fft.fft(taps, N_FFT)[bins(USED_K)]


def apply_taps(x, taps):
    return np.convolve(x, taps)[:len(x)]


def apply_cfo(x, eps):
    """Carrier frequency offset of eps subcarrier spacings."""
    return x * np.exp(2j * np.pi * eps * np.arange(len(x)) / N_FFT)


def awgn(x, snr_db, rng):
    """Noise for a per-subcarrier Es/N0 of snr_db (unit-energy constellations)."""
    s2 = 10 ** (-snr_db / 10)
    return x + np.sqrt(s2 / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))


def clip(x, clip_ratio_db):
    """Soft-limiter PA: amplitude clipped at clip_ratio_db above the payload RMS."""
    a = 10 ** (clip_ratio_db / 20) * np.sqrt(np.mean(np.abs(x) ** 2))
    mag = np.abs(x)
    scale = np.minimum(1.0, a / np.maximum(mag, 1e-12))
    return x * scale


def iq_imbalance(x, gain_db, phase_deg):
    """IQ gain/phase imbalance of a quadrature modulator: y = mu*x + nu*conj(x)."""
    g = 10 ** (gain_db / 20)
    phi = np.deg2rad(phase_deg)
    mu = (1 + g * np.exp(-1j * phi)) / 2
    nu = (1 - g * np.exp(1j * phi)) / 2
    return mu * x + nu * np.conj(x)


def phase_noise(x, sigma_deg, rng):
    """Free-running LO: Wiener phase with sigma_deg standard deviation per sample."""
    theta = np.cumsum(np.deg2rad(sigma_deg) * rng.standard_normal(len(x)))
    return x * np.exp(1j * theta)
