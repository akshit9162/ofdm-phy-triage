"""PHY sanity figure: simulated uncoded BER on AWGN vs closed-form theory.

usage: python -m scripts.ber_curves OUT.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ofdmtriage import channel as ch
from ofdmtriage.phy import RxConfig, build_frame, receive
from tests.test_phy import theory_ber

GENIE = RxConfig(correct_cfo=False, track_pilots=False, genie_channel=np.ones(52))


def sim_ber(mod, snr, cfg, rng, min_err=300, max_bits=3e6):
    errs = bits = 0
    while errs < min_err and bits < max_bits:
        f = build_frame(mod, 40, rng)
        rx = receive(ch.awgn(f.samples, snr, rng), mod, 40, cfg)
        errs += int(np.sum(rx.bits != f.bits))
        bits += f.bits.size
    return errs / bits


def main(out):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    colors = {"BPSK": "#6366f1", "QPSK": "#0ea5e9", "16QAM": "#10b981", "64QAM": "#f59e0b"}
    for mod, c in colors.items():
        snr = np.arange(0, 31, 2.0)
        th = theory_ber(mod, snr)
        snr = snr[th > 3e-5]
        ax.semilogy(snr, theory_ber(mod, snr), "-", color=c, lw=1.6, label=f"{mod} theory")
        ax.semilogy(snr, [sim_ber(mod, s, GENIE, rng) for s in snr], "o", color=c, ms=5,
                    label=f"{mod} sim, perfect CSI")
        ax.semilogy(snr, [sim_ber(mod, s, RxConfig(), rng) for s in snr], "x", color=c, ms=6,
                    label=f"{mod} sim, LS estimate + CFO/pilot tracking")
    ax.set_xlabel("per-subcarrier Es/N0 (dB)")
    ax.set_ylabel("uncoded BER")
    ax.set_ylim(1e-5, 0.6)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("OFDM PHY on AWGN: simulation vs theory", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
