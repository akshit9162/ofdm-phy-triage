# OFDM PHY Validation + ML Failure Triage

**Project page:** https://akshit9162.github.io/ofdm-phy-triage

An 802.11a-style OFDM link simulator (Python/NumPy) built as a **validation harness**.
Every test case runs a packet through a channel, measures it the way test equipment would
(EVM, channel impulse response, pilot phase, constellation statistics), judges it against
an EVM limit and writes a plain-text log. A **triage model then reads only those logs** and
names the root cause of each failure: one of seven injected RF/receiver faults, or "the SNR
was just too low for this modulation" (a config problem, not a bug).

## Results (independent test campaign, seed 1)

The model is trained on one campaign (8,000 cases, seed 0) and tested on a separately
generated one (4,000 cases, seed 1). Accuracy and macro-F1 are over **1,554 failures with a
single root cause**.

| Triage | Accuracy | Macro-F1 | Held-out 64QAM accuracy (never seen in training) |
|---|---:|---:|---:|
| Hand-ordered rules (thresholds fit on train) | 84.9% | 0.839 | 86.0% |
| Gradient-boosted trees on the same 17 features | **99.2%** | **0.992** | 94.5% |

- **The PHY matches theory.** With perfect channel knowledge, simulated uncoded BER sits on the
  closed-form Gray-QAM curves for BPSK through 64QAM (a unit test checks this to within
  0.15 decades). The practical receiver (two-symbol LS channel estimate, CFO correction,
  pilot phase tracking) loses about 2 dB, close to the 1.8 dB that two-symbol LS
  estimation alone costs in theory.
- **Rules miss the subtle faults.** Phase noise (51% recall) and PA clipping (74%) overlap with
  ordinary noise once a single threshold is put on them. The trees combine features and get 98% and 97%.
- **Generalization gap.** Trained without 64QAM, clipping recall on 64QAM falls to **58%**: the
  amount of constellation compression depends on the modulation.
- **Multi-cause failures are the real problem.** 1,020 of the 2,574 test failures (40%) still
  fail when the injected fault is removed, because the SNR was also too low. The model still gives
  **91% of them ≥ 0.9 confidence**: it names the injected fault 67% of the time and low SNR
  32% of the time, and never says "both". Confidence does rank them lower: of the least-confident
  10% of all failures, 89% are multi-cause. But deferring that 10% to a human catches only **23%**
  of the multi-cause cases.

## The eight root causes

| Label | What was injected | Signature the logs expose |
|---|---|---|
| `low_snr` | nothing: SNR below what the MCS needs on this channel | error ≈ what the LTF noise estimate predicts |
| `cfo_not_corrected` | CFO estimated but never removed | pilot common phase ramps linearly per symbol |
| `stale_channel_estimate` | equalizer uses a previous packet's channel | per-subcarrier error follows the old channel's fades |
| `timing_late` | FFT window starts 1–8 samples past the CP | channel impulse response wraps to negative delay |
| `pa_clipping` | soft-limiter PA, −2 to +4 dB above RMS | constellation gain < 1, outer points compressed |
| `iq_imbalance` | modulator gain/phase imbalance | error correlates with the mirror subcarrier's symbol |
| `phase_noise` | Wiener LO phase noise | pilot common phase random-walks |
| `delay_spread_exceeds_cp` | 28-tap channel, CP is 16 | impulse-response energy beyond the CP |

Ground truth is counterfactual: each failing case is re-run with the same random streams and
the fault removed. If it then passes, the fault is the root cause. If it still fails, the case
is marked **ambiguous** and kept out of training and scoring.

## Layout

```
ofdmtriage/phy.py       Gray QAM, 802.11a framing (64-FFT, CP 16, 48 data + 4 pilots, LTF), receiver
ofdmtriage/channel.py   multipath (exponential PDP Rayleigh), CFO, AWGN, clipping, IQ imbalance, phase noise
ofdmtriage/validate.py  test cases, fault injection, measurements, log writer, counterfactual labels
ofdmtriage/triage.py    log parser, 17 physics features, rule triage, gradient-boosted triage
scripts/                campaign runner, evaluation, triage report, BER-vs-theory figure
tests/                  31 pytest tests (BER vs theory, LS loss, ISI floor, log round-trip, each fault)
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q
python -m scripts.campaign 8000 runs/main 0      # ~90 s
python -m scripts.campaign 4000 runs/test 1
python -m scripts.evaluate runs/main runs/test results
python -m scripts.triage_report runs/main runs/test results/triage_report.md
python -m scripts.ber_curves results/ber_awgn.png
```

CI runs the unit tests, a smaller campaign, evaluation and the triage report on every push, and
posts the report to the job summary.

## Limits

Uncoded, single-antenna, single-packet link with EVM as the only pass criterion. Each case has
at most one injected fault, and the impairment models are textbook models, not measured
hardware. The 99% figure says the features separate these simulated faults. It does not say a real
lab's failures would triage that cleanly. The 40% ambiguous share is the more realistic warning.
