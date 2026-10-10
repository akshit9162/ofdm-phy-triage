# Triage report: `runs/test`

4000 test cases, **2574 failed**. 2474 triaged automatically, **100 sent to human review** (confidence < 0.9).

| Root cause | Failures | Mean confidence | Mean EVM miss (dB) |
|---|---:|---:|---:|
| `low_snr` | 703 | 0.997 | +6.9 |
| `stale_channel_estimate` | 339 | 0.999 | +18.4 |
| `timing_late` | 324 | 0.999 | +9.5 |
| `delay_spread_exceeds_cp` | 284 | 0.998 | +7.8 |
| `cfo_not_corrected` | 252 | 0.998 | +8.6 |
| `iq_imbalance` | 213 | 0.998 | +6.6 |
| `pa_clipping` | 186 | 0.997 | +6.2 |
| `phase_noise` | 173 | 0.997 | +6.0 |

## `low_snr` (703)

Expected failure: SNR too low for this MCS on this channel. Check the test config, not the DUT.

Most affected configs: BPSK / medium (99), BPSK / short (97), QPSK / medium (85)

Examples: `tc_00002`, `tc_00007`, `tc_00016`, `tc_00025`, `tc_00026`, `tc_00031`

## `stale_channel_estimate` (339)

Equalizer error follows the channel's fades: estimate not refreshed per packet.

Most affected configs: QPSK / short (49), 16QAM / short (48), 64QAM / short (46)

Examples: `tc_00005`, `tc_00009`, `tc_00029`, `tc_00066`, `tc_00074`, `tc_00095`

## `timing_late` (324)

Channel impulse response wraps to negative delay: FFT window starts past the CP.

Most affected configs: 64QAM / flat (38), QPSK / medium (33), 64QAM / short (32)

Examples: `tc_00030`, `tc_00032`, `tc_00056`, `tc_00062`, `tc_00069`, `tc_00082`

## `delay_spread_exceeds_cp` (284)

Channel taps beyond the 16-sample CP: ISI. Check the channel emulator profile.

Most affected configs: 16QAM / long (113), 64QAM / long (84), QPSK / long (58)

Examples: `tc_00013`, `tc_00023`, `tc_00050`, `tc_00051`, `tc_00055`, `tc_00073`

## `cfo_not_corrected` (252)

Common phase ramps linearly across symbols: CFO is estimated but not removed.

Most affected configs: 64QAM / medium (38), 16QAM / short (35), 64QAM / flat (32)

Examples: `tc_00017`, `tc_00058`, `tc_00061`, `tc_00080`, `tc_00119`, `tc_00128`

## `iq_imbalance` (213)

Error correlates with the mirror subcarrier: IQ gain/phase imbalance in the modulator.

Most affected configs: 64QAM / short (43), 64QAM / flat (41), 64QAM / medium (31)

Examples: `tc_00024`, `tc_00028`, `tc_00049`, `tc_00067`, `tc_00076`, `tc_00079`

## `pa_clipping` (186)

Constellation compressed (gain < 1, outer points shrink): PA driven into saturation.

Most affected configs: 64QAM / medium (33), 16QAM / flat (28), 64QAM / flat (27)

Examples: `tc_00008`, `tc_00044`, `tc_00070`, `tc_00077`, `tc_00139`, `tc_00152`

## `phase_noise` (173)

Common phase random-walks between symbols: LO phase noise.

Most affected configs: 64QAM / flat (30), 64QAM / medium (29), 64QAM / short (21)

Examples: `tc_00018`, `tc_00027`, `tc_00033`, `tc_00054`, `tc_00098`, `tc_00099`

## Needs human review (100)

| Test | Config | Best guess | Confidence |
|---|---|---|---:|
| `tc_00004` | 64QAM @ 27.3 dB, short | `pa_clipping` | 0.40 |
| `tc_03447` | BPSK @ 6.3 dB, medium | `low_snr` | 0.48 |
| `tc_02030` | 16QAM @ 18.8 dB, short | `phase_noise` | 0.50 |
| `tc_03961` | BPSK @ 13.6 dB, short | `stale_channel_estimate` | 0.50 |
| `tc_01312` | 64QAM @ 34.4 dB, short | `timing_late` | 0.52 |
| `tc_00092` | QPSK @ 10.2 dB, long | `delay_spread_exceeds_cp` | 0.52 |
| `tc_00270` | 64QAM @ 25.0 dB, medium | `phase_noise` | 0.52 |
| `tc_02194` | QPSK @ 12.8 dB, short | `low_snr` | 0.52 |
| `tc_02117` | BPSK @ 10.1 dB, short | `pa_clipping` | 0.53 |
| `tc_00940` | BPSK @ 5.9 dB, medium | `low_snr` | 0.55 |
| `tc_03683` | BPSK @ 5.8 dB, long | `delay_spread_exceeds_cp` | 0.55 |
| `tc_03893` | QPSK @ 18.3 dB, medium | `pa_clipping` | 0.55 |
| `tc_01022` | BPSK @ 9.4 dB, long | `delay_spread_exceeds_cp` | 0.55 |
| `tc_01461` | BPSK @ 19.2 dB, short | `iq_imbalance` | 0.56 |
| `tc_01427` | 16QAM @ 16.2 dB, medium | `phase_noise` | 0.56 |
| `tc_03762` | QPSK @ 9.4 dB, flat | `low_snr` | 0.58 |
| `tc_00729` | BPSK @ 10.0 dB, long | `low_snr` | 0.59 |
| `tc_00798` | QPSK @ 10.9 dB, medium | `phase_noise` | 0.59 |
| `tc_03352` | 16QAM @ 27.9 dB, short | `low_snr` | 0.60 |
| `tc_03727` | 16QAM @ 34.9 dB, medium | `phase_noise` | 0.60 |
| `tc_00824` | 16QAM @ 17.8 dB, short | `low_snr` | 0.61 |
| `tc_00686` | QPSK @ 10.8 dB, long | `delay_spread_exceeds_cp` | 0.61 |
| `tc_01459` | 16QAM @ 16.6 dB, medium | `iq_imbalance` | 0.62 |
| `tc_03585` | BPSK @ 10.8 dB, long | `delay_spread_exceeds_cp` | 0.62 |
| `tc_01275` | BPSK @ 6.9 dB, long | `delay_spread_exceeds_cp` | 0.63 |
