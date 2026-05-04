# SPC + Anomaly Detection Report

## Control Limits

- Center line: 100.105
- Sigma estimate: 0.937
- UCL: 102.917
- LCL: 97.293

## Signal Counts

- SPC signals: 279
- ML anomalies: 20
- Combined signals: 282

## SPC Rule Counts

- Rule 1, beyond 3 sigma: 231
- Rule 2, beyond 2 sigma: 266
- Rule 3, 8 points same side: 229
- Rule 4, 6-point trend: 2

## Suggested Review

Review rows where `combined_signal` is true, then classify each signal as special-cause, expected transition, or false positive.
