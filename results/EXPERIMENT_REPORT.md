# Experiment Report

## Privacy-Utility Results

| Policy | Leak Total | Utility Proxy | Mean Latency (ms) | P90 Latency (ms) |
| --- | --- | --- | --- | --- |
| raw | 3.0256 | 1.0 | 0.072 | 0.09 |
| weak | 2.0 | 0.512195 | 0.076 | 0.089 |
| pseudo | 0.5128 | 1.0 | 0.087 | 0.102 |
| redact | 0.5128 | 0.505051 | 0.082 | 0.102 |
| adaptive | 0.5641 | 1.0 | 0.962 | 0.995 |

## Leakage Breakdown

| Policy | Text | ASR | Image | Waveform | Audio |
| --- | --- | --- | --- | --- | --- |
| raw | 0.5128 | 0.5128 | 0.7436 | 0.5128 | 0.7436 |
| weak | 0.5128 | 0.0 | 0.7436 | 0.0 | 0.7436 |
| pseudo | 0.5128 | 0.0 | 0.0 | 0.0 | 0.0 |
| redact | 0.5128 | 0.0 | 0.0 | 0.0 | 0.0 |
| adaptive | 0.359 | 0.0 | 0.1026 | 0.0 | 0.1026 |

## Latency Summary

| Policy | Mean (ms) | P50 (ms) | P90 (ms) |
| --- | --- | --- | --- |
| raw | 0.072 | 0.072 | 0.09 |
| weak | 0.076 | 0.08 | 0.089 |
| pseudo | 0.087 | 0.089 | 0.102 |
| redact | 0.082 | 0.088 | 0.102 |
| adaptive | 0.962 | 0.881 | 0.995 |

## Adaptive Policy Notes

Cross-modal synergy triggered localized retokenization 2 time(s).

### Output Files
- `audit_log_signed_adaptive.jsonl` (if audit signing enabled)
- `audit_checkpoints_adaptive.jsonl` (if audit signing enabled)
- `audit_fhir_adaptive.jsonl` (if audit signing enabled)
- `dcpg_snapshot.json` (if enabled)
- `dcpg_crdt_demo.json` (if enabled)
- `rl_reward_stats.json` (if enabled)
- `sample_dag.dot` / `sample_dag.json`
- `privacy_utility_curve.png` (if matplotlib installed)
