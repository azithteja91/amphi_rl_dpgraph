

```mermaid
flowchart LR
%% === Inputs ===
A["Streaming Inputs\nText notes / ASR transcripts / Image proxies / Waveform headers / Audio features"] --> B

%% === Cross-modal + longitudinal state ===
B["Normalization + PHI Signals\n(modality-specific detectors / proxy flags)"] --> C

C["Entity Resolver\n(assign to entity_key, token version)"] --> D

D["Exposure State Store\n(Longitudinal accumulation over time)\n- cumulative exposure\n- recency-aware memory\n- cross-modal link signals"] --> E

%% === Cross-modal aggregation / graph abstraction ===
E["Cross-Modal Aggregation Layer\n(links + co-occurrence)\nRisk Graph Summary (optional)"] --> F

%% === Risk + decision ===
F["Risk Engine\n(entity-level risk score)\n+ escalation triggers"] --> G

G["Adaptive Policy Controller\n(state-dependent choice)\nraw / weak / pseudo / redact / synthetic"] --> H

%% === Execution ===
H["Masking Execution Layer\n(CMO registry / compiled DAG)\nmodality-aware transforms"] --> I

I["Sanitized Output Stream\n(masked text + transformed proxies)\n+ downstream-safe artifacts"] --> J

%% === Auditability ===
J["Structured Audit Log\n(reproducible decisions)\npolicy, risk, latency, linkage signals"] --> K

%% === Feedback / lifecycle ===
K["Replay + Evaluation\n(leakage checks, utility proxies,\noptional RL reward logging)"] --> D

