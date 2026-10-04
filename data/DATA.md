# Dataset: real-vs-formula production traces

Empirical measurements that back **§7.2** and **Figure 5** of the companion
preprint *"Exact Makespan Bounds and the GPipe Homogeneity Bias for Heterogeneous
LLM Inference Pipelines"* (GitHub: `ytidb6688/heteropipe`).

## What this is

Per-(model, node) service-time statistics aggregated from **8,571 completed
inference requests** served by a production LLM gateway (the "inference-stack"
UIG → UMR pipeline: `legalone` + `bge-m3`, routed across heterogeneous GPUs).

Each row is one `(model, node)` combination with the percentile distribution of
per-request service time `duration_s` (≈ lower-bound stage time τ_j), end-to-end
time `total_s` (≈ realized makespan T_real), and queue wait `queue_wait_s`.

These numbers are the empirical basis for:

- **Corollary 1** (GPipe homogeneity bias): the constant `Σ(τ_max − τ_j)` that
  the homogeneous GPipe formula over-predicts by — `2.05 s` for `legalone:8b`
  across three GPU generations.
- **Corollary 5** (sandwich): `T_async ≤ T_real ≤ T_block` — verified because
  every completion satisfies `total_s ≥ duration_s`.

## Provenance

| Item | Value |
|---|---|
| Source | inference-stack UIG gateway `audit.log*` (JSON-lines, one record per request) |
| Collection window | 2026-09-03 → 2026-10-04 (UTC) |
| Raw log files scanned | 31 |
| Completed requests | 8,571 |
| Global queue-wait fraction | 61.78 % of requests waited in queue |
| Global queue-wait median / max | 0.001 s / 273.6 s |

Extraction is fully scripted and read-only: `scripts/extract_tau_real.py`
scans the logs and prints the aggregated JSON
(`db_facts_real_vs_formula.json`); `scripts/make_dataset_csv.py` turns that JSON
into this CSV.

## De-identification

The raw log records contain a `host` field (revealing the internal host-naming
scheme, e.g. `host-rtx3090-97`) and external dependencies
(`host-ollama-100`, `cloud-siliconflow`), plus a `request_id` and a `ts`
timestamp. **None of these appear in the published dataset:**

- `host` → a neutral `node-N` label (mapping below).
- `request_id`, `ts`, payload text, and any IP address are **dropped** during
  extraction (the extractor only reads `model`, `host`, `duration_s`,
  `total_s`, `queue_wait_s`).
- No timestamps are published, so individual requests cannot be re-linked to a
  time or to each other.

The mapping is intentionally one-way and documented only at the anonymized level
below; it preserves the scientific signal (distinct service times per distinct
node / GPU generation) without disclosing infrastructure topology, IP addresses,
or external endpoint names.

### Node dictionary (anonymized)

| node | description (de-identified) |
|---|---|
| node-01 | Internal gateway node, RTX 5090 laptop — primary `legalone:8b` host (n=5049) |
| node-02 | Internal node, RTX 3090 |
| node-03 | Internal node, RTX 3090 |
| node-04 | Internal node, RTX 3080 |
| node-05 | Internal node, RTX 3080 |
| node-06 | Internal node, Tesla P40 |
| node-07 | Remote cloud embedding API (siliconflow) |
| node-08 | External Ollama endpoint |
| node-09 | External Ollama endpoint |
| node-10 | External Ollama endpoint |
| node-11 | External Ollama endpoint |

## Data dictionary (`real_vs_formula.csv`)

| column | meaning |
|---|---|
| `model` | model tag as served (e.g. `legalone:8b`, `bge-m3:latest`, `qwen2.5:7b`) |
| `node` | anonymized host label (see node dictionary) |
| `n` | number of completed requests in this `(model, node)` cell |
| `dur_min` | min per-request `duration_s` (s) — empirical lower bound τ_j |
| `dur_p5` | 5th percentile of `duration_s` (s) |
| `dur_p50` | median `duration_s` (s) |
| `dur_p95` | 95th percentile of `duration_s` (s) |
| `dur_mean` | mean `duration_s` (s) |
| `tot_p50` | median end-to-end `total_s` (s) ≈ realized makespan |
| `q_p50` | median `queue_wait_s` (s) |
| `q_frac` | fraction of requests in this cell that incurred a queue wait |

## Reproduce

```bash
# 1. (on the source host) extract aggregated stats from the audit logs
python scripts/extract_tau_real.py > db_facts_real_vs_formula.json
# 2. (anywhere) build the de-identified CSV
python scripts/make_dataset_csv.py
# 3. (optional) re-render Figure 5
python scripts/plot_real_vs_formula.py
```

## License

The dataset is released under **Creative Commons Attribution 4.0
(CC-BY-4.0)** — separate from the code, which is Apache-2.0. Attribution to
ZHU Wenbo (Yantai Vocational College of Culture and Tourism) is requested when
the data is used. See `data/DATA_LICENSE.md`.
