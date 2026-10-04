#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ZHU Wenbo (Yantai Vocational College of Culture and Tourism).
# SPDX-License-Identifier: Apache-2.0
"""Convert the raw extraction JSON (db_facts_real_vs_formula.json) into a
de-identified, publishable dataset CSV (data/real_vs_formula.csv).

De-identification
-----------------
The production audit log records a `host` field that reveals the internal
host-naming scheme (e.g. "host-rtx3090-97") and external dependencies
("host-ollama-100", "cloud-siliconflow"). We map every distinct host to a
neutral `node-N` label. The mapping is documented (anonymized) in
data/DATA.md so the scientific signal — distinct service times per distinct
node/GPU — is preserved without disclosing infrastructure topology, IP
addresses, or external endpoint names. No timestamps, request IDs, or payload
text are present in the source JSON, so nothing else needs stripping.

Run:  python scripts/make_dataset_csv.py
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "db_facts_real_vs_formula.json")
OUT = os.path.join(ROOT, "data", "real_vs_formula.csv")

# host label (as it appears in the audit log) -> neutral node id + description.
HOST_MAP = {
    "umr-local-59":     ("node-01", "Internal gateway node, RTX 5090 laptop (primary legalone:8b host)."),
    "host-rtx3090-97":  ("node-02", "Internal node, RTX 3090."),
    "host-rtx3090-98":  ("node-03", "Internal node, RTX 3090."),
    "host-rtx3080-69":  ("node-04", "Internal node, RTX 3080."),
    "host-rtx3080-99":  ("node-05", "Internal node, RTX 3080."),
    "host-teslap40-69": ("node-06", "Internal node, Tesla P40."),
    "cloud-siliconflow":("node-07", "Remote cloud embedding API (siliconflow)."),
    "host-ollama-100":  ("node-08", "External Ollama endpoint."),
    "host-ollama-161":  ("node-09", "External Ollama endpoint."),
    "host-ollama-170":  ("node-10", "External Ollama endpoint."),
    "host-ollama-42":   ("node-11", "External Ollama endpoint."),
}

FIELDS = [
    "model", "node", "n",
    "dur_min", "dur_p5", "dur_p50", "dur_p95", "dur_mean",
    "tot_p50", "q_p50", "q_frac",
]


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for key, stats in data["by_model_host"].items():
        model, _, host = key.partition(" | ")
        node, _desc = HOST_MAP.get(host, (host, ""))
        rows.append({
            "model": model,
            "node": node,
            "n": stats["n"],
            "dur_min": stats["dur_min"],
            "dur_p5": stats["dur_p5"],
            "dur_p50": stats["dur_p50"],
            "dur_p95": stats["dur_p95"],
            "dur_mean": round(stats["dur_mean"], 6),
            "tot_p50": stats["tot_p50"],
            "q_p50": stats["q_p50"],
            "q_frac": round(stats["q_frac"], 6),
        })
    # stable order: node, then model
    rows.sort(key=lambda r: (r["node"], r["model"]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
