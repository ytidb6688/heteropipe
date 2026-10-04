# Copyright (c) 2026 ZHU Wenbo (Yantai Vocational College of Culture and Tourism).
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-(model,host) service-time statistics from inference-stack audit logs.
Read-only: scans ~/AI-Stack/projects/inference-stack/uig/logs/audit.log* and prints JSON.
Used to build the "closed-form vs real scheduler" comparison for the preprint.
"""
import json, glob, os, statistics as st
from collections import defaultdict

LOGDIR = os.path.expanduser("~/AI-Stack/projects/inference-stack/uig/logs")
PATTERNS = ["audit.log", "audit.log.*"]

def pct(vals, q):
    if not vals: return None
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    f = int(k); c = min(f + 1, len(vals) - 1)
    if f == c: return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)

def main():
    files = []
    for p in PATTERNS:
        files += glob.glob(os.path.join(LOGDIR, p))
    files = sorted(set(files))
    # match completion records by request_id
    completions = {}  # request_id -> dict
    starts = {}
    for fp in files:
        try:
            fh = open(fp, "r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msg = rec.get("msg", "")
                rid = rec.get("request_id")
                if msg == "请求开始" and rid:
                    starts[rid] = rec
                elif msg == "请求完成" and rid:
                    completions[rid] = rec

    # per (model,host) duration / total / queue stats
    by_mh = defaultdict(lambda: {"dur": [], "tot": [], "q": []})
    n_complete = 0
    n_queued = 0
    all_q = []
    for rid, rec in completions.items():
        dur = rec.get("duration_s"); tot = rec.get("total_s"); q = rec.get("queue_wait_s")
        model = rec.get("model"); host = rec.get("host")
        if dur is None or model is None or host is None:
            continue
        n_complete += 1
        by_mh[(model, host)]["dur"].append(float(dur))
        if tot is not None:
            by_mh[(model, host)]["tot"].append(float(tot))
        if q is not None:
            qf = float(q); by_mh[(model, host)]["q"].append(qf)
            all_q.append(qf)
            if qf > 0: n_queued += 1

    out = {"n_files": len(files), "n_complete": n_complete,
           "global_queue_frac": (n_queued / n_complete) if n_complete else None,
           "global_queue_median": pct(all_q, 0.5),
           "global_queue_max": max(all_q) if all_q else None,
           "by_model_host": {}}
    for (model, host), d in sorted(by_mh.items()):
        dur = d["dur"]; tot = d["tot"]; q = d["q"]
        out["by_model_host"][f"{model} | {host}"] = {
            "n": len(dur),
            "dur_min": min(dur), "dur_p5": pct(dur, 0.05), "dur_p50": pct(dur, 0.5),
            "dur_p95": pct(dur, 0.95), "dur_mean": st.mean(dur),
            "tot_p50": pct(tot, 0.5) if tot else None,
            "q_p50": pct(q, 0.5) if q else None,
            "q_frac": (sum(1 for x in q if x > 0) / len(q)) if q else None,
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
