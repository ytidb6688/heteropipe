# Copyright (c) 2026 ZHU Wenbo (Yantai Vocational College of Culture and Tourism).
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Add SPDX copyright headers to heteropipe source files (idempotent).

Run:  python scripts/add_headers.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PY_HEADER = (
    "# Copyright (c) 2026 ZHU Wenbo (Yantai Vocational College of Culture and Tourism).\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "\n"
)
HTML_HEADER = (
    "<!--\n"
    "  Copyright (c) 2026 ZHU Wenbo (Yantai Vocational College of Culture and Tourism).\n"
    "  SPDX-License-Identifier: Apache-2.0\n"
    "-->\n"
)

MARK = "Copyright (c) 2026 ZHU Wenbo"

PY_FILES = [
    "src/heteropipe/__init__.py",
    "src/heteropipe/core.py",
    "src/heteropipe/pipeline.py",
    "src/heteropipe/cli.py",
    "tests/test_core.py",
    "scripts/extract_tau_real.py",
    "scripts/plot_real_vs_formula.py",
    "scripts/make_dataset_csv.py",
]
HTML_FILES = [
    "calculator.html",
    "calculator_zh.html",
]


def prepend_py(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if MARK in content:
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(PY_HEADER + content)
    return True


def prepend_html(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if MARK in content:
        return False
    # Insert the header right after the <!DOCTYPE html> line.
    idx = content.find("\n", content.find("<!DOCTYPE"))
    if idx == -1:
        idx = 0
    new_content = content[: idx + 1] + HTML_HEADER + content[idx + 1 :]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)
    return True


def main():
    changed = []
    for rel in PY_FILES:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p) and prepend_py(p):
            changed.append(rel)
    for rel in HTML_FILES:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p) and prepend_html(p):
            changed.append(rel)
    print("changed:", changed if changed else "(none — headers already present)")


if __name__ == "__main__":
    main()
