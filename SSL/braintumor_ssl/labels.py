"""IDH label + clinical-covariate loading for the downstream probe.

Only UCSF-PDGM is wired up so far, from the release's own `UCSF-PDGM-metadata_v5.csv`.

Two things here are not boilerplate and are the reason this is a module rather than three lines
inline:

1. **The ID join is a trap.** The metadata's `ID` column is zero-padded to THREE digits
   (`UCSF-PDGM-004`) while the image directories use FOUR (`UCSF-PDGM-0004`). A naive join matches
   6 of 501 subjects and fails silently — you get a tiny cohort and a meaningless AUC rather than an
   error. `load_ucsf_labels` normalizes the numeric suffix and reports the match count.

2. **IDH is nearly collinear with WHO grade in this cohort**, so the covariates travel WITH the
   label. Of 402 grade-4 subjects 374 are wildtype; of 56 grade-2 subjects 46 are mutant. On this
   metadata alone, with no imaging whatsoever, age gives AUC 0.90 and grade+age 0.96. An imaging
   AUC reported without that baseline next to it says nothing, so `age` / `sex` / `grade` are always
   returned and the probe reports the clinical-only arm alongside the image arm.

Label definition: `IDH == "wildtype"` -> 0, everything else (IDH1 p.R132H, "mutated (NOS)", the rarer
R132C/G/S and IDH2 variants) -> 1. Subtype resolution is not supported by the counts.
"""
from __future__ import annotations

import csv
import os
import re

UCSF_METADATA = "/datasets/mri_datasets/UCSF-PDGM-metadata_v5.csv"

_NUM = re.compile(r"^(?P<stem>.*?)(?P<num>\d+)$")


def normalize_id(subject_id: str, width: int = 4) -> str:
    """`UCSF-PDGM-004` and `UCSF-PDGM-0004` both -> `UCSF-PDGM-0004` (see module docstring)."""
    m = _NUM.match(subject_id.strip())
    return f"{m.group('stem')}{m.group('num').zfill(width)}" if m else subject_id.strip()


def _num(v: str) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return float("nan")


def load_ucsf_labels(path: str = UCSF_METADATA, verbose: bool = True) -> dict[str, dict]:
    """{normalized_subject_id: {idh, idh_raw, age, sex, grade, diagnosis}} for UCSF-PDGM.

    `idh` is the binary target (1 = mutant). Rows whose IDH cell is empty/unknown are dropped, so a
    missing molecular result never silently becomes a wildtype label.
    """
    if not os.path.exists(path):
        raise SystemExit(f"UCSF metadata not found at {path} — download it from the TCIA release.")
    out: dict[str, dict] = {}
    dropped = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            raw = (r.get("IDH") or "").strip()
            if not raw or raw.lower() in ("unknown", "na", "n/a"):
                dropped += 1
                continue
            out[normalize_id(r["ID"])] = {
                "idh": 0 if raw.lower() == "wildtype" else 1,
                "idh_raw": raw,
                "age": _num(r.get("Age at MRI")),
                "sex": 1.0 if (r.get("Sex") or "").strip().upper().startswith("M") else 0.0,
                "grade": _num(r.get("WHO CNS Grade")),
                "diagnosis": (r.get("Final pathologic diagnosis (WHO 2021)") or "").strip(),
            }
    if verbose:
        pos = sum(v["idh"] for v in out.values())
        print(f"[labels] UCSF-PDGM: {len(out)} labelled subjects "
              f"({pos} mutant / {len(out) - pos} wildtype, {100 * pos / max(1, len(out)):.1f}% positive)"
              + (f"; {dropped} row(s) dropped for missing IDH" if dropped else ""))
    return out


def attach(records: list[dict], labels: dict[str, dict], verbose: bool = True) -> tuple[list[dict], list[dict]]:
    """Join image records to labels on the normalized id; returns (matched_records, matched_labels).

    Loud about the join rate on purpose — a silent 6/501 match is exactly the failure this module
    exists to prevent.
    """
    recs, labs = [], []
    for r in records:
        key = normalize_id(r["id"])
        if key in labels:
            recs.append(r)
            labs.append(labels[key])
    if verbose:
        rate = 100 * len(recs) / max(1, len(records))
        print(f"[labels] matched {len(recs)}/{len(records)} image records ({rate:.1f}%)")
        if rate < 90:
            print("[labels] WARNING: low join rate — check the subject-id format on both sides "
                  "(metadata is 3-digit, image dirs are 4-digit).")
    return recs, labs


if __name__ == "__main__":       # quick inspection: distribution + the clinical-only baselines
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=UCSF_METADATA)
    args = ap.parse_args()
    labs = load_ucsf_labels(args.metadata)

    def auc(score, y):
        pos = [s for s, t in zip(score, y) if t == 1]
        neg = [s for s, t in zip(score, y) if t == 0]
        if not pos or not neg:
            return float("nan")
        return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

    y = [v["idh"] for v in labs.values()]
    print(f"[labels] clinical-only baselines (no imaging):")
    print(f"           age   (younger -> mutant): AUC {auc([-v['age'] for v in labs.values()], y):.3f}")
    print(f"           grade (lower   -> mutant): AUC {auc([-v['grade'] for v in labs.values()], y):.3f}")
    print("         Any imaging AUC must be reported next to these.")
