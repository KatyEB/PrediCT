"""
evaluate.py — the OPTIONAL layer. Only runs when a reference exists.

A general user has no ground truth, so scoring must work without one. That is
why this is a separate file: score_volume() never touches it.

Two uses:
  * model vs annotation   — your 66-patient test set
  * model vs model        — A1 against A3, no ground truth anywhere

WHY RISK AGREEMENT IS THE HEADLINE, NOT DICE OR MEAN MAE
    Dice and mean absolute error are dominated by a handful of patients with
    very large calcium burdens. A method can lose on mean MAE and still be the
    better clinical tool, because what changes management is which risk band a
    patient lands in. Report risk accuracy first, median AE second, mean MAE
    with the outlier caveat attached.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scoring import RiskCategory, categorise, distance


# ---------------------------------------------------------------------------
# Mask overlap
# ---------------------------------------------------------------------------

def dice(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> float:
    """Dice coefficient. Returns 1.0 when both are empty.

    That convention matters: a patient with no calcium, correctly predicted as
    having none, is a perfect result. Scoring it 0 would drag the mean down for
    exactly the cases the model got right.
    """
    p = np.asarray(pred) > threshold
    t = np.asarray(truth) > threshold

    total = p.sum() + t.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, t).sum() / total)


# ---------------------------------------------------------------------------
# Agreement between two sets of scores
# ---------------------------------------------------------------------------

@dataclass
class Agreement:
    n: int
    mae: float
    median_ae: float
    bias: float
    pearson_r: float
    r_squared: float
    risk_accuracy: float
    risk_confusion: dict
    two_band_errors: int

    def report(self) -> str:
        lines = [
            f"n = {self.n}",
            "",
            f"  risk accuracy    {self.risk_accuracy * 100:.1f}%    "
            f"<- the clinically meaningful number",
            f"  two-band errors  {self.two_band_errors}",
            "",
            f"  median AE        {self.median_ae:.2f}    "
            f"<- typical patient",
            f"  mean MAE         {self.mae:.2f}    "
            f"<- dominated by large-burden outliers",
            f"  bias             {self.bias:+.2f}",
            f"  Pearson r        {self.pearson_r:.4f}  (R2 {self.r_squared:.3f})",
            "",
            "  risk confusion (rows = reference, cols = prediction)",
        ]
        header = "        " + "".join(f"{c:>10}" for c in RiskCategory.ORDER)
        lines.append(header)
        for row in RiskCategory.ORDER:
            cells = "".join(
                f"{self.risk_confusion.get((row, col), 0):>10}"
                for col in RiskCategory.ORDER
            )
            lines.append(f"  {row:>6}{cells}")
        return "\n".join(lines)


def agreement(predicted: list[float], reference: list[float]) -> Agreement:
    """Compare two aligned lists of Agatston scores."""
    if len(predicted) != len(reference):
        raise ValueError(
            f"{len(predicted)} predictions but {len(reference)} references"
        )
    if not predicted:
        raise ValueError("nothing to compare")

    p = np.asarray(predicted, dtype=float)
    t = np.asarray(reference, dtype=float)
    errors = p - t

    # A constant series has zero variance, so correlation is undefined rather
    # than zero. Reporting nan is honest; reporting 0.0 would look like a
    # finding.
    if p.std() == 0 or t.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(p, t)[0, 1])

    confusion: dict[tuple[str, str], int] = {}
    agreed = 0
    two_band = 0
    for pi, ti in zip(p, t):
        key = (categorise(ti), categorise(pi))
        confusion[key] = confusion.get(key, 0) + 1
        gap = distance(pi, ti)
        agreed += gap == 0
        two_band += gap >= 2

    return Agreement(
        n=len(p),
        mae=float(np.abs(errors).mean()),
        median_ae=float(np.median(np.abs(errors))),
        bias=float(errors.mean()),
        pearson_r=r,
        r_squared=float(r * r) if not math.isnan(r) else float("nan"),
        risk_accuracy=agreed / len(p),
        risk_confusion=confusion,
        two_band_errors=two_band,
    )


def mcnemar(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Exact McNemar test on paired risk-categorisation correctness.

    Only the discordant pairs carry information: cases where both methods agree
    tell you nothing about which is better. With few discordant pairs the test
    is underpowered, and this reports that plainly rather than leaving a
    non-significant p-value to be misread as evidence of no difference.
    """
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if bb and not a)
    n = b + c

    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0,
                "note": "no discordant pairs; the methods never disagree"}

    # Two-sided exact binomial test at p = 0.5.
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_value = min(1.0, 2 * tail)

    note = ""
    if n < 10:
        note = (f"only {n} discordant pairs; underpowered. A non-significant "
                "result here is not evidence of no difference.")

    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value, "note": note}


# ---------------------------------------------------------------------------
# Comparing results.csv files
# ---------------------------------------------------------------------------

def load_scores(csv_path: str | Path, model_id: str | None = None
                ) -> dict[str, float]:
    """Read {patient_id: agatston} from a results.csv, optionally for one model."""
    scores: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            if model_id and row.get("model_id") != model_id:
                continue
            scores[row["patient_id"]] = float(row["agatston"])
    return scores


def compare(csv_path: str | Path, model_a: str, model_b: str) -> Agreement:
    """Compare two models over the patients both scored successfully."""
    a = load_scores(csv_path, model_a)
    b = load_scores(csv_path, model_b)

    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(f"no patients scored by both {model_a} and {model_b}")

    missing = (set(a) | set(b)) - set(shared)
    if missing:
        print(f"note: {len(missing)} patient(s) scored by only one model, excluded")

    return agreement([a[k] for k in shared], [b[k] for k in shared])


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Compare two models in a results.csv")
    p.add_argument("csv")
    p.add_argument("model_a")
    p.add_argument("model_b")
    args = p.parse_args()

    print(f"\n{args.model_a} vs {args.model_b}\n")
    print(compare(args.csv, args.model_a, args.model_b).report())
    print()


if __name__ == "__main__":
    main()
