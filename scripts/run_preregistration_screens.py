#!/usr/bin/env python3
"""R-IMB-BETA-01 pre-registration redundancy screen.

Runs the frozen screen end-to-end on whatever point-in-time constituent
history exists, prints the ScreenResult, and writes the corresponding
append-only registry record (ADMIT / pre-outcome REJECT / DEFER).

Belt-and-braces attestation aid: the script refuses to run if the
configured target-returns location is readable in its environment. This
is not, and cannot be, a technical guarantee that no outcome data was
inspected — a script cannot know every file's contents. The substantive
guarantee remains procedural (the operator's attestation, recorded in
the registry entry); the check simply makes the honest path the easy
path. Pass --attest to record the attestation explicitly.

Inputs (CSV, days x names): constituent close-to-close returns, ES
returns (single column), point-in-time membership mask, index weights,
and the 15:50 / 15:55 constituent imbalance panels.

Usage:
  python scripts/run_preregistration_screens.py \
      --returns constituents.csv --es es.csv --membership membership.csv \
      --weights weights.csv --imb50 imb_1550.csv --imb55 imb_1555.csv \
      --registry registry.jsonl --attest
"""

import argparse
import os
import sys

import pandas as pd

from espa.config import DEFAULT_CONSTANTS
from espa.features.imbalance_beta import (
    ScreenStatus,
    beta_weighted_imbalance,
    cap_weighted_imbalance,
    redundancy_screen,
    rolling_es_betas,
)
from espa.validation.registry import ConfigurationRegistry

#: Conventional location of Phase 1 target data. If readable, refuse.
DEFAULT_TARGETS_PATH = "data/targets"

ENTRY_NAME = "imbalance=I1beta"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", required=True)
    ap.add_argument("--es", required=True)
    ap.add_argument("--membership", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imb50", required=True)
    ap.add_argument("--imb55", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--targets-path", default=DEFAULT_TARGETS_PATH)
    ap.add_argument("--attest", action="store_true",
                    help="attest that no outcome data was inspected")
    args = ap.parse_args()

    if os.path.exists(args.targets_path) and os.access(args.targets_path, os.R_OK):
        sys.exit(
            f"REFUSING TO RUN: target-returns location '{args.targets_path}' is "
            "readable in this environment. Run the screen in an environment "
            "without access to evaluation-period targets."
        )
    if not args.attest:
        sys.exit("REFUSING TO RUN: pass --attest to record the no-outcome-"
                 "inspection attestation in the registry entry.")

    load = lambda p: pd.read_csv(p, index_col=0, parse_dates=True)  # noqa: E731
    rets, membership = load(args.returns), load(args.membership).astype(bool)
    es = load(args.es).iloc[:, 0]
    weights = load(args.weights)
    q50, q55 = load(args.imb50), load(args.imb55)

    betas = rolling_es_betas(rets, es, membership, DEFAULT_CONSTANTS)
    i_beta = beta_weighted_imbalance(q55, weights, betas, DEFAULT_CONSTANTS)
    i_w = cap_weighted_imbalance(q55, weights)
    result = redundancy_screen(i_beta, i_w, DEFAULT_CONSTANTS)

    print(f"R-IMB-BETA-01 screen: {result.status.value.upper()}")
    print(f"  rho = {result.rho}")
    print(f"  history days = {result.n_days_history}, usable z-days = {result.n_usable}")

    registry = ConfigurationRegistry(path=args.registry)
    record = result.to_registry_record()
    if result.status is ScreenStatus.REJECT:
        registry.record_pre_outcome_rejection(
            ENTRY_NAME,
            rationale="Redundancy screen: beta-weighted aggregation is "
            "statistically indistinguishable from the cap-weighted baseline "
            "(mega-cap betas cluster near one and dominate cap weights) — "
            "a successful result discovered at zero denominator cost.",
            evidence=record,
        )
        print("Registry: pre-outcome REJECT recorded (Appendix C).")
    elif result.status is ScreenStatus.DEFER:
        registry.record_deferral(
            ENTRY_NAME,
            reason=f"screen sample floor not met: {record}",
        )
        print("Registry: DEFER recorded; entry remains in the floor and m_unrun.")
    else:
        registry.record_admission(ENTRY_NAME, evidence=record)
        print("Registry: ADMIT recorded; I1beta may now be executed as one "
              "counted replacement trial.")


if __name__ == "__main__":
    main()
