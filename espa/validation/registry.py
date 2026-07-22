"""The configuration registry (Section 22, amended by the research round).

Every configuration ever run is logged, including abandoned ones,
because the denominator of the multiple-testing correction is the honest
count. Additions during research are appended, never substituted — the
registry file is append-only JSON lines, and nothing in this class can
delete an entry.

Research-round amendment: the prespecified list is split into
*predictive* configurations (which floor the trial count) and
*non-predictive* entries (methodological corrections and diagnostics,
which must neither inflate nor dilute the denominator). Two new
append-only record types exist: pre-outcome rejections (a configuration
rejected under a prespecified rule before any outcome inspection — the
redundancy screen, data infeasibility) which leave both the floor and
``m_unrun``; and deferrals (the screen-sample-floor case) which stay in
both until resolved.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from espa.config import RunConfig

#: Predictive configurations as prespecified (Section 22 plus
#: R-IMB-BETA-01). These floor the multiple-testing denominator from day
#: one whether or not they have been run; a name leaves the floor only
#: through a recorded pre-outcome rejection.
PRESPECIFIED_PREDICTIVE_CONFIGURATIONS: tuple[str, ...] = (
    "rho=0.0",
    "rho=0.25",
    "rho=0.5",
    "cross_asset=fixed",
    "cross_asset=no_ty",
    "cross_asset=pca",
    "imbalance=I1",
    "imbalance=I2",
    "imbalance=I1beta",
    "post_1600_imbalance_message_test",
    "absorption_interaction",
    "full_day_return_stage1",
    "standalone_gamma_density_admission",
    "vol_conditioner_h_eq_1_null",
    "overnight_horizon_exit",
    "boosted_model_comparison",
    "kernel_lambda_s",
    "kernel_lambda_t",
)

#: Methodological corrections and measurement infrastructure. Logged for
#: provenance; excluded from the trial denominator in both directions.
PRESPECIFIED_NONPREDICTIVE_ENTRIES: tuple[str, ...] = (
    "R-NEFF-01",
    "D-STAGE2-ID-01",
)

#: Back-compatibility alias (pre-amendment name).
PRESPECIFIED_CONFIGURATIONS = PRESPECIFIED_PREDICTIVE_CONFIGURATIONS


@dataclass
class ConfigurationRegistry:
    path: Path | None = None
    _entries: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is not None and Path(self.path).exists():
            with open(self.path) as f:
                self._entries = [json.loads(line) for line in f if line.strip()]

    def _append(self, entry: dict) -> None:
        self._entries.append(entry)
        if self.path is not None:
            with open(self.path, "a") as f:  # append-only, by construction
                f.write(json.dumps(entry, default=str) + "\n")

    def log(self, config: RunConfig, result: dict | None = None) -> None:
        self._append(
            {
                "type": "run",
                "name": config.name,
                "key": list(config.key()),
                "config": _as_jsonable(config),
                "result": result,
            }
        )

    def record_pre_outcome_rejection(
        self, name: str, rationale: str, evidence: dict
    ) -> None:
        """A configuration rejected under a prespecified rule, pre-outcome.

        The record must show the frozen rule, the computed statistic, and
        an attestation that no outcome data was inspected. Rejected
        entries leave ``trial_count()``'s floor and ``m_unrun``.
        """
        self._append(
            {
                "type": "pre_outcome_rejection",
                "name": name,
                "rationale": rationale,
                "evidence": evidence,
                "attestation": "no outcome data (targets, losses, Sharpes, "
                "coefficients) was inspected before this rejection",
            }
        )

    def record_deferral(self, name: str, reason: str) -> None:
        """Screen-sample-floor case: deferred, not decided.

        The entry stays in the predictive floor and in ``m_unrun`` until
        resolved by a later ADMIT or pre-outcome rejection.
        """
        self._append({"type": "deferral", "name": name, "reason": reason})

    def record_admission(self, name: str, evidence: dict) -> None:
        """Screen ADMIT: the configuration may now be executed."""
        self._append({"type": "admission", "name": name, "evidence": evidence})

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def pre_outcome_rejected_names(self) -> set[str]:
        return {e["name"] for e in self._entries if e.get("type") == "pre_outcome_rejection"}

    def admitted_names(self) -> set[str]:
        return {e["name"] for e in self._entries if e.get("type") == "admission"}

    def is_admitted(self, name: str) -> bool:
        return name in self.admitted_names()

    def trial_count(self) -> int:
        """The honest multiple-testing denominator.

        Distinct executed configuration keys, floored at the prespecified
        *predictive* list minus recorded pre-outcome rejections. Running
        fewer configurations than were prespecified does not shrink the
        denominator; non-predictive entries never enter it; a pre-outcome
        rejection under a frozen rule is the only exit.
        """
        distinct = {
            tuple(e["key"]) for e in self._entries if e.get("type", "run") == "run"
        }
        rejected = self.pre_outcome_rejected_names()
        floor = len(
            [n for n in PRESPECIFIED_PREDICTIVE_CONFIGURATIONS if n not in rejected]
        )
        return max(len(distinct), floor)

    def unrun_count(self, executed_names: set[str]) -> int:
        """m_unrun for R-NEFF-01: prespecified predictive entries neither
        executed nor pre-outcome rejected. Deferred entries remain
        counted — declining (or being unable) to run a candidate can
        never lower the denominator.
        """
        rejected = self.pre_outcome_rejected_names()
        return len(
            [
                n
                for n in PRESPECIFIED_PREDICTIVE_CONFIGURATIONS
                if n not in executed_names and n not in rejected
            ]
        )


def _as_jsonable(config: RunConfig) -> dict:
    d = dataclasses.asdict(config)
    return json.loads(json.dumps(d, default=str))
