"""The configuration registry (Section 22).

Every configuration ever run is logged, including abandoned ones,
because the denominator of the multiple-testing correction is the honest
count. Additions during research are appended, never substituted — the
registry file is append-only JSON lines, and nothing in this class can
delete an entry.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from espa.config import RunConfig

#: The registry as it stood at specification time (Section 22). These are
#: counted from day one whether or not they have been run yet.
PRESPECIFIED_CONFIGURATIONS: tuple[str, ...] = (
    "rho=0.0",
    "rho=0.25",
    "rho=0.5",
    "cross_asset=fixed",
    "cross_asset=no_ty",
    "cross_asset=pca",
    "imbalance=I1",
    "imbalance=I2",
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


@dataclass
class ConfigurationRegistry:
    path: Path | None = None
    _entries: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is not None and Path(self.path).exists():
            with open(self.path) as f:
                self._entries = [json.loads(line) for line in f if line.strip()]

    def log(self, config: RunConfig, result: dict | None = None) -> None:
        entry = {
            "name": config.name,
            "key": list(config.key()),
            "config": _as_jsonable(config),
            "result": result,
        }
        self._entries.append(entry)
        if self.path is not None:
            with open(self.path, "a") as f:  # append-only, by construction
                f.write(json.dumps(entry, default=str) + "\n")

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def trial_count(self) -> int:
        """The honest multiple-testing denominator.

        Distinct configuration keys ever logged, floored at the
        prespecified registry size — running fewer configurations than
        were prespecified does not shrink the denominator, because the
        prespecified alternatives were all live hypotheses.
        """
        distinct = {tuple(e["key"]) for e in self._entries}
        return max(len(distinct), len(PRESPECIFIED_CONFIGURATIONS))


def _as_jsonable(config: RunConfig) -> dict:
    d = dataclasses.asdict(config)
    return json.loads(json.dumps(d, default=str))
