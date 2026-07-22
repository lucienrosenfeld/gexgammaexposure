"""Registry amendment: predictive/non-predictive split, rejections, deferrals."""

from espa.config import RunConfig
from espa.validation.registry import (
    PRESPECIFIED_NONPREDICTIVE_ENTRIES,
    PRESPECIFIED_PREDICTIVE_CONFIGURATIONS,
    ConfigurationRegistry,
)

FLOOR = len(PRESPECIFIED_PREDICTIVE_CONFIGURATIONS)


def test_nonpredictive_entries_excluded_from_floor():
    assert "R-NEFF-01" in PRESPECIFIED_NONPREDICTIVE_ENTRIES
    assert "D-STAGE2-ID-01" in PRESPECIFIED_NONPREDICTIVE_ENTRIES
    for n in PRESPECIFIED_NONPREDICTIVE_ENTRIES:
        assert n not in PRESPECIFIED_PREDICTIVE_CONFIGURATIONS
    reg = ConfigurationRegistry()
    assert reg.trial_count() == FLOOR  # 18 with I1beta included


def test_pre_outcome_rejection_leaves_floor_and_unrun(tmp_path):
    reg = ConfigurationRegistry(path=tmp_path / "r.jsonl")
    assert reg.trial_count() == FLOOR
    assert reg.unrun_count(set()) == FLOOR
    reg.record_pre_outcome_rejection(
        "imbalance=I1beta",
        rationale="redundancy screen: |rho| > 0.90",
        evidence={"rho": 0.97},
    )
    assert reg.trial_count() == FLOOR - 1
    assert reg.unrun_count(set()) == FLOOR - 1
    # the record carries rule, statistic, and attestation
    rej = [e for e in reg.entries if e["type"] == "pre_outcome_rejection"][0]
    assert "rho" in rej["evidence"] and "attestation" in rej


def test_deferral_retained_in_floor_and_unrun(tmp_path):
    reg = ConfigurationRegistry(path=tmp_path / "r.jsonl")
    reg.record_deferral("imbalance=I1beta", reason="history below 250-day floor")
    assert reg.trial_count() == FLOOR  # deferred, not decided
    assert reg.unrun_count(set()) == FLOOR


def test_unrun_count_subtracts_executed():
    reg = ConfigurationRegistry()
    executed = {"rho=0.0", "imbalance=I1", "cross_asset=fixed"}
    assert reg.unrun_count(executed) == FLOOR - 3


def test_admission_gate_readable():
    reg = ConfigurationRegistry()
    assert not reg.is_admitted("imbalance=I1beta")
    reg.record_admission("imbalance=I1beta", evidence={"rho": 0.4})
    assert reg.is_admitted("imbalance=I1beta")


def test_append_only_through_new_record_types(tmp_path):
    p = tmp_path / "r.jsonl"
    reg = ConfigurationRegistry(path=p)
    reg.log(RunConfig(name="a"))
    reg.record_deferral("imbalance=I1beta", "floor")
    reg.record_pre_outcome_rejection("imbalance=I1beta", "screen", {"rho": 0.95})
    reg.record_admission("vol_conditioner_h_eq_1_null", {})
    reloaded = ConfigurationRegistry(path=p)
    assert [e["type"] for e in reloaded.entries] == [
        "run", "deferral", "pre_outcome_rejection", "admission",
    ]
    assert reloaded.trial_count() == FLOOR - 1


def test_executed_count_still_floors():
    reg = ConfigurationRegistry()
    for i in range(FLOOR + 3):
        reg.log(RunConfig(name=f"c{i}", rho=float(i)))  # distinct keys
    assert reg.trial_count() == FLOOR + 3
