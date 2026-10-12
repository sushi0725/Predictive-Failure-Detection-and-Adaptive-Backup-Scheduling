"""
Decision Layer — Step 4 of the project pipeline
================================================
Maps a predicted failure-risk probability to a concrete backup / protection
action. This is the "adaptive scheduling" half of the project: instead of
backing everything up on a fixed cron schedule (reactive), backup frequency
and urgency scale with predicted risk (predictive).

Design note for the project defense:
    The original proposal used an MDP + reinforcement learning agent here.
    This module implements the equivalent *policy* as explicit tiered rules.
    It is deterministic, auditable, and trivially explainable -- which is
    exactly what an ops team would actually deploy. An RL agent can be
    dropped in later behind the same `decide()` interface without any
    changes to the dashboard or the logging schema.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict


# ----------------------------------------------------------------------
# Action definitions
# ----------------------------------------------------------------------

class Action(str, Enum):
    NO_ACTION = "no_action"
    INCREASE_MONITORING = "increase_monitoring"
    SCHEDULE_INCREMENTAL = "schedule_incremental_backup"
    IMMEDIATE_SNAPSHOT = "immediate_full_snapshot"
    PREPARE_FAILOVER = "prepare_failover"


class Urgency(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Decision:
    """One decision for one drive/server at one point in time."""
    serial_number: str
    risk_score: float
    risk_tier: str
    action: Action
    urgency: Urgency
    backup_interval_hours: int
    next_backup_due: datetime
    reason: str
    top_factors: Optional[List[str]] = None  # filled in from SHAP, if available

    def to_row(self) -> Dict:
        """Flatten for insertion into the backup_events / decisions DB table."""
        d = asdict(self)
        d["action"] = self.action.value
        d["urgency"] = self.urgency.value
        d["next_backup_due"] = self.next_backup_due.isoformat()
        d["top_factors"] = ", ".join(self.top_factors) if self.top_factors else None
        return d


# ----------------------------------------------------------------------
# Policy configuration
# ----------------------------------------------------------------------
# Each tier: (lower_bound_inclusive, tier_name, action, urgency,
#             backup_interval_hours, human_readable_reason)
#
# Tuning these thresholds IS the experiment for this part of the project:
# lower thresholds  -> more backups, more storage/IO cost, fewer missed failures
# higher thresholds -> cheaper, but more failures slip through unprotected
# Report a threshold-vs-cost tradeoff table in your final write-up.

POLICY_TIERS = [
    (0.80, "critical", Action.PREPARE_FAILOVER,      Urgency.CRITICAL, 2,
     "Failure highly likely within the prediction horizon; snapshot now and stage a replacement."),
    (0.50, "high",     Action.IMMEDIATE_SNAPSHOT,    Urgency.HIGH,     6,
     "Elevated failure probability; take a full snapshot immediately rather than waiting for the next window."),
    (0.25, "moderate", Action.SCHEDULE_INCREMENTAL,  Urgency.MEDIUM,   12,
     "Degradation trend detected; tighten the incremental backup interval."),
    (0.10, "low",      Action.INCREASE_MONITORING,   Urgency.LOW,      24,
     "Mild anomaly in telemetry; sample this drive more frequently but keep the normal backup cadence."),
    (0.00, "nominal",  Action.NO_ACTION,             Urgency.NONE,     168,
     "Telemetry within normal range; standard weekly backup schedule is sufficient."),
]


# ----------------------------------------------------------------------
# Core decision function
# ----------------------------------------------------------------------

def decide(serial_number: str,
           risk_score: float,
           now: Optional[datetime] = None,
           top_factors: Optional[List[str]] = None) -> Decision:
    """
    Convert a risk probability in [0, 1] into a concrete protection action.

    Args:
        serial_number: drive/server identifier
        risk_score:    P(failure within horizon), from the ML model
        now:           timestamp for scheduling (defaults to current time)
        top_factors:   optional list of top SHAP features, for the audit trail

    Returns:
        A Decision object ready to display or write to the database.
    """
    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(f"risk_score must be in [0, 1], got {risk_score}")

    now = now or datetime.now()

    for threshold, tier, action, urgency, interval_h, reason in POLICY_TIERS:
        if risk_score >= threshold:
            return Decision(
                serial_number=serial_number,
                risk_score=round(risk_score, 4),
                risk_tier=tier,
                action=action,
                urgency=urgency,
                backup_interval_hours=interval_h,
                next_backup_due=now + timedelta(hours=interval_h),
                reason=reason,
                top_factors=top_factors,
            )

    # unreachable: the 0.00 tier always matches
    raise RuntimeError("No policy tier matched; check POLICY_TIERS ordering.")


def decide_batch(risk_scores: Dict[str, float],
                 now: Optional[datetime] = None,
                 factors: Optional[Dict[str, List[str]]] = None) -> List[Decision]:
    """Run `decide` across a whole fleet; returns decisions sorted by risk (highest first)."""
    now = now or datetime.now()
    factors = factors or {}
    decisions = [
        decide(serial, score, now=now, top_factors=factors.get(serial))
        for serial, score in risk_scores.items()
    ]
    return sorted(decisions, key=lambda d: d.risk_score, reverse=True)


# ----------------------------------------------------------------------
# Cost model -- for the "was this actually better than fixed scheduling?" slide
# ----------------------------------------------------------------------

def estimate_backup_load(decisions: List[Decision],
                         horizon_days: int = 7) -> Dict[str, float]:
    """
    Estimates how many backup operations the adaptive policy triggers over a
    horizon, versus a fixed daily-backup baseline. This is the number that
    justifies the whole project: adaptive scheduling should protect the
    at-risk drives *more* while doing *fewer* total operations overall.
    """
    hours = horizon_days * 24
    adaptive_ops = sum(hours / d.backup_interval_hours for d in decisions)
    fixed_daily_ops = len(decisions) * horizon_days

    at_risk = [d for d in decisions if d.risk_score >= 0.50]
    at_risk_adaptive = sum(hours / d.backup_interval_hours for d in at_risk)
    at_risk_fixed = len(at_risk) * horizon_days

    return {
        "horizon_days": horizon_days,
        "drives": len(decisions),
        "adaptive_backup_ops": round(adaptive_ops, 1),
        "fixed_daily_backup_ops": round(fixed_daily_ops, 1),
        "ops_change_pct": round(
            100 * (adaptive_ops - fixed_daily_ops) / max(fixed_daily_ops, 1), 1),
        "high_risk_drives": len(at_risk),
        "high_risk_adaptive_ops": round(at_risk_adaptive, 1),
        "high_risk_fixed_ops": round(at_risk_fixed, 1),
        "high_risk_coverage_gain_pct": round(
            100 * (at_risk_adaptive - at_risk_fixed) / max(at_risk_fixed, 1), 1
        ) if at_risk else 0.0,
    }


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Stand-in for real model output -- replace with predictions loaded
    # from baseline_risk_model.joblib once your teammate's model is ready.
    sample_risks = {
        "DRIVE0001": 0.03,
        "DRIVE0002": 0.14,
        "DRIVE0003": 0.31,
        "DRIVE0004": 0.62,
        "DRIVE0005": 0.91,
        "DRIVE0006": 0.07,
        "DRIVE0007": 0.55,
    }
    sample_factors = {
        "DRIVE0005": ["smart_197_raw_mean (pending sectors)", "smart_5_raw_slope", "disk_io_latency_ms_mean"],
        "DRIVE0004": ["smart_187_raw_mean", "smart_198_raw_slope"],
    }

    decisions = decide_batch(sample_risks, factors=sample_factors)

    print(f"{'DRIVE':<12}{'RISK':>7}  {'TIER':<10}{'ACTION':<28}{'EVERY':>8}")
    print("-" * 68)
    for d in decisions:
        print(f"{d.serial_number:<12}{d.risk_score:>7.2f}  {d.risk_tier:<10}"
              f"{d.action.value:<28}{str(d.backup_interval_hours) + 'h':>8}")

    print("\nLoad comparison vs fixed daily backups:")
    for k, v in estimate_backup_load(decisions).items():
        print(f"  {k:<32} {v}")
