"""Choose a versioned outreach action from reported survey values."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from numbers import Integral, Real
from operator import ge, le
from typing import NamedTuple, TypedDict

from mh_risk_outreach.config import CRITICAL_INDICATORS
from mh_risk_outreach.prepare import SCALED_RANGES

SAFETY_REVIEW_ACTION = "Human-led safety review"

_BEHAVIORAL = "Behavioral-health resource review"
_SOCIAL = "Social-support referral"
_FINANCIAL = "Financial/social-services referral"
_WORKPLACE = "Workplace-support referral"
_LIFESTYLE = "Lifestyle/wellness support"
_ROUTINE_ACTIONS = (_BEHAVIORAL, _SOCIAL, _FINANCIAL, _WORKPLACE, _LIFESTYLE)
_BUILT_IN_RULE_VERSION = "outreach-rules-v1"


class _Trigger(NamedTuple):
    feature: str
    reason: str
    cutoff: float
    qualifies: Callable[[float, float], bool]


class Recommendation(TypedDict):
    recommendation_action: str
    recommendation_reason: str
    recommendation_rule_version: str


@dataclass(frozen=True)
class RecommendationRules:
    """Thresholds and tie order for the routine actions."""

    version: str = _BUILT_IN_RULE_VERSION
    feeling_sad_down_min: float = 8
    anxious_nervous_min: float = 8
    social_support_max: float = 3
    financial_stress_min: float = 8
    work_stress_level_min: float = 8
    sleep_hours_night_max: float = 6
    precedence: tuple[str, ...] = _ROUTINE_ACTIONS

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("recommendation rules require a nonempty version")
        if (
            not isinstance(self.precedence, tuple)
            or len(self.precedence) != len(_ROUTINE_ACTIONS)
            or any(self.precedence.count(action) != 1 for action in _ROUTINE_ACTIONS)
        ):
            raise ValueError("precedence must list all five routine actions once")
        for triggers in self._triggers().values():
            for trigger in triggers:
                low, high = SCALED_RANGES[trigger.feature]
                value = _finite_number(trigger.cutoff)
                if (
                    value is None
                    or (low is not None and value < low)
                    or (high is not None and value > high)
                ):
                    raise ValueError(
                        f"{trigger.feature} threshold must be numeric and within {low}–{high}"
                    )
        if self.version == _BUILT_IN_RULE_VERSION and any(
            getattr(self, field.name) != field.default for field in fields(self)
        ):
            raise ValueError(
                f"{_BUILT_IN_RULE_VERSION} is reserved for the default thresholds "
                "and precedence. Use a new version to change them."
            )

    def _triggers(self) -> dict[str, tuple[_Trigger, ...]]:
        return {
            _BEHAVIORAL: (
                _Trigger(
                    "Feeling_Sad_Down",
                    "Reported feeling sad or down was {}.",
                    self.feeling_sad_down_min,
                    ge,
                ),
                _Trigger(
                    "Anxious_Nervous",
                    "Reported feeling anxious or nervous was {}.",
                    self.anxious_nervous_min,
                    ge,
                ),
            ),
            _SOCIAL: (
                _Trigger(
                    "Social_Support",
                    "Reported social support was {}.",
                    self.social_support_max,
                    le,
                ),
            ),
            _FINANCIAL: (
                _Trigger(
                    "Financial_Stress",
                    "Reported financial stress was {}.",
                    self.financial_stress_min,
                    ge,
                ),
            ),
            _WORKPLACE: (
                _Trigger(
                    "Work_Stress_Level",
                    "Reported work stress was {}.",
                    self.work_stress_level_min,
                    ge,
                ),
            ),
            _LIFESTYLE: (
                _Trigger(
                    "Sleep_Hours_Night",
                    "Reported {} hours of sleep per night.",
                    self.sleep_hours_night_max,
                    le,
                ),
            ),
        }


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


DEFAULT_RECOMMENDATION_RULES = RecommendationRules()


def recommend_action(
    inputs: Mapping[str, object],
    contributions: Mapping[str, float] | None = None,
    *,
    rules: RecommendationRules = DEFAULT_RECOMMENDATION_RULES,
) -> Recommendation:
    """Return one recommendation, with required safety flags checked first."""
    reported_flags = []
    for field in CRITICAL_INDICATORS:
        flag = inputs.get(field)
        if (
            isinstance(flag, bool)
            or not isinstance(flag, Integral)
            or flag not in (0, 1)
        ):
            raise ValueError(f"{field} must be integer 0 or 1")
        if flag == 1:
            label = field.replace("_", " ").lower().replace("self harm", "self-harm")
            reported_flags.append(f"{label} (1)")
    if reported_flags:
        return {
            "recommendation_action": SAFETY_REVIEW_ACTION,
            "recommendation_reason": f"Reported {' and '.join(reported_flags)}.",
            "recommendation_rule_version": rules.version,
        }

    action = "General wellness check-in"
    reason = "No configured outreach trigger was met."
    strongest = -1.0
    evidence = contributions if contributions is not None else {}
    triggers = rules._triggers()
    for candidate in rules.precedence:
        for trigger in triggers[candidate]:
            reported = inputs.get(trigger.feature)
            value = _finite_number(reported)
            if value is None or not trigger.qualifies(value, trigger.cutoff):
                continue
            support = max(0.0, _finite_number(evidence.get(trigger.feature)) or 0.0)
            if support > strongest:
                strongest = support
                action = candidate
                reason = trigger.reason.format(str(reported).removesuffix(".0"))

    return {
        "recommendation_action": action,
        "recommendation_reason": reason,
        "recommendation_rule_version": rules.version,
    }
