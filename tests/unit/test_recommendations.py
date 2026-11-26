from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mh_risk_outreach.recommendations import (
    DEFAULT_RECOMMENDATION_RULES,
    RecommendationRules,
    recommend_action,
)

NO_FLAGS = {"Self_Harm_Thoughts": 0, "Suicidal_Thoughts": 0}
GENERAL = "General wellness check-in"
BEHAVIORAL = "Behavioral-health resource review"
SOCIAL = "Social-support referral"
FINANCIAL = "Financial/social-services referral"
WORKPLACE = "Workplace-support referral"
LIFESTYLE = "Lifestyle/wellness support"


@pytest.mark.parametrize(
    "flags,reason",
    [
        ((1, 0), "Reported self-harm thoughts (1)."),
        ((0, 1), "Reported suicidal thoughts (1)."),
        ((1, 1), "Reported self-harm thoughts (1) and suicidal thoughts (1)."),
    ],
)
def test_safety_review_overrides_routine_triggers(flags, reason):
    inputs = {
        "Self_Harm_Thoughts": np.int64(flags[0]),
        "Suicidal_Thoughts": flags[1],
        "Feeling_Sad_Down": 10,
    }
    expected = {
        "recommendation_action": "Human-led safety review",
        "recommendation_reason": reason,
        "recommendation_rule_version": "outreach-rules-v1",
    }
    assert recommend_action(inputs) == expected
    assert recommend_action(inputs, {"Feeling_Sad_Down": 10.0}) == expected


@pytest.mark.parametrize("bad_flag", [None, np.nan, True, np.bool_(False), 1.0, "1", 2])
def test_both_safety_flags_must_follow_the_required_contract(bad_flag):
    inputs = {"Self_Harm_Thoughts": 1, "Suicidal_Thoughts": bad_flag}
    with pytest.raises(ValueError, match="Suicidal_Thoughts must be integer 0 or 1"):
        recommend_action(inputs)
    del inputs["Suicidal_Thoughts"]
    with pytest.raises(ValueError, match="Suicidal_Thoughts must be integer 0 or 1"):
        recommend_action(inputs)


@pytest.mark.parametrize(
    "feature,cutoff,nonqualifying,action,reason",
    [
        ("Feeling_Sad_Down", 8, 7, BEHAVIORAL, "feeling sad or down was 8"),
        ("Anxious_Nervous", 8, 7, BEHAVIORAL, "feeling anxious or nervous was 8"),
        ("Social_Support", 3, 4, SOCIAL, "social support was 3"),
        ("Financial_Stress", 8, 7, FINANCIAL, "financial stress was 8"),
        ("Work_Stress_Level", 8, 7, WORKPLACE, "work stress was 8"),
        ("Sleep_Hours_Night", 6, 6.01, LIFESTYLE, "6 hours of sleep per night"),
    ],
)
def test_routine_cutoffs_include_equality(
    feature, cutoff, nonqualifying, action, reason
):
    result = recommend_action({**NO_FLAGS, feature: np.float32(cutoff)})
    assert result == {
        "recommendation_action": action,
        "recommendation_reason": f"Reported {reason}.",
        "recommendation_rule_version": "outreach-rules-v1",
    }
    assert (
        recommend_action({**NO_FLAGS, feature: nonqualifying})["recommendation_action"]
        == GENERAL
    )


def test_fractional_sleep_is_reported_in_hours():
    result = recommend_action({**NO_FLAGS, "Sleep_Hours_Night": np.float32(5.9)})
    assert result["recommendation_action"] == LIFESTYLE
    assert result["recommendation_reason"] == "Reported 5.9 hours of sleep per night."


def test_missing_measurements_and_other_evidence_cannot_trigger_an_action():
    result = recommend_action(
        {
            **NO_FLAGS,
            "risk_score": 100,
            "Age": 72,
            "Family_History_Mental_Illness": 1,
            "Work_Stress_Level": 7,
            "Feeling_Sad_Down": np.nan,
            "Social_Support": pd.NA,
            "Financial_Stress": None,
        },
        {
            "Age": 20.0,
            "Family_History_Mental_Illness": 20.0,
            "Work_Stress_Level": 20.0,
            "Feeling_Sad_Down": 20.0,
        },
    )
    assert result["recommendation_action"] == GENERAL
    assert result["recommendation_reason"] == "No configured outreach trigger was met."


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"Financial_Stress": -0.5, "Anxious_Nervous": 0.0, "Social_Support": -2.5},
        {"Feeling_Sad_Down": 0.5, "Anxious_Nervous": 0.5, "Social_Support": 0.5},
    ],
)
def test_ties_use_category_precedence_and_feature_order(evidence):
    inputs = {
        **NO_FLAGS,
        "Financial_Stress": 9,
        "Anxious_Nervous": 10,
        "Social_Support": 3,
        "Feeling_Sad_Down": 8,
    }
    result = recommend_action(inputs, evidence)
    assert result["recommendation_action"] == BEHAVIORAL
    assert result["recommendation_reason"] == "Reported feeling sad or down was 8."
    assert result == recommend_action(dict(reversed(list(inputs.items()))), evidence)


def test_largest_positive_contribution_among_qualifying_features_wins():
    inputs = {
        **NO_FLAGS,
        "Feeling_Sad_Down": 7,
        "Anxious_Nervous": 9,
        "Financial_Stress": 10,
        "Social_Support": 2,
    }
    evidence = {
        "Feeling_Sad_Down": 1.0,
        "Anxious_Nervous": 0.3,
        "Social_Support": 0.5,
        "Financial_Stress": float("nan"),
    }
    result = recommend_action(inputs, evidence)
    assert result["recommendation_action"] == SOCIAL
    assert result["recommendation_reason"] == "Reported social support was 2."

    inputs["Feeling_Sad_Down"] = 8
    evidence["Feeling_Sad_Down"] = 0.1
    evidence["Anxious_Nervous"] = 0.6
    result = recommend_action(inputs, evidence)
    assert result["recommendation_action"] == BEHAVIORAL
    assert (
        result["recommendation_reason"] == "Reported feeling anxious or nervous was 9."
    )


def test_thresholds_precedence_and_version_can_be_configured():
    inputs = {**NO_FLAGS, "Social_Support": 3, "Feeling_Sad_Down": 8}
    assert recommend_action(inputs)["recommendation_action"] == BEHAVIORAL

    thresholds = replace(
        DEFAULT_RECOMMENDATION_RULES,
        version="higher-sadness-v2",
        feeling_sad_down_min=9,
    )
    result = recommend_action(inputs, rules=thresholds)
    assert result["recommendation_action"] == SOCIAL
    assert result["recommendation_rule_version"] == "higher-sadness-v2"

    precedence = replace(
        DEFAULT_RECOMMENDATION_RULES,
        version="social-first-v2",
        precedence=(SOCIAL, BEHAVIORAL, FINANCIAL, WORKPLACE, LIFESTYLE),
    )
    result = recommend_action(inputs, rules=precedence)
    assert result["recommendation_action"] == SOCIAL
    assert result["recommendation_rule_version"] == "social-first-v2"


@pytest.mark.parametrize(
    "changes",
    [
        {"social_support_max": 4},
        {"precedence": (SOCIAL, BEHAVIORAL, FINANCIAL, WORKPLACE, LIFESTYLE)},
    ],
)
def test_changed_policy_requires_a_new_version(changes):
    with pytest.raises(ValueError, match="outreach-rules-v1 is reserved"):
        RecommendationRules(**changes)
    with pytest.raises(ValueError, match="outreach-rules-v1 is reserved"):
        replace(DEFAULT_RECOMMENDATION_RULES, **changes)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"version": " "}, "version"),
        ({"feeling_sad_down_min": -1}, "Feeling_Sad_Down"),
        ({"anxious_nervous_min": float("nan")}, "Anxious_Nervous"),
        ({"financial_stress_min": float("inf")}, "Financial_Stress"),
        ({"work_stress_level_min": "8"}, "Work_Stress_Level"),
        ({"social_support_max": True}, "Social_Support"),
        ({"sleep_hours_night_max": 25}, "Sleep_Hours_Night"),
        ({"precedence": (SOCIAL,)}, "precedence"),
        (
            {"precedence": (BEHAVIORAL, BEHAVIORAL, FINANCIAL, WORKPLACE, LIFESTYLE)},
            "precedence",
        ),
    ],
)
def test_invalid_rule_configuration_is_rejected(changes, error):
    with pytest.raises(ValueError, match=error):
        replace(DEFAULT_RECOMMENDATION_RULES, **changes)
