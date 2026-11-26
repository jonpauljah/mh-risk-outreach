"""Score the operational population and calculate read-only candidate ranks."""

import json
import os
from datetime import UTC, datetime
from functools import partial
from uuid import uuid4

import pandas as pd

from mh_risk_outreach.config import (
    CRITICAL_INDICATORS,
    DEFAULT_DB_PATH,
    FEATURES,
    MEMBER_ID,
    OPERATIONAL_MODEL_ENV,
)
from mh_risk_outreach.config import (
    OPERATIONAL_SCOPE as OPERATIONAL_SCOPE,  # noqa: PLC0414
)
from mh_risk_outreach.datasets import split_dataset
from mh_risk_outreach.db.operational_scores import (
    AssessmentInputs,
    assessment_score_payload,
    load_operational_inputs,
    load_operational_scores,
    save_operational_scores,
)
from mh_risk_outreach.db.repository import load_from_db
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY
from mh_risk_outreach.models.artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    ScoringArtifact,
    artifact_model_version,
    load_scoring_artifact,
)
from mh_risk_outreach.recommendations import (
    DEFAULT_RECOMMENDATION_RULES,
    RecommendationRules,
    recommend_action,
)
from mh_risk_outreach.scoring import (
    SHAP_CONTRIBUTIONS,
    rank_members,
    score_members,
)


def operational_model_name() -> str:
    """Resolve one configured outreach model using the existing model registry."""
    model = os.environ.get(OPERATIONAL_MODEL_ENV, DEFAULT_MODEL)
    if model not in REGISTRY:
        choices = ", ".join(sorted(REGISTRY))
        raise ValueError(f"invalid {OPERATIONAL_MODEL_ENV}={model!r}; choose {choices}")
    return model


def _active_artifact_matches(
    model: str, expected_model: str, expected_version: str, artifacts_dir: str
) -> bool:
    return (
        model == expected_model
        and artifact_model_version(model, artifacts_dir) == expected_version
    )


def _validate_active_artifact(artifact: ScoringArtifact, artifacts_dir: str) -> None:
    model = operational_model_name()
    if not _active_artifact_matches(
        model, artifact.model_name, artifact.model_version, artifacts_dir
    ):
        raise ValueError("operational model changed during scoring; run scoring again")


def refresh_operational_worklist(
    *, db_path: str = DEFAULT_DB_PATH, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
) -> pd.DataFrame:
    """Score all imported members and atomically replace the operational worklist."""
    model = operational_model_name()
    artifact = load_scoring_artifact(model, artifacts_dir)
    patients, assessment_inputs, cohort = load_operational_inputs(db_path)
    worklist = _build_operational_worklist(patients, assessment_inputs, artifact)
    save_operational_scores(
        worklist,
        expected_inputs=assessment_inputs,
        expected_cohort_id=cohort.cohort_id if cohort else None,
        validate_model=partial(_validate_active_artifact, artifact, artifacts_dir),
        db_path=db_path,
    )
    return worklist


def finalize_operational_reassessment(
    assessment_id: int,
    *,
    recommendation_action: str | None = None,
    recommendation_reason: str | None = None,
    recommendation_rule_version: str | None = None,
    recommendation_rules: RecommendationRules = DEFAULT_RECOMMENDATION_RULES,
    db_path: str = DEFAULT_DB_PATH,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> None:
    """Finalize the score and recommendation and publish the updated cohort.

    Supplied recommendation metadata must agree with the generated fields.
    """
    patients, assessment_inputs, cohort = load_operational_inputs(db_path)
    pending = next(
        (
            assessment
            for assessment in assessment_inputs
            if assessment.current_id == assessment_id
            and assessment.status == "PendingScoring"
        ),
        None,
    )
    if pending is None:
        raise ValueError("assessment must be the current pending reassessment")
    if cohort is None:
        raise ValueError(
            "no operational worklist; run scoring with --operational first"
        )
    model = operational_model_name()
    if not _active_artifact_matches(
        model, cohort.model_name, cohort.model_version, artifacts_dir
    ):
        raise ValueError(
            "operational model changed; refresh scoring with --operational"
        )
    artifact = load_scoring_artifact(model, artifacts_dir, version=cohort.model_version)
    worklist = _build_operational_worklist(
        patients, assessment_inputs, artifact, pending=pending
    )
    row = worklist.loc[worklist[MEMBER_ID].eq(pending.member_id)].iloc[0]
    contributions = worklist.pop(SHAP_CONTRIBUTIONS).at[row.name]
    result = assessment_score_payload(
        row,
        cohort_id=str(row["cohort_id"]),
        cohort_calculated_at=str(row["calculated_at"]),
        cohort_size=len(worklist),
    )
    result.update(
        recommend_action(
            json.loads(pending.current_inputs),
            contributions,
            rules=recommendation_rules,
        )
    )
    supplied = {
        "recommendation_action": recommendation_action,
        "recommendation_reason": recommendation_reason,
        "recommendation_rule_version": recommendation_rule_version,
    }
    for field, value in supplied.items():
        if value is not None and value != result[field]:
            raise ValueError(f"{field} disagrees with the generated recommendation")
    save_operational_scores(
        worklist,
        expected_inputs=assessment_inputs,
        expected_cohort_id=cohort.cohort_id,
        reassessment=(assessment_id, result),
        validate_model=partial(_validate_active_artifact, artifact, artifacts_dir),
        db_path=db_path,
    )


def _build_operational_worklist(
    patients: pd.DataFrame,
    assessment_inputs: list[AssessmentInputs],
    artifact: ScoringArtifact,
    *,
    pending: AssessmentInputs | None = None,
) -> pd.DataFrame:
    if patients.empty:
        raise ValueError(
            "import the patient dataset before running operational scoring"
        )
    background = split_dataset(patients).train
    members = patients.set_index(MEMBER_ID, drop=False)
    provenance = {}
    for assessment in assessment_inputs:
        member_id = assessment.member_id
        is_finalizing = (
            pending is not None and assessment.current_id == pending.current_id
        )
        scored_inputs = (
            assessment.current_inputs if is_finalizing else assessment.scored_inputs
        )
        if scored_inputs is not None:
            stored_inputs = json.loads(scored_inputs)
            members.loc[member_id, FEATURES] = [
                stored_inputs[name] for name in FEATURES
            ]
        # The flags can change while updated model inputs await explicit scoring.
        latest_inputs = json.loads(assessment.current_inputs)
        members.loc[member_id, list(CRITICAL_INDICATORS)] = [
            latest_inputs[name] for name in CRITICAL_INDICATORS
        ]
        provenance[member_id] = (
            assessment.current_id if is_finalizing else assessment.scored_id,
            assessment.current_id,
            assessment.status == "PendingScoring" and not is_finalizing,
        )

    worklist = score_members(
        members.reset_index(drop=True),
        artifact,
        background=background,
        contributions_for_member=pending.member_id if pending is not None else None,
    )
    source_columns = [
        "score_assessment_id",
        "safety_assessment_id",
        "inputs_pending_scoring",
    ]
    worklist[source_columns] = pd.DataFrame(
        [
            provenance.get(member_id, (None, None, False))
            for member_id in worklist[MEMBER_ID]
        ],
        columns=source_columns,
    ).astype({"score_assessment_id": "Int64", "safety_assessment_id": "Int64"})
    worklist["operational_model_name"] = artifact.model_name
    worklist["cohort_id"] = uuid4().hex
    worklist["calculated_at"] = datetime.now(UTC).isoformat()

    return worklist


def score_operational_member(
    member_id: int,
    inputs: dict[str, object],
    *,
    recommendation_rules: RecommendationRules = DEFAULT_RECOMMENDATION_RULES,
    db_path: str = DEFAULT_DB_PATH,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, object]:
    """Score and recommend an action for one complete survey input snapshot.

    The payload fits #32's scoring API. It records rank against the worklist
    read for this calculation, with the candidate substituted for its old row.
    """
    model = operational_model_name()
    if set(inputs) != set(FEATURES):
        raise ValueError("candidate inputs must contain all survey fields")
    cohort = load_operational_scores(db_path)
    version = str(cohort["model_version"].iat[0])
    if not _active_artifact_matches(
        model, cohort["operational_model_name"].iat[0], version, artifacts_dir
    ):
        raise ValueError(
            "operational model changed; refresh scoring with --operational"
        )
    if not cohort[MEMBER_ID].eq(member_id).any():
        raise ValueError(f"member {member_id} is outside the operational worklist")

    artifact = load_scoring_artifact(model, artifacts_dir, version=version)
    background = split_dataset(load_from_db(db_path).sort_values(MEMBER_ID)).train
    candidate = score_members(
        pd.DataFrame([{MEMBER_ID: member_id, **inputs}]),
        artifact,
        background=background,
        contributions_for_member=member_id,
    )
    contributions = candidate.pop(SHAP_CONTRIBUTIONS).iat[0]
    peers = cohort.loc[~cohort[MEMBER_ID].eq(member_id), list(candidate.columns)]
    ranked = rank_members(pd.concat([peers, candidate], ignore_index=True))
    row = ranked.loc[ranked[MEMBER_ID].eq(member_id)].iloc[0]
    row["operational_model_name"] = model
    result = assessment_score_payload(
        row,
        cohort_id=str(cohort["cohort_id"].iat[0]),
        cohort_calculated_at=str(cohort["calculated_at"].iat[0]),
        cohort_size=len(cohort),
        calculated_at=datetime.now(UTC).isoformat(),
    )
    result.update(recommend_action(inputs, contributions, rules=recommendation_rules))
    return result
