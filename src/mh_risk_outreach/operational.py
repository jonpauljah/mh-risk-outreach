"""Score the operational population and calculate read-only candidate ranks."""

import json
import os
from contextlib import closing
from datetime import UTC, datetime
from functools import partial
from sqlite3 import Connection
from threading import Lock
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
from mh_risk_outreach.datasets import DatasetSplit
from mh_risk_outreach.db.assessments import _connect
from mh_risk_outreach.db.operational_scores import (
    NO_WORKLIST,
    AssessmentInputs,
    _published_cohort,
    _save_baseline_recommendations,
    assessment_score_payload,
    load_candidate_scoring_inputs,
    load_operational_inputs,
    save_operational_scores,
    worklist_requires_refresh,
)
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY
from mh_risk_outreach.models.artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    ScoringArtifact,
    artifact_model_version,
    load_scoring_artifact,
)
from mh_risk_outreach.reasons import build_reasons_and_contributions
from mh_risk_outreach.recommendations import (
    DEFAULT_RECOMMENDATION_RULES,
    Recommendation,
    RecommendationRules,
    recommend_action,
)
from mh_risk_outreach.scoring import (
    REASON_CHUNK_SIZE,
    SHAP_CONTRIBUTIONS,
    rank_members,
    score_members,
)

_BASELINE_FILL_LOCK = Lock()


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
    """Score the held-out members and atomically replace the operational worklist."""
    model = operational_model_name()
    artifact = load_scoring_artifact(model, artifacts_dir)
    split, assessment_inputs, cohort = load_operational_inputs(db_path)
    worklist, recommendations, _ = _build_operational_worklist(
        split, assessment_inputs, artifact
    )
    save_operational_scores(
        worklist,
        expected_inputs=assessment_inputs,
        expected_cohort_id=cohort.cohort_id if cohort else None,
        baseline_recommendations=recommendations,
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
    split, assessment_inputs, cohort = load_operational_inputs(
        db_path, require_current_population=True
    )
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
        raise ValueError(NO_WORKLIST)
    model = operational_model_name()
    if not _active_artifact_matches(
        model, cohort.model_name, cohort.model_version, artifacts_dir
    ):
        raise ValueError(
            "operational model changed; refresh scoring with --operational"
        )
    artifact = load_scoring_artifact(model, artifacts_dir, version=cohort.model_version)
    worklist, recommendations, contributions = _build_operational_worklist(
        split, assessment_inputs, artifact, pending=pending
    )
    row = worklist.loc[worklist[MEMBER_ID].eq(pending.member_id)].iloc[0]
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
        baseline_recommendations=recommendations,
        reassessment=(assessment_id, result),
        validate_model=partial(_validate_active_artifact, artifact, artifacts_dir),
        db_path=db_path,
    )


def _build_operational_worklist(
    split: DatasetSplit,
    assessment_inputs: list[AssessmentInputs],
    artifact: ScoringArtifact,
    *,
    pending: AssessmentInputs | None = None,
) -> tuple[pd.DataFrame, dict[int, Recommendation], dict[str, float] | None]:
    if split.evaluation.empty:
        raise ValueError(
            "import the patient dataset before running operational scoring"
        )
    members = split.evaluation.sort_values(MEMBER_ID).set_index(MEMBER_ID, drop=False)
    provenance = {}
    for assessment in assessment_inputs:
        member_id = assessment.member_id
        if member_id not in members.index:
            continue
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

    baseline_ids = set(members.index) - provenance.keys()
    worklist = score_members(
        members.reset_index(drop=True),
        artifact,
        background=split.train,
        contributions_for_member=pending.member_id if pending is not None else None,
        contributions_for_members=baseline_ids,
    )
    contributions = (
        dict(zip(worklist[MEMBER_ID], worklist.pop(SHAP_CONTRIBUTIONS), strict=True))
        if SHAP_CONTRIBUTIONS in worklist
        else {}
    )
    recommendations = _recommend_baselines(
        members.loc[members.index.isin(baseline_ids)], contributions
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

    return (
        worklist,
        recommendations,
        contributions.get(pending.member_id) if pending else None,
    )


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
    cohort, background = load_candidate_scoring_inputs(member_id, db_path)
    version = str(cohort["model_version"].iat[0])
    if not _active_artifact_matches(
        model, cohort["operational_model_name"].iat[0], version, artifacts_dir
    ):
        raise ValueError(
            "operational model changed; refresh scoring with --operational"
        )
    artifact = load_scoring_artifact(model, artifacts_dir, version=version)
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


def recommend_operational_baselines(
    member_ids: tuple[int, ...],
    cohort_id: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> dict[int, Recommendation]:
    """Fill missing recommendations for worklists published by older versions."""
    with _BASELINE_FILL_LOCK:
        return _fill_baseline_recommendations(
            member_ids, cohort_id, db_path, artifacts_dir
        )


def _fill_baseline_recommendations(
    member_ids: tuple[int, ...], cohort_id: str, db_path: str, artifacts_dir: str
) -> dict[int, Recommendation]:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("BEGIN")
        cohort = _published_cohort(conn)
        if cohort is None or cohort.cohort_id != cohort_id:
            raise ValueError(
                "Operational scoring changed. Refresh the outreach worklists."
            )
        existing = _saved_baseline_recommendations(conn, member_ids, cohort_id)

    model = operational_model_name()
    if not _active_artifact_matches(
        model, cohort.model_name, cohort.model_version, artifacts_dir
    ):
        raise ValueError("Operational model changed. Refresh the outreach worklists.")
    missing = set(member_ids) - existing.keys()
    if not missing:
        return existing

    split, _, current_cohort = load_operational_inputs(
        db_path, require_current_population=True
    )
    if current_cohort != cohort:
        raise ValueError("Operational scoring changed. Refresh the outreach worklists.")
    members = split.evaluation.loc[split.evaluation[MEMBER_ID].isin(missing)]
    if set(members[MEMBER_ID]) != missing:
        raise ValueError("A selected member is outside the operational population.")
    artifact = load_scoring_artifact(model, artifacts_dir, version=cohort.model_version)
    calculated = _compute_baseline_recommendations(members, split.train, artifact)
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        if _published_cohort(conn) != cohort or worklist_requires_refresh(
            conn, set(split.evaluation[MEMBER_ID])
        ):
            raise ValueError(
                "Operational scoring changed. Refresh the outreach worklists."
            )
        _validate_active_artifact(artifact, artifacts_dir)
        _save_baseline_recommendations(conn, cohort_id, calculated)
        return _saved_baseline_recommendations(conn, member_ids, cohort_id)


def _saved_baseline_recommendations(
    conn: Connection, member_ids: tuple[int, ...], cohort_id: str
) -> dict[int, Recommendation]:
    return {
        row["Member_ID"]: Recommendation(
            recommendation_action=row["recommendation_action"],
            recommendation_reason=row["recommendation_reason"],
            recommendation_rule_version=row["recommendation_rule_version"],
        )
        for row in conn.execute(
            """SELECT * FROM operational_recommendations
               WHERE cohort_id = ? AND Member_ID IN (SELECT value FROM json_each(?))""",
            (cohort_id, json.dumps(member_ids)),
        )
    }


def _compute_baseline_recommendations(
    members: pd.DataFrame, background: pd.DataFrame, artifact: ScoringArtifact
) -> dict[int, Recommendation]:
    _, contributions = build_reasons_and_contributions(
        artifact.pipeline,
        members,
        background,
        contribution_rows=range(len(members)),
        chunk_size=REASON_CHUNK_SIZE,
    )
    return _recommend_baselines(
        members,
        {member_id: contributions[i] for i, member_id in enumerate(members[MEMBER_ID])},
    )


def _recommend_baselines(
    members: pd.DataFrame, contributions: dict[int, dict[str, float]]
) -> dict[int, Recommendation]:
    recommendations = {}
    for member in members.to_dict("records"):
        member_id = member[MEMBER_ID]
        try:
            recommendations[member_id] = recommend_action(
                member, contributions[member_id]
            )
        except ValueError:
            # A malformed row can be left unavailable while other rows are usable.
            continue
    return recommendations
