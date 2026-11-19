"""Score the operational population and calculate read-only candidate ranks."""

import json
import os
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from mh_risk_outreach.config import (
    DEFAULT_DB_PATH,
    FEATURES,
    MEMBER_ID,
    OPERATIONAL_MODEL_ENV,
)
from mh_risk_outreach.datasets import split_dataset
from mh_risk_outreach.db.operational_scores import (
    load_operational_inputs,
    load_operational_scores,
    save_operational_scores,
)
from mh_risk_outreach.db.repository import load_from_db
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY
from mh_risk_outreach.models.artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    artifact_model_version,
    load_scoring_artifact,
)
from mh_risk_outreach.scoring import CRITICAL_INDICATORS, rank_members, score_members

OPERATIONAL_SCOPE = "all_imported_members"
RANK_ORDER = ("safety_review_required desc", "risk_score desc", "Member_ID asc")


def operational_model_name() -> str:
    """Resolve one configured outreach model using the existing model registry."""
    model = os.environ.get(OPERATIONAL_MODEL_ENV, DEFAULT_MODEL)
    if model not in REGISTRY:
        choices = ", ".join(sorted(REGISTRY))
        raise ValueError(f"invalid {OPERATIONAL_MODEL_ENV}={model!r}; choose {choices}")
    return model


def refresh_operational_worklist(
    *, db_path: str = DEFAULT_DB_PATH, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
) -> pd.DataFrame:
    """Score all imported members and atomically replace the operational worklist."""
    model = operational_model_name()
    artifact = load_scoring_artifact(model, artifacts_dir)
    patients, assessment_inputs = load_operational_inputs(db_path)
    if patients.empty:
        raise ValueError(
            "import the patient dataset before running operational scoring"
        )
    background = split_dataset(patients).train
    members = patients.set_index(MEMBER_ID, drop=False)
    provenance = {}
    for assessment in assessment_inputs:
        member_id = assessment.member_id
        if assessment.scored_inputs is not None:
            stored_inputs = json.loads(assessment.scored_inputs)
            members.loc[member_id, FEATURES] = [
                stored_inputs[name] for name in FEATURES
            ]
        # The flags can change while updated model inputs await explicit scoring.
        latest_inputs = json.loads(assessment.current_inputs)
        members.loc[member_id, list(CRITICAL_INDICATORS)] = [
            latest_inputs[name] for name in CRITICAL_INDICATORS
        ]
        provenance[member_id] = (
            assessment.scored_id,
            assessment.current_id,
            assessment.status == "PendingScoring",
        )

    worklist = score_members(
        members.reset_index(drop=True),
        artifact.pipeline,
        model_version=artifact.model_version,
        cut_points=artifact.cut_points,
        background=background,
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
    worklist["operational_model_name"] = model
    worklist["cohort_id"] = uuid4().hex
    worklist["calculated_at"] = datetime.now(UTC).isoformat()

    if artifact_model_version(model, artifacts_dir) != artifact.model_version:
        raise ValueError("operational model changed during scoring; run scoring again")
    save_operational_scores(
        worklist, expected_inputs=assessment_inputs, db_path=db_path
    )
    return worklist


def score_operational_member(
    member_id: int,
    inputs: dict[str, object],
    *,
    db_path: str = DEFAULT_DB_PATH,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, object]:
    """Score one complete survey input snapshot against the published worklist.

    The payload fits #32's scoring API. It records rank against the worklist
    read for this calculation, with the candidate substituted for its old row.
    """
    model = operational_model_name()
    if set(inputs) != set(FEATURES):
        raise ValueError("candidate inputs must contain all survey fields")
    cohort = load_operational_scores(db_path)
    version = str(cohort["model_version"].iat[0])
    if (
        model != cohort["operational_model_name"].iat[0]
        or artifact_model_version(model, artifacts_dir) != version
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
        artifact.pipeline,
        model_version=version,
        cut_points=artifact.cut_points,
        background=background,
    )
    peers = cohort.loc[~cohort[MEMBER_ID].eq(member_id), list(candidate.columns)]
    ranked = rank_members(pd.concat([peers, candidate], ignore_index=True))
    row = ranked.loc[ranked[MEMBER_ID].eq(member_id)].iloc[0]
    return {
        "risk_score": float(row["risk_score"]),
        "band_level_priority": str(row["band_level_priority"]),
        "rank_priority": int(row["rank_priority"]),
        "explanation": str(row["reasoning"]),
        "operational_model_name": model,
        "model_version": version,
        "rank_context": {
            "cohort_id": str(cohort["cohort_id"].iat[0]),
            "cohort_calculated_at": str(cohort["calculated_at"].iat[0]),
            "calculated_at": datetime.now(UTC).isoformat(),
            "scope": OPERATIONAL_SCOPE,
            "cohort_size": len(cohort),
            "safety_review_required": bool(row["safety_review_required"]),
            "ordering": list(RANK_ORDER),
        },
    }
