"""Build manager worklists and enforce outreach eligibility when selected."""

from contextlib import closing
from dataclasses import dataclass
from sqlite3 import Connection

from mh_risk_outreach.config import CRITICAL_INDICATORS, DEFAULT_DB_PATH, FEATURES
from mh_risk_outreach.db.assessments import _connect
from mh_risk_outreach.db.outreach import _create_case
from mh_risk_outreach.models.artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    artifact_model_version,
)
from mh_risk_outreach.operational import (
    operational_model_name,
    score_operational_member,
)
from mh_risk_outreach.recommendations import SAFETY_REVIEW_ACTION


@dataclass(frozen=True)
class OutreachCandidate:
    """A member's current worklist entry and the provenance of a selection."""

    member_id: int
    assessment_id: int | None
    cohort_id: str | None
    band: str | None
    rank: int | None
    safety_flags: tuple[str, ...]
    pending_scoring: bool
    selectable: bool


@dataclass(frozen=True)
class OutreachWorklists:
    active_cases: int
    standard_slots: int
    safety_review: tuple[OutreachCandidate, ...]
    standard_outreach: tuple[OutreachCandidate, ...]


# Use live input flags instead of cached flags on the published score. A manual
# submission can report a new concern before any rescore has run.
_LATEST_FLAGS = ", ".join(
    f"COALESCE(json_extract(a.inputs_json, '$.{flag}'), p.{flag}) AS {flag}"
    for flag in CRITICAL_INDICATORS
)
_CANDIDATES = f"""
    SELECT p.Member_ID, state.assessment_id, a.source, a.scoring_status,
           a.recommendation_action,
           score.cohort_id, score.band_level_priority, score.rank_priority,
           {_LATEST_FLAGS},
           EXISTS (
               SELECT 1 FROM outreach_cases AS active WHERE active.Member_ID = p.Member_ID
               AND active.status IN ('Pending', 'Accepted')
           ) AS has_active_case,
           EXISTS (
               SELECT 1 FROM outreach_cases AS used WHERE used.assessment_id = state.assessment_id
           ) AS used_assessment,
           (
               SELECT MAX(assessment_id) FROM outreach_cases AS completed
               WHERE completed.Member_ID = p.Member_ID AND completed.status = 'Completed'
           ) AS last_completed_assessment
    FROM patients AS p
    LEFT JOIN member_assessment_state AS state ON state.Member_ID = p.Member_ID
    LEFT JOIN risk_assessments AS a ON a.assessment_id = state.assessment_id
    LEFT JOIN operational_scores AS score ON score.Member_ID = p.Member_ID
         AND score.operational_model_name = ? AND score.model_version = ?
         AND score.score_assessment_id IS state.assessment_id
         AND score.safety_assessment_id IS state.assessment_id
         AND score.inputs_pending_scoring = 0
         AND (state.assessment_id IS NULL OR (
             a.scoring_status = 'Scored'
             AND a.operational_model_name = score.operational_model_name
             AND a.model_version = score.model_version
         ))
    ORDER BY score.rank_priority NULLS LAST, p.Member_ID
"""


@dataclass(frozen=True)
class OutreachPolicy:
    """Use the same configured capacity for listing and creating outreach cases."""

    capacity: int
    db_path: str = DEFAULT_DB_PATH
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity < 0:
            raise ValueError("outreach capacity must be a nonnegative integer")

    def worklists(self) -> OutreachWorklists:
        """Read both queues and the shared capacity from one SQLite snapshot."""
        with closing(_connect(self.db_path)) as conn, conn:
            conn.execute("BEGIN")
            return self._worklists(conn)

    def _worklists(self, conn: Connection) -> OutreachWorklists:
        active = conn.execute(
            "SELECT COUNT(*) FROM outreach_cases WHERE status IN ('Pending', 'Accepted')"
        ).fetchone()[0]
        slots = max(0, self.capacity - active)
        model = operational_model_name()
        version = artifact_model_version(model, self.artifacts_dir)
        safety: list[OutreachCandidate] = []
        standard: list[OutreachCandidate] = []
        for row in conn.execute(_CANDIDATES, (model, version)):
            if row["used_assessment"]:
                continue
            flags = tuple(flag for flag in CRITICAL_INDICATORS if row[flag] == 1)
            if not flags and (
                len(standard) >= slots
                or row["band_level_priority"] not in ("High", "Very High")
            ):
                continue

            assessment_id = row["assessment_id"]
            completed_id = row["last_completed_assessment"]
            action = row["recommendation_action"]
            selectable = (
                not row["has_active_case"]
                and row["cohort_id"] is not None
                and (
                    assessment_id is None
                    or (
                        bool(action) and (action == SAFETY_REVIEW_ACTION) == bool(flags)
                    )
                )
            )
            if not flags and (
                not selectable
                or (
                    completed_id is not None
                    and (
                        row["source"] != "reassessment" or assessment_id <= completed_id
                    )
                )
            ):
                continue

            candidate = OutreachCandidate(
                member_id=row["Member_ID"],
                assessment_id=assessment_id,
                cohort_id=row["cohort_id"],
                band=row["band_level_priority"],
                rank=row["rank_priority"],
                safety_flags=flags,
                pending_scoring=row["scoring_status"] == "PendingScoring",
                selectable=selectable,
            )
            (safety if flags else standard).append(candidate)
        return OutreachWorklists(active, slots, tuple(safety), tuple(standard))

    def create_case(self, selection: OutreachCandidate) -> int:
        """Create a selected case after rechecking the policy under a write lock."""
        baseline = None
        if selection.assessment_id is None:
            # Calculate SHAP evidence outside the write transaction. Use only
            # persisted survey inputs and verify the cohort again before writing.
            with closing(_connect(self.db_path)) as conn, conn:
                conn.execute("BEGIN")
                self._require_selection(conn, selection)
                member = conn.execute(
                    "SELECT * FROM patients WHERE Member_ID=?", (selection.member_id,)
                ).fetchone()
                inputs = {name: member[name] for name in FEATURES}
            baseline = score_operational_member(
                selection.member_id,
                inputs,
                db_path=self.db_path,
                artifacts_dir=self.artifacts_dir,
            )
            context = baseline["rank_context"]
            if (
                not isinstance(context, dict)
                or context.get("cohort_id") != selection.cohort_id
            ):
                raise ValueError(
                    "operational scoring changed; refresh the outreach worklists"
                )

        with closing(_connect(self.db_path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_selection(conn, selection)
            if baseline is not None:
                return _create_case(
                    conn,
                    selection.member_id,
                    str(baseline["recommendation_action"]),
                    recommendation_reason=str(baseline["recommendation_reason"]),
                    recommendation_rule_version=str(
                        baseline["recommendation_rule_version"]
                    ),
                )
            action = conn.execute(
                "SELECT recommendation_action FROM risk_assessments WHERE assessment_id=?",
                (selection.assessment_id,),
            ).fetchone()[0]
            return _create_case(conn, selection.member_id, action)

    def _require_selection(
        self, conn: Connection, selection: OutreachCandidate
    ) -> None:
        worklists = self._worklists(conn)
        if selection.safety_flags:
            candidates = worklists.safety_review
        else:
            if worklists.standard_slots == 0:
                raise ValueError("standard outreach is at capacity")
            candidates = worklists.standard_outreach
        if not selection.selectable or selection not in candidates:
            raise ValueError(
                "selection is no longer eligible; refresh the outreach worklists"
            )
