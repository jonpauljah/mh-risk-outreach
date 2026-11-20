# Milestone 3 — Project Plan and Implementation Overview

## Milestone Overview

This proof of concept uses a synthetic mental health survey to support human
outreach decisions. The existing Python application imports member records into
SQLite and provides Streamlit views for records, model evaluation, and risk
scoring. See the [dataset](../milestone-1/dataset.md), [ML specification](../ml-spec.md),
and [current risk-scoring design](../risk-scoring.md) for the baseline.

Milestone 3 aims to turn predictions into a manageable outreach workflow:
preserve assessment history, recommend actions, let a manager select and track
cases, and report outcomes. The [architecture and workflows](architecture-and-workflows.md)
describe the intended data flow. Except where marked merged below, these are
planned outcomes, not completed application behavior.

## Objectives and Scope

- Preserve the approved imported dataset and stable member identities.
- Select one operational model using the three-model evaluation; assign stable,
  model-derived risk bands and separate safety review from model predictions.
- Keep on-demand assessment history and support explicit manual reassessment.
- Generate explainable, versioned Next Best Action (NBA) recommendations.
- Prioritize Safety Review and Standard Outreach within shared capacity, with
  manager-selected case creation and an explicit safety exception.
- Track cases from Pending through Accepted to Completed; retain decisions,
  outcomes, timestamps, and historical context.
- Provide Streamlit worklists, case management, reassessment, and read-only
  reporting and member comparisons.

## Case Manager User Stories

These stories describe the planned outreach workflow and its intended value.

- As a case manager, I want to run population scoring when I choose, so that I
  can review an updated outreach worklist.
- As a case manager, I want to see safety concerns separately from model risk
  bands, including concerns in newly submitted inputs, so that I can review them
  even when standard outreach capacity is full.
- As a case manager, I want to see eligible members, available capacity, and an
  explained Next Best Action, so that I can select which outreach cases to create.
- As a case manager, I want to accept a recommendation or record a justified
  alternative, so that my decision and the original recommendation are retained.
- As a case manager, I want to record an outcome and manually complete an accepted
  case, so that active work and completed history remain accurate.
- As a case manager, I want to submit updated survey information without changing
  the imported record, so that the member has a new assessment while prior
  assessments and cases remain available. I can see when scoring is pending.
- As a case manager, I want to review a member's assessments and completed cases,
  so that I can compare available results without assuming outreach changed risk.

## Implementation Progress

Status below describes implementation progress. Dependencies are a logical
implementation guide.

| Issue | Objective | Main dependencies | Implementation status |
| --- | --- | --- | --- |
| [#31](https://github.com/jonpauljah/mh-risk-outreach/issues/31) | Protect initial import and member identity | Existing ETL | **Merged:** [PR #44](https://github.com/jonpauljah/mh-risk-outreach/pull/44) skips reimports into populated `patients`. |
| [#32](https://github.com/jonpauljah/mh-risk-outreach/issues/32) | Assessment and case persistence | Coordinate #31 | **Merged:** [PR #45](https://github.com/jonpauljah/mh-risk-outreach/pull/45) implements assessment and case storage operations. |
| [#33](https://github.com/jonpauljah/mh-risk-outreach/issues/33) | Exclude safety flags and reevaluate three models | Existing ML pipeline | **Merged:** [PR #46](https://github.com/jonpauljah/mh-risk-outreach/pull/46) removes safety flags from model predictors and reevaluates all three models. |
| [#34](https://github.com/jonpauljah/mh-risk-outreach/issues/34) | Operational model and stable bands | #33, #32 | Operational scoring and versioned bands implemented in [PR #47](https://github.com/jonpauljah/mh-risk-outreach/pull/47). |
| [#35](https://github.com/jonpauljah/mh-risk-outreach/issues/35) | Current and historical assessments | #32, #34 | Backend integration implemented for first-case baselines and explicit reassessment finalization. |
| [#36](https://github.com/jonpauljah/mh-risk-outreach/issues/36) | Deterministic NBA rules | #33, #34 | Planned. |
| [#37](https://github.com/jonpauljah/mh-risk-outreach/issues/37) | Eligibility, safety, capacity | #35, #36 | Planned. |
| [#38](https://github.com/jonpauljah/mh-risk-outreach/issues/38) | Case lifecycle | #32, #37; #35, #36 | Lifecycle policy is planned. Case storage primitives are available from #32. |
| [#39](https://github.com/jonpauljah/mh-risk-outreach/issues/39) | Worklists and case creation UI | #37, #38 | Planned. |
| [#40](https://github.com/jonpauljah/mh-risk-outreach/issues/40) | Active case management UI | #38, #39 | Planned. |
| [#41](https://github.com/jonpauljah/mh-risk-outreach/issues/41) | Manual reassessment and demo presets | #34–#38 | Streamlit submission flow and presets are planned. Backend submission and finalization are available from #32 and #35. |
| [#42](https://github.com/jonpauljah/mh-risk-outreach/issues/42) | Read-only history and comparisons | #38, #39, #41 | Planned. |

## Remaining Work and Implementation Order

1. Implement deterministic NBA rules (#36) using the assessment persistence
   and operational scoring from #32–#35.
2. Build eligibility and capacity policy (#37), then enforce the case lifecycle
   and transactional creation (#38). The manager-driven creation flow differs
   from client contract wording and requires sponsor approval, as #37 notes.
3. Integrate worklists (#39), case management (#40), and deliberate manual
   reassessment (#41) in Streamlit. Add read-only reporting (#42) after case and
   assessment histories are available. These groups may overlap where their
   contracts are agreed; they do not imply assignments or dates.

**Issue wording to update:** #41 asks for immediate recalculation and an atomic
scored submission. The backend saves a new current snapshot as
`PendingScoring` at submission. A separate scoring operation finalizes it
and publishes the updated worklist together. The UI and eligibility policy
must distinguish the latest submitted inputs from the inputs behind the
published score. See
[scoring and assessment workflow](architecture-and-workflows.md#risk-scoring-and-assessment-workflow).

## Milestone Deliverables

- A working manager-driven outreach flow from population scoring and separate
  safety review through recommendations, selected cases, outcomes, and history.
- Tests for database relationships and rollback, stable scoring, deterministic
  rules, eligibility and capacity, case transitions, and Streamlit workflows.
- Demonstrations of first-case baseline creation, a safety exception at capacity,
  an override and completed case, pending and subsequently scored reassessment,
  and history with and without a later assessment.
- Updated user guidance, verified results and limitations, and presentation
  material explaining the decisions and demonstration.

## Limitations and Scope Boundaries

The 10,000-row synthetic dataset and this local proof of concept do not establish
clinical validity or real-world performance. Outputs support human decisions;
they are not diagnoses, clinical thresholds, or automated treatment. The scope
does not add model families, automated treatment, or another database. Rule
thresholds and workflow assumptions require sponsor/domain validation.

Return to the [documentation index](../README.md).
