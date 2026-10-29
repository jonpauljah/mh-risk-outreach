# Documentation

Start with [getting started](../getting-started.md) to install dependencies and
run the current application and baseline test.

| Folder | What you will find |
| --- | --- |
| [Architecture](architecture/README.md) | Index of architecture drafts currently maintained in Milestone 1 |
| [Milestone 1](milestone-1/README.md) | Dataset details, ETL rules, architecture decisions, and deliverable verification |
| [ML spec](ml-spec.md) | The risk-scoring model: design, the extensible model framework, training/evaluation, and measured results |
| [Risk scoring](risk-scoring.md) | Scoring & prioritization (issue #16): SHAP reasons, K-means risk bands, the critical-indicator override, and the per-model `<model>_scores` tables |
| [Dependencies](dependencies/README.md) | Runtime and development tools, commands, and dependency changes |
| [Hooks](hooks/README.md) | Setup and use of the local pre-push checks |

The current application is a scaffold. Documents distinguish existing behavior
from planned milestone 1 behavior. Sections marked **Pending** need a confirmed
decision or implementation evidence before they can be completed.
