# Dataset placement

Place public healthcare CSV datasets for the POC in this directory. Dataset files
can be committed to Git so everyone uses the same inputs. Dataset selection, source attribution,
usage instructions, and field mappings are documented as they become available in
the [dataset guide](../docs/milestone-1/dataset.md) and
[ETL rules](../docs/milestone-1/etl-rules.md).

Use small synthetic datasets in `tests/fixtures/` for future automated tests.
No dataset or fixture is required for the scaffolding baseline test.

The selected `mental_health.csv` has been downloaded locally. See the dataset
guide for its source, counts, and checksum. At about 1.69 MiB,
it exceeds the default 1 MiB file-size limit. The hook excludes exactly
`data/mental_health.csv` so it can be committed and shared with the team.
Confirmation of the redistribution license remains pending.
