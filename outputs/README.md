# Outputs

Local output area for experiment artifacts.

Expected layout:

- `logs/`: runtime logs.
- `metrics/`: JSON/CSV metric exports.
- `checkpoints/`: model checkpoints and Flower artifacts.
- `reports/`: generated experiment summaries.

Large outputs should not be committed. Add ignore rules before storing real experiment artifacts here.

