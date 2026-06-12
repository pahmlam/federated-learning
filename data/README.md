# Data

Local data staging area.

Do not commit large, private, or sensitive camera/face/PPE datasets directly to the repository. This directory is for local development and should later be paired with dataset documentation and ignore rules.

Expected layout:

- `raw/`: original downloaded or collected data.
- `processed/`: normalized/cropped/converted data ready for experiments.
- `partitions/`: client/site split files for IID and non-IID experiments.

