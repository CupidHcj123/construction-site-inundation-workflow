# Construction-site inundation risk-screening workflow

Code accompanying the manuscript, *A Modelling Workflow for Construction-site Inundation Risk Screening Using UAV LiDAR, Inertial-wave Simulation and Machine Learning*.

This release intentionally contains **code and configuration only**. It does not contain any site DSM, orthophoto, land-cover grid, pit mask, rainfall record, event table, model output, figure, or manuscript file.

## Repository layout

- `src/hydrodynamic/`: the inertial-wave model, build script, idealized/design-storm runners, and Chicago-hyetograph generator.
- `src/analysis/`: event extraction, grouped statistical analysis, and Random Forest/XGBoost classification pipeline.
- `src/sensitivity/`: Horton-infiltration sensitivity workflow.
- `src/figures_tables/`: scripts that reproduce manuscript figures and tables once authorized input data are supplied.
- `config/pipeline_config.example.json`: path-only configuration template for the analysis pipeline.
- `scripts/`: environment setup and dependency check.

## Environment

Create the Python environment with either Conda or a virtual environment:

```bash
conda env create -f environment.yml
conda activate xiongan
```

or:

```bash
python3 -m venv .venv-xiongan
source .venv-xiongan/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

## Hydrodynamic simulations

Compile the model on Linux with OpenMP support:

```bash
bash src/hydrodynamic/build_model.sh
```

The public runner scripts deliberately require paths to be supplied at execution time, rather than storing machine-specific paths. For an idealized-rainfall matrix:

```bash
DSM_PATH=/path/to/dsm.asc \
LC_PATH=/path/to/landcover.asc \
bash src/hydrodynamic/run_idealized_matrix.sh
```

For Chicago design storms:

```bash
DSM_PATH=/path/to/dsm.asc \
LC_PATH=/path/to/landcover.asc \
bash src/hydrodynamic/run_design_rain.sh
```

`ROW`, `COL`, `RESOLUTION`, `OUT_ROOT`, `HORTON_F0`, `HORTON_FC`, `HORTON_K`, `TPJ`, and `JOBS` can also be overridden as environment variables. The Chicago-hyetograph utility uses the Beijing IDF parameters cited in the manuscript; users should provide the applicable local design-storm standard for another study area.

## Analysis pipeline

Copy `config/pipeline_config.example.json` to a private configuration file, replace every path with an authorized local path, and run:

```bash
python src/analysis/run_pipeline.py --config config/pipeline_config.private.json
```

The private configuration file, all input data, and all generated results are ignored by Git. The pipeline uses grouped cross-validation by `Pit_ID`; the Random Forest and XGBoost components use a fixed random seed (`42`) for repeatable runs.

## Data and code availability

The construction-site spatial inputs and derived event-level data are not included in this public code-only release. They should be shared only when data ownership, site-access permission, and co-author approval allow it. A future archival release may link an authorized dataset through a DOI repository.

## Before public release

Confirm that the C++ solver is fully owned by the authors or that its original licence and required attribution permit redistribution. Add a licence only after this provenance check is complete.
