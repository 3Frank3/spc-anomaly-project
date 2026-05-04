# SPC + Anomaly Detection Starter

This project combines Statistical Process Control (SPC) with machine learning anomaly detection for process monitoring.

It starts from synthetic manufacturing-style sensor data, then:

- Calculates I-MR chart limits
- Applies common SPC rules
- Trains an Isolation Forest anomaly detector
- Compares SPC signals with model-based anomaly flags
- Exports plots and a short monitoring report

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   └── figures/
├── src/
│   └── spc_anomaly/
│       ├── anomaly.py
│       ├── data.py
│       ├── pipeline.py
│       ├── plotting.py
│       └── spc.py
├── tests/
├── environment.yml
├── requirements.txt
└── README.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate SPC
```

Optional, to keep Matplotlib cache files inside the project:

```bash
mkdir -p .matplotlib-cache
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
```

If the environment already exists, update it instead:

```bash
conda env update -n SPC -f environment.yml --prune
conda activate SPC
```

## Run the Full Pipeline

```bash
python -m src.spc_anomaly.pipeline
```

Outputs:

- `data/raw/process_data.csv`
- `data/processed/scored_process_data.csv`
- `outputs/figures/spc_chart.png`
- `outputs/figures/anomaly_scores.png`
- `outputs/report.md`

## Run Tests

```bash
pytest
```

## What to Try Next

- Replace the synthetic data with real sensor or quality measurements.
- Tune SPC rules and Isolation Forest contamination for your process.
- Add subgroup-based X-bar/R charts if your data is collected in batches.
- Add a dashboard with Streamlit.
