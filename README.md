# iotj-uwb

Code and data accompanying the IEEE Internet of Things Journal paper on
UWB-based indoor localization.

## Repository structure

- `use case 1/` — static positioning experiments
- `use case 2/` — two-anchor positioning experiments
- `use case 3/` — multi-anchor UWB/IMU fusion experiments (code + sample data)

## Requirements

Python 3.8+ with the following packages:

```bash
pip install numpy pandas matplotlib scipy
```

## Running Use Case 3

Use Case 3 loads the included P26 sample dataset (`use case 3/data/`), runs the
multilateration (trilateration) and EKF pipelines, and evaluates them against
the LiDAR ground truth.

```bash
cd "use case 3/code"
python3 main.py
```

`main.py` is configured to process a single path (P26, via
`start_path_id = end_path_id = 26`). To run other paths, add the corresponding
data under `use case 3/data/` and update `start_path_id` / `end_path_id` in
`main.py`.

Key configuration in `main.py`:

- `methods = ["trilateration", "ekf"]`
- `max_num_anchorss = [6]`
- `time_intervals = [0.5]`

## Data layout (`use case 3/data/`)

The input data is organized by batch number. Batch folders are named `batch1`,
`batch2`, … (no space).

- `iphone/` — iPhone sensor logs, one batch folder per run:
  - `iphone/batch1/B26_UWB.csv` — UWB range/azimuth measurements
  - `iphone/batch1/B26_IMU.csv` — inertial (IMU) measurements
- `lidar/` — LiDAR logs, one batch folder per run. The lidar folder must
  contain the ground-truth locations of the anchors (one `S*.log` file per
  anchor, e.g. `S1.log`, `S2.log`, …) and the ground-truth trajectory of each
  measured path (one `P*.log` file per path, e.g. `P26.log`).

## Output

Running `main.py` writes the evaluation results to `use case 3/output/`:

- `stats_anchors=6_time_interval=0.5.csv` — per-method error statistics
  (position error, distance error, azimuth error, direction accuracy,
  inference time, memory footprint)
- `complexity_sensitivity.{csv,png,pdf}` — sensitivity plots

Intermediate cached results are written to `npy/`, `ground-truth/`, and
`path_files/` inside `use case 3/code/` and can be deleted safely between runs.
