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

## Output

Running `main.py` produces (inside `use case 3/code/`):

- `stats/stats_anchors=6_time_interval=0.5.csv` — per-method error statistics
  (position error, distance error, azimuth error, direction accuracy,
  inference time, memory footprint)
- `stats/complexity_sensitivity.{csv,png,pdf}` — sensitivity plots

Intermediate cached results are written to `npy/`, `ground-truth/`, and
`path_files/` and can be deleted safely between runs.
