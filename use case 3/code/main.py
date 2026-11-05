import copy
import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import sys 
import util 
import dataset 
import matplotlib.ticker as ticker
import re
import glob

NUM_SHOPS = 6

def _extract_mean_stat(stats_df, index_name):
	if index_name not in stats_df.index:
		return np.nan
	row = stats_df.loc[index_name]
	if isinstance(row, pd.DataFrame):
		row = row.iloc[0]
	if "mean" in row.index and not pd.isna(row["mean"]):
		return float(row["mean"])
	vals = pd.to_numeric(row, errors="coerce")
	vals = vals[~pd.isna(vals)]
	return float(vals.iloc[0]) if len(vals) > 0 else np.nan

def generate_complexity_sensitivity_artifacts(stats_dir="../output", out_dir="../output", max_num_anchorss=None, time_intervals=None):
	"""Aggregate complexity stats across all tested settings and render sensitivity plots.

	Returns:
		(csv_path, png_path, pdf_path) or (None, None, None) if no stats files found.
	"""
	os.makedirs(out_dir, exist_ok=True)
	stats_files = glob.glob(os.path.join(stats_dir, "stats_anchors=*_time_interval=*.csv"))
	if len(stats_files) == 0:
		return None, None, None

	parsed_rows = []
	pattern = re.compile(r"stats_anchors=(\d+)_time_interval=([0-9.]+)\.csv$")
	for stats_file in stats_files:
		basename = os.path.basename(stats_file)
		match = pattern.match(basename)
		if match is None:
			continue
		anchors = int(match.group(1))
		ti = float(match.group(2))
		if max_num_anchorss is not None and anchors not in set(max_num_anchorss):
			continue
		if time_intervals is not None and ti not in set(float(v) for v in time_intervals):
			continue

		temp = pd.read_csv(stats_file)
		if temp.shape[1] < 2:
			continue
		temp = temp.set_index(temp.columns[0])

		for method_key, method_label in [("trilateration", "Multilateration"), ("ekf", "EKF"), ("UWB only", "UWB only")]:
			latency_ms = _extract_mean_stat(temp, f"Inference time ({method_key})")
			memory_mb = _extract_mean_stat(temp, f"Memory footprint ({method_key})")
			if not pd.isna(latency_ms):
				parsed_rows.append({
					"max_num_anchors": anchors,
					"time_interval": ti,
					"method_key": method_key,
					"method": method_label,
					"metric": "Inference Time (ms/update)",
					"value": float(latency_ms),
				})
			if not pd.isna(memory_mb):
				parsed_rows.append({
					"max_num_anchors": anchors,
					"time_interval": ti,
					"method_key": method_key,
					"method": method_label,
					"metric": "Memory (MB)",
					"value": float(memory_mb),
				})

	if len(parsed_rows) == 0:
		return None, None, None

	sensitivity_df = pd.DataFrame(parsed_rows).sort_values(["metric", "method", "time_interval", "max_num_anchors"]).reset_index(drop=True)
	csv_path = os.path.join(out_dir, "complexity_sensitivity.csv")
	sensitivity_df.to_csv(csv_path, index=False)

	methods_to_plot = [("trilateration", "Multilateration"), ("ekf", "EKF")]
	metrics_to_plot = ["Inference Time (ms/update)", "Memory (MB)"]
	fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
	for row_i, (method_key, method_label) in enumerate(methods_to_plot):
		for col_i, metric_name in enumerate(metrics_to_plot):
			ax = axes[row_i, col_i]
			subset = sensitivity_df[(sensitivity_df["method_key"] == method_key) & (sensitivity_df["metric"] == metric_name)]
			if len(subset) == 0:
				ax.set_title(f"{method_label} - {metric_name}")
				ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
				continue
			for ti, ti_df in subset.groupby("time_interval"):
				ti_df = ti_df.sort_values("max_num_anchors")
				ax.plot(ti_df["max_num_anchors"], ti_df["value"], marker="o", linewidth=1.5, label=f"TI={ti:g}s")
			ax.set_title(f"{method_label} - {metric_name}")
			ax.grid(True, alpha=0.3)
			ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
			if row_i == len(methods_to_plot) - 1:
				ax.set_xlabel("Maximum Number of Anchors")
			if col_i == 0:
				ax.set_ylabel("Value")

	handles, labels = axes[0, 0].get_legend_handles_labels()
	if len(handles) > 0:
		fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, -0.02))
	fig.tight_layout(rect=[0, 0.04, 1, 1])

	png_path = os.path.join(out_dir, "complexity_sensitivity.png")
	pdf_path = os.path.join(out_dir, "complexity_sensitivity.pdf")
	fig.savefig(png_path, dpi=300, bbox_inches="tight")
	fig.savefig(pdf_path, bbox_inches="tight")
	plt.close(fig)

	return csv_path, png_path, pdf_path

def inject_noise_into_consolidated_ranges(
	iphone_paths,
	data_type="unfiltered",
	range_mu_m=0.0,
	range_sigma_m=0.0,
	seed=42,
	min_distance_m=0.05,
):
	"""Inject Gaussian noise directly into raw UWB range logs used by direction inference.

	The current pipeline derives trilateration/EKF inputs from
	iphone_paths[path_id]["consolidated"][data_type][i]["distance"], so perturbing this field
	keeps the evaluation consistent with the implemented method.
	"""
	rng = np.random.default_rng(seed)
	updated = 0

	for path_id in iphone_paths:
		if "consolidated" not in iphone_paths[path_id]:
			continue
		if data_type not in iphone_paths[path_id]["consolidated"]:
			continue

		for row in iphone_paths[path_id]["consolidated"][data_type]:
			if "distance" not in row:
				continue
			try:
				base = float(row["distance"])
			except Exception:
				continue

			if np.isnan(base):
				continue

			noise = rng.normal(range_mu_m, range_sigma_m)
			row["distance"] = max(min_distance_m, base + noise)
			updated += 1

	return updated

def extract_classification_accuracy(stats_df):
	"""Extract direction classification accuracy rows from stats dataframe."""
	rows = []
	for idx in stats_df.index:
		idx_str = str(idx)
		if not idx_str.startswith("Classification ("):
			continue
		method = idx_str.replace("Classification (", "").replace(")", "")
		row = stats_df.loc[idx]
		rows.append({
			"Method": method,
			"Samples": int(row.get("Samples", 0)) if not pd.isna(row.get("Samples", np.nan)) else np.nan,
			"Accuracy": float(row.get("Accuracy", np.nan)) if not pd.isna(row.get("Accuracy", np.nan)) else np.nan,
			"Accuracy CI Lower": float(row.get("Accuracy CI Lower", np.nan)) if not pd.isna(row.get("Accuracy CI Lower", np.nan)) else np.nan,
			"Accuracy CI Upper": float(row.get("Accuracy CI Upper", np.nan)) if not pd.isna(row.get("Accuracy CI Upper", np.nan)) else np.nan,
		})
	if len(rows) == 0:
		return pd.DataFrame(columns=["Method", "Samples", "Accuracy", "Accuracy CI Lower", "Accuracy CI Upper"])
	return pd.DataFrame(rows).sort_values("Method").reset_index(drop=True)

def run_noise_injection_direction_experiment(
	iphone_paths,
	true_paths,
	shop_locations,
	imu_data,
	methods,
	directions_mode,
	azimuth_fusion_methods,
	max_num_anchors=6,
	time_interval=0.5,
	range_mu_m=0.15,
	range_sigma_m=0.30,
	seed=42,
	stats_dir="../output",
):
	"""Run baseline vs noisy direction-inference evaluation using existing implementation."""
	os.makedirs(stats_dir, exist_ok=True)

	base_paths = copy.deepcopy(iphone_paths)
	if "consolidated" not in next(iter(base_paths.values())):
		dataset.consolidate_shop_measurements_by_timestamp(base_paths, "unfiltered")

	scenarios = [
		{"name": "baseline", "mu": 0.0, "sigma": 0.0, "seed": seed},
		{"name": "nlos_like", "mu": range_mu_m, "sigma": range_sigma_m, "seed": seed},
	]

	all_rows = []
	for scenario in scenarios:
		paths_eval = copy.deepcopy(base_paths)
		updated = inject_noise_into_consolidated_ranges(
			paths_eval,
			data_type="unfiltered",
			range_mu_m=scenario["mu"],
			range_sigma_m=scenario["sigma"],
			seed=scenario["seed"],
		)
		print(
			f"[Noise scenario={scenario['name']}] injected rows={updated}, "
			f"mu={scenario['mu']:.3f}m, sigma={scenario['sigma']:.3f}m"
		)

		paths_eval = dataset.average_iphone_positions(
			paths_eval,
			shop_locations,
			time_interval,
			max_num_anchors,
			methods,
			imu_data,
			dilateration_enabled=False,
			use_cached_outputs=False,
		)

		(_, _, _, stats_df, dir_accs, *_) = dataset.evaluate_motion_errors(
			paths_eval,
			true_paths,
			shop_locations,
			methods,
			directions_mode,
			azimuth_fusion_methods,
		)

		stats_path = os.path.join(
			stats_dir,
			f"stats_noise={scenario['name']}_anchors={max_num_anchors}_time_interval={time_interval}.csv",
		)
		stats_df.to_csv(stats_path)
		print(f"Saved noise scenario stats -> {stats_path}")

		cls = extract_classification_accuracy(stats_df)
		if len(cls) > 0:
			cls["Scenario"] = scenario["name"]
			cls["RangeMu(m)"] = scenario["mu"]
			cls["RangeSigma(m)"] = scenario["sigma"]
			all_rows.append(cls)
		print(f"Direction accuracy summary ({scenario['name']}):")
		print(cls)
		print("Direction-by-category summary:", dir_accs)

	if len(all_rows) == 0:
		return None

	summary_df = pd.concat(all_rows, axis=0).reset_index(drop=True)
	summary_path = os.path.join(
		stats_dir,
		f"noise_injection_summary_anchors={max_num_anchors}_time_interval={time_interval}.csv",
	)
	summary_df.to_csv(summary_path, index=False)
	print(f"Saved noise injection summary -> {summary_path}")
	return summary_path

if __name__ == "__main__":
	lidar_data_dir = "../data/lidar"
	iphone_data_dir = "../data/iphone"
	start_path_id, end_path_id = 26, 26
	lidar_orientations = True 
	by_measured = True
	cache_paths_on_disk = True
	methods = ["trilateration", "ekf"]
	dilateration_enabled = False  
	run_noise_injection_experiment = False
	directions_mode = "all without front and back"
	azimuth_fusion_methods = []
	max_num_anchorss = [6]
	time_intervals = [0.5] 
	print("Methods:", methods)
	print("Dilateration enabled:", dilateration_enabled)
	if not os.path.exists("ground-truth"):
		os.makedirs("ground-truth")
	if not os.path.exists("path_files"):
		os.makedirs("path_files")
	
	shop_locations, true_paths, iphone_paths, _, _, imu_data = dataset.load_data(
		"npy/shops.csv",
		[],
		lidar_data_dir, 
		lidar_orientations,
		iphone_data_dir,
		by_measured,
		start=start_path_id,
		finish=end_path_id,
		cache_paths_on_disk=cache_paths_on_disk
	)
	print("Shop locations:", shop_locations)
	print("\n=== PATHS BEING PROCESSED ===")
	print("True path ids:", sorted(true_paths.keys()))
	print("iPhone path ids:", sorted(iphone_paths.keys()))
	print(f"Total paths: {len(true_paths)}")
	print("============================\n")

	if run_noise_injection_experiment:
		summary_csv = run_noise_injection_direction_experiment(
			iphone_paths=iphone_paths,
			true_paths=true_paths,
			shop_locations=shop_locations,
			imu_data=imu_data,
			methods=methods,
			directions_mode=directions_mode,
			azimuth_fusion_methods=azimuth_fusion_methods,
			max_num_anchors=6,
			time_interval=0.5,
			range_mu_m=0.15,
			range_sigma_m=0.30,
			seed=42,
			stats_dir="../output",
		)
		print("Noise injection experiment completed.")
		if summary_csv is not None:
			print(f"Summary file: {summary_csv}")
		sys.exit(0)
	
	for max_num_anchors in max_num_anchorss:
		for time_interval in time_intervals[::-1]:
			print(f"Time interval: {time_interval}, max num anchors: {max_num_anchors}")
			iphone_paths = dataset.average_iphone_positions(iphone_paths, shop_locations, time_interval, max_num_anchors, methods, imu_data, dilateration_enabled=dilateration_enabled, use_cached_outputs=True)

			(_, _, _, stats_df, dir_accs, *_) = dataset.evaluate_motion_errors(iphone_paths, true_paths, shop_locations, methods, directions_mode, azimuth_fusion_methods)
			stats_df.to_csv(f"../output/stats_anchors={max_num_anchors}_time_interval={time_interval}.csv")
			print("dir_accs:", dir_accs)

	sensitivity_csv_path, sensitivity_png_path, sensitivity_pdf_path = generate_complexity_sensitivity_artifacts(
		stats_dir="../output",
		out_dir="../output",
		max_num_anchorss=max_num_anchorss,
		time_intervals=time_intervals,
	)
	if sensitivity_csv_path:
		print(f"Generated complexity sensitivity CSV: {sensitivity_csv_path}")
		print(f"Generated complexity sensitivity plot (PNG): {sensitivity_png_path}")
		print(f"Generated complexity sensitivity plot (PDF): {sensitivity_pdf_path}")

	