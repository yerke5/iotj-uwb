import numpy as np
import matplotlib.pyplot as plt 
import math

DUMMY = -1000 

def calculate_classification_metrics(true_labels, predicted_labels): 
	# print("Number of nan values in true labels:", len(true_labels[np.isnan(true_labels)]))
	# print("Number of nan values in predicted labels:", len(predicted_labels[np.isnan(predicted_labels)]))
	metrics = ['F1', 'Macro F1', 'Weighted F1', 'Precision', 'Recall', 'Accuracy', 'Sensitivity', 'Specificity'] 
	result = {metric: 0 for metric in metrics}

	# Convert to numpy arrays for easier manipulation
	true_labels = np.array(true_labels)
	predicted_labels = np.array(predicted_labels)
	
	# Calculate overall Accuracy
	result['Accuracy'] = np.sum(predicted_labels == true_labels) / len(true_labels) if len(true_labels) > 0 else 0
	
	# Get unique classes (should be 4 classes)
	classes = np.unique(np.concatenate([true_labels, predicted_labels]))
	
	# Calculate metrics per class using one-vs-all approach, then macro average
	precisions = []
	recalls = []
	specificities = []
	f1_scores = []
	supports = []
	
	for cls in classes:
		# For this class: treat it as positive, all others as negative
		true_positives = np.sum((predicted_labels == cls) & (true_labels == cls))
		true_negatives = np.sum((predicted_labels != cls) & (true_labels != cls))
		false_positives = np.sum((predicted_labels == cls) & (true_labels != cls))
		false_negatives = np.sum((predicted_labels != cls) & (true_labels == cls))
		
		# Precision for this class
		precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
		precisions.append(precision)
		
		# Recall for this class
		recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
		recalls.append(recall)

		# F1 for this class
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
		f1_scores.append(f1)
		supports.append(np.sum(true_labels == cls))
		
		# Specificity for this class
		specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
		specificities.append(specificity)
	
	# Macro average across all classes
	result['Precision'] = np.mean(precisions) if len(precisions) > 0 else 0
	result['Recall'] = np.mean(recalls) if len(recalls) > 0 else 0
	result['Sensitivity'] = result['Recall']  # Sensitivity is same as Recall
	result['Specificity'] = np.mean(specificities) if len(specificities) > 0 else 0
	
	# F1 scores
	result['Macro F1'] = np.mean(f1_scores) if len(f1_scores) > 0 else 0
	result['Weighted F1'] = np.average(f1_scores, weights=supports) if np.sum(supports) > 0 else 0
	result['F1'] = result['Macro F1']

	return result 

def bootstrap_classification_metrics_mbb(true_labels, predicted_labels, n_bootstrap=2000, confidence=0.95, block_length=None, random_state=42):
	"""Moving-block bootstrap for classification metrics.
	
	Unlike the regular bootstrap which resamples individual observations
	(assuming independence), this method resamples contiguous blocks to
	preserve local temporal dependence in trajectory data.
	
	Args:
		true_labels, predicted_labels: chronologically ordered arrays
		n_bootstrap: number of bootstrap iterations
		confidence: confidence level for intervals
		block_length: block length for MBB (if None, estimated from ACF)
		random_state: seed for reproducibility
	"""
	true_labels = np.array(true_labels)
	predicted_labels = np.array(predicted_labels)
	n = len(true_labels)
	if n == 0:
		return {}
	
	# Estimate block length from ACF of correctness sequence if not provided
	if block_length is None:
		correctness = (predicted_labels == true_labels).astype(float)
		# Compute ACF up to lag 30
		mean_c = np.mean(correctness)
		var_c = np.var(correctness)
		if var_c > 0:
			max_lag = min(30, n // 4)
			acf_vals = np.zeros(max_lag + 1)
			acf_vals[0] = 1.0
			for k in range(1, max_lag + 1):
				if n - k <= 1:
					break
				acf_vals[k] = np.sum((correctness[:-k] - mean_c) * (correctness[k:] - mean_c)) / ((n - k) * var_c)
			# Decorrelation lag: first lag where |ACF| < 0.05 for 3 consecutive lags
			block_length = 1
			for k in range(1, len(acf_vals) - 3):
				if all(abs(acf_vals[k + i]) < 0.05 for i in range(4)):
					block_length = k
					break
			if block_length < 1:
				block_length = 1
		else:
			block_length = 1
	
	block_length = max(1, int(block_length))
	n_blocks = max(2, int(np.ceil(n / block_length)))
	
	rng = np.random.default_rng(random_state)
	metric_names = ['Accuracy', 'Precision', 'Recall', 'Macro F1', 'Weighted F1', 'Sensitivity', 'Specificity']
	bootstrap_values = {metric: [] for metric in metric_names}
	point_estimates = calculate_classification_metrics(true_labels, predicted_labels)
	
	for _ in range(n_bootstrap):
		# Sample block start indices
		block_starts = rng.integers(0, max(1, n - block_length + 1), size=n_blocks)
		# Build bootstrap sample from contiguous blocks
		indices = np.concatenate([np.arange(s, min(s + block_length, n)) for s in block_starts])
		indices = indices[:n]  # trim to original length
		
		metrics = calculate_classification_metrics(true_labels[indices], predicted_labels[indices])
		for metric in metric_names:
			bootstrap_values[metric].append(metrics[metric])
	
	alpha = (1 - confidence) / 2
	result = {}
	for metric in metric_names:
		values = np.array(bootstrap_values[metric], dtype=float)
		result[metric] = {
			'mean': float(point_estimates[metric]),
			'ci_lower': float(np.quantile(values, alpha)),
			'ci_upper': float(np.quantile(values, 1 - alpha)),
			'block_length': int(block_length),
		}
	
	return result

def _compute_acf(series, max_lag=30):
	series = np.asarray(series, float)
	n = len(series)
	mean, var = np.mean(series), np.var(series)
	if var == 0:
		return np.zeros(min(max_lag + 1, n))
	acf = np.zeros(min(max_lag + 1, n))
	acf[0] = 1.0
	for k in range(1, len(acf)):
		if n - k <= 1:
			break
		acf[k] = np.sum((series[:-k] - mean) * (series[k:] - mean)) / ((n - k) * var)
	return acf

def _decorrelation_lag(acf, threshold=0.05):
	for k in range(1, len(acf) - 3):
		if all(abs(acf[k + i]) < threshold for i in range(4)):
			return k
	return max(1, len(acf) - 1)

def paired_accuracy_block_bootstrap(correct_a, total_a, correct_b, total_b,
                                     n_bootstrap=2000, confidence=0.95,
                                     block_length=None, random_state=42):
	"""MBB CI for total accuracy difference between two methods.

	Takes per-timestamp counts for two methods (already aligned by the caller
	to a common time grid). Block-bootstraps timestamps and recomputes the
	total accuracy difference Σc_A/Σn_A - Σc_B/Σn_B for each bootstrap replicate.

	Parameters
	----------
	correct_a, total_a : 1-D arrays of length T
		Per-timestamp correct count and total count for method A.
	correct_b, total_b : 1-D arrays of length T
		Per-timestamp correct count and total count for method B.
	block_length : int or None
		If None, estimated from decorrelation lag of per-timestamp accuracy
		difference sequence.

	Returns
	-------
	point_diff : float                 total accuracy difference (A - B)
	ci_lower, ci_upper : float         95% MBB CI
	block_length : int                 block length used
	"""
	correct_a = np.asarray(correct_a, float)
	total_a = np.asarray(total_a, float)
	correct_b = np.asarray(correct_b, float)
	total_b = np.asarray(total_b, float)

	mask = (total_a > 0) & (total_b > 0)
	T = int(np.sum(mask))
	if T < 10:
		return np.nan, np.nan, np.nan, 0

	ca = correct_a[mask]
	ta = total_a[mask]
	cb = correct_b[mask]
	tb = total_b[mask]

	acc_diff = ca / ta - cb / tb
	point_diff = np.sum(ca) / np.sum(ta) - np.sum(cb) / np.sum(tb)

	if block_length is None:
		a = _compute_acf(acc_diff, max_lag=30)
		block_length = _decorrelation_lag(a, threshold=0.05)
	block_length = max(1, int(block_length))

	nblk = max(2, int(np.ceil(T / block_length)))
	rng = np.random.default_rng(random_state)
	bdiffs = np.zeros(n_bootstrap)

	for i in range(n_bootstrap):
		starts = rng.integers(0, max(1, T - block_length + 1), size=nblk)
		idx = np.concatenate([np.arange(s, min(s + block_length, T)) for s in starts])
		idx = idx[:T]
		cas, tas = np.sum(ca[idx]), np.sum(ta[idx])
		cbs, tbs = np.sum(cb[idx]), np.sum(tb[idx])
		bdiffs[i] = (cas / tas) - (cbs / tbs) if (tas > 0 and tbs > 0) else 0.0

	alpha = (1 - confidence) / 2
	ci_lo = float(np.quantile(bdiffs, alpha))
	ci_hi = float(np.quantile(bdiffs, 1 - alpha))

	return float(point_diff), ci_lo, ci_hi, block_length

def _regularized_gamma_p(a, x, eps=1e-12, max_iter=1000):
	if x <= 0:
		return 0.0
	if x < a + 1.0:
		ap = a
		delta = 1.0 / a
		sum_val = delta
		for _ in range(max_iter):
			ap += 1.0
			delta *= x / ap
			sum_val += delta
			if abs(delta) < abs(sum_val) * eps:
				break
		return sum_val * math.exp(-x + a * math.log(x) - math.lgamma(a))

	b = x + 1.0 - a
	c = 1.0 / 1e-30
	d = 1.0 / b
	h = d
	for i in range(1, max_iter + 1):
		an = -i * (i - a)
		b += 2.0
		d = an * d + b
		if abs(d) < 1e-30:
			d = 1e-30
		c = b + an / c
		if abs(c) < 1e-30:
			c = 1e-30
		d = 1.0 / d
		delta = d * c
		h *= delta
		if abs(delta - 1.0) < eps:
			break
	return 1.0 - h * math.exp(-x + a * math.log(x) - math.lgamma(a))

def _regularized_gamma_q(a, x, eps=1e-12, max_iter=1000):
	return 1.0 - _regularized_gamma_p(a, x, eps=eps, max_iter=max_iter)

def _chi_square_survival(statistic, degrees_of_freedom):
	if statistic < 0:
		raise ValueError("chi-square statistic must be non-negative")
	if degrees_of_freedom <= 0:
		raise ValueError("degrees_of_freedom must be positive")
	return _regularized_gamma_q(degrees_of_freedom / 2.0, statistic / 2.0)

def chi_square_test_independence(correct_counts, total_counts):
	correct_counts = np.array(correct_counts, dtype=float)
	total_counts = np.array(total_counts, dtype=float)
	if len(correct_counts) != len(total_counts):
		raise ValueError("correct_counts and total_counts must have the same length")
	if np.any(correct_counts < 0) or np.any(total_counts <= 0) or np.any(correct_counts > total_counts):
		raise ValueError("Counts must satisfy 0 <= correct_counts <= total_counts and total_counts > 0")

	incorrect_counts = total_counts - correct_counts
	observed = np.column_stack([correct_counts, incorrect_counts])
	row_totals = observed.sum(axis=1, keepdims=True)
	col_totals = observed.sum(axis=0, keepdims=True)
	grand_total = observed.sum()
	expected = row_totals @ col_totals / grand_total

	# Degenerate case: one outcome column has zero total (e.g., all predictions are correct).
	# In this case chi-square is not informative; return a neutral, non-significant result.
	nonzero_row_count = int(np.sum(row_totals.ravel() > 0))
	nonzero_col_count = int(np.sum(col_totals.ravel() > 0))
	degrees_of_freedom = int((nonzero_row_count - 1) * (nonzero_col_count - 1))
	if degrees_of_freedom <= 0:
		return {
			'statistic': 0.0,
			'p_value': 1.0,
			'degrees_of_freedom': 0,
			'expected': expected,
		}

	valid_cells = expected > 0
	statistic = float(np.sum(((observed - expected) ** 2 / expected)[valid_cells]))
	p_value = float(_chi_square_survival(statistic, degrees_of_freedom))
	return {
		'statistic': statistic,
		'p_value': p_value,
		'degrees_of_freedom': degrees_of_freedom,
		'expected': expected,
	}

def two_proportion_z_test(successes_a, total_a, successes_b, total_b, ess_ratio=0.059):
	if total_a <= 0 or total_b <= 0:
		raise ValueError("Sample sizes must be positive")
	if not (0 <= successes_a <= total_a and 0 <= successes_b <= total_b):
		raise ValueError("Success counts must lie between 0 and total sample sizes")

	proportion_a = successes_a / total_a
	proportion_b = successes_b / total_b
	pooled_proportion = (successes_a + successes_b) / (total_a + total_b)
	# ESS correction: inflate SE to account for temporal dependence in trajectory data
	standard_error = math.sqrt(
		pooled_proportion * (1.0 - pooled_proportion) * (1.0 / (total_a * ess_ratio) + 1.0 / (total_b * ess_ratio))
	)
	if standard_error == 0:
		z_statistic = 0.0
		p_value = 1.0
	else:
		z_statistic = (proportion_a - proportion_b) / standard_error
		p_value = math.erfc(abs(z_statistic) / math.sqrt(2.0))

	return {
		'z_statistic': float(z_statistic),
		'p_value': float(p_value),
		'proportion_a': float(proportion_a),
		'proportion_b': float(proportion_b),
		'difference': float(proportion_a - proportion_b),
	}

def bonferroni_correction(p_values):
	p_values = list(p_values)
	m = len(p_values)
	if m == 0:
		return []
	return [min(1.0, p_value * m) for p_value in p_values]

def _confusion_label_mapping():
	return {
		-2: "back left",
		-1: "front left",
		0: "front",
		1: "front right",
		2: "back right",
		3: "back"
	}

def _build_confusion_matrix(true_labels, predicted_labels, classes=None):
	true_labels = np.array(true_labels)
	predicted_labels = np.array(predicted_labels)

	if classes is None:
		classes = np.unique(np.concatenate([true_labels, predicted_labels]))
	classes = np.array(classes)
	conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

	for i, true_class in enumerate(classes):
		for j, pred_class in enumerate(classes):
			conf_matrix[i, j] = np.sum((true_labels == true_class) & (predicted_labels == pred_class))

	return conf_matrix, classes

def _plot_confusion_matrix_on_axis(ax, conf_matrix, classes, title, show_ylabel=False):
	label_mapping = _confusion_label_mapping()
	class_labels = [label_mapping.get(c, str(c)).capitalize() for c in classes]
	im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	ax.set(
		xticks=np.arange(len(classes)),
		yticks=np.arange(len(classes)),
		xticklabels=class_labels,
		yticklabels=class_labels,
		title=title,
	)
	if not show_ylabel:
		ax.set_yticklabels([])

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	thresh = conf_matrix.max() / 2.0 if conf_matrix.size > 0 else 0.0
	for i in range(conf_matrix.shape[0]):
		row_total = np.sum(conf_matrix[i, :])
		for j in range(conf_matrix.shape[1]):
			count = conf_matrix[i, j]
			percentage = 100.0 * count / row_total if row_total > 0 else 0.0
			ax.text(
				j,
				i,
				f'{count:d}\n({percentage:.1f}%)',
				ha="center",
				va="center",
				color="white" if conf_matrix[i, j] > thresh else "black",
			)

	return im

def calculate_position_availability(timestamps, time_interval):
	last_ts = timestamps[-1]
	i = 0
	c = 0
	t = 0 
	total_intervals = 0
	first_ts = timestamps[0]
	while t < len(timestamps) and timestamps[t] < last_ts:
		start_ts = first_ts + time_interval * i 
		end_ts = first_ts + time_interval * (i + 1)
		#print("start_ts:", start_ts - first_ts, "end_ts:", end_ts - first_ts)
		while t < len(timestamps) and timestamps[t] < start_ts:
			t += 1
		# we are now after the start_ts, let's check if it's less than end_ts 
		if t < len(timestamps):
			if timestamps[t] <= end_ts:
				#print(timestamps[t] - first_ts, "is between", start_ts - first_ts, "and", end_ts - first_ts)
				c += 1 
			total_intervals += 1
		t += 1
		i += 1

	return c, total_intervals

def azimuth_mean(azimuths):
	azimuths = convert_azimuths360(azimuths) 
	avg = circular_average(azimuths)  
	avg = avg - 360 if avg >= 180 else avg
	return avg

def circular_average(angles_deg):
	"""
	Calculates the circular average of a list of angles in degrees.
	The result is returned in the [0, 360) range.
	
	Args:
		angles_deg (array-like): A list or numpy array of angles in degrees.

	Returns:
		float: The circular average of the angles in degrees.
	"""
	# Convert angles from degrees to radians for trigonometric functions
	angles_rad = np.deg2rad(angles_deg)
	
	# Convert each angle to a 2D vector (x, y) on the unit circle
	x_coords = np.cos(angles_rad)
	y_coords = np.sin(angles_rad)
	
	# Compute the average of these vectors
	mean_x = np.mean(x_coords)
	mean_y = np.mean(y_coords)
	
	# Convert the average vector back to an angle in radians using atan2
	# atan2 correctly handles all quadrants and gives an angle in [-pi, pi]
	mean_angle_rad = np.arctan2(mean_y, mean_x)
	
	# Convert the resulting angle from radians back to degrees
	mean_angle_deg = np.rad2deg(mean_angle_rad)
	
	# Normalize the angle to be in the [0, 360) range
	return (mean_angle_deg + 360) % 360

def azimuth_means(angles1, angles2):
	angles1 = convert_azimuths360(angles1)
	angles2 = convert_azimuths360(angles2)
	angles_rad1 = np.deg2rad(angles1)
	angles_rad2 = np.deg2rad(angles2)

	# Convert each angle to a 2D vector (x, y) on the unit circle
	x_coords1 = np.cos(angles_rad1)
	y_coords1 = np.sin(angles_rad1)

	x_coords2 = np.cos(angles_rad2)
	y_coords2 = np.sin(angles_rad2)
	
	# Compute the average of these vectors
	mean_x = (x_coords1 + x_coords2) / 2
	mean_y = (y_coords1 + y_coords2) / 2

	# Convert the average vector back to an angle in radians using atan2
	# atan2 correctly handles all quadrants and gives an angle in [-pi, pi]
	mean_angle_rad = np.arctan2(mean_y, mean_x)
	
	# Convert the resulting angle from radians back to degrees
	mean_angle_deg = np.rad2deg(mean_angle_rad)
	
	# Normalize the angle to be in the [0, 360) range
	mean_angle_deg = (mean_angle_deg + 360) % 360
	mean_angle_deg[mean_angle_deg >= 180] -= 360
	
	return mean_angle_deg

def convert_azimuths360(azimuths):
	azimuths = np.array(azimuths)
	azimuths[azimuths < 0] += 360
	return azimuths

def angular_difference(a1, a2):
	"""
	Calculate the angular difference between two compass angles.
	Handles the circular nature of angles correctly.
	
	Args:
		a1: First angle in degrees (0-360)
		a2: Second angle in degrees (0-360)
	
	Returns:
		Angular difference in degrees (0-180)
	
	Example:
		angular_difference(0, 355) returns 5
		angular_difference(10, 350) returns 20
		angular_difference(180, 0) returns 180
	"""
	# Convert to numpy arrays to handle both single values and arrays
	a1 = np.asarray(a1)
	a2 = np.asarray(a2)
	
	# Calculate the raw difference
	diff = np.abs(a1 - a2)
	
	# Handle the circular nature - if difference > 180, take the complement
	diff = np.where(diff > 180, 360 - diff, diff)
	
	return diff

def azimuth_error(angle1, angle2):
	"""
	Calculate azimuth angle error between two angles.
	This is an alias for angular_difference for clarity.
	
	Args:
		angle1: First compass angle in degrees (0-360)
		angle2: Second compass angle in degrees (0-360)
	
	Returns:
		Angle error in degrees (0-180)
	
	Example:
		compass_angle_error(0, 355) returns 5
		compass_angle_error(10, 350) returns 20
	"""
	angle1 = convert_azimuths360(angle1)
	angle2 = convert_azimuths360(angle2)
	return angular_difference(angle1, angle2)

def positions2orientations(positions, steps=1):
	orientations = np.zeros((len(positions),))
	for i in range(len(positions)):
		if i < steps:
			orientations[i] = np.nan
		else:
			delta = positions[i] - positions[i - steps]
			orientations[i] = calculate_vector_theta(delta) 
	return degrees2radians(orientations)

def degrees2radians(deg):
	return deg / 180 * np.pi 

def radians2degrees(rad):
	return rad / np.pi * 180

def calculate_vector_theta(vector):
	x, y = vector[0], vector[1]
	theta = np.arctan(np.abs(y) / np.abs(x)) / np.pi * 180

	# determine quadrant 
	if x > 0: # quadrant I or IV
		if y < 0: # quadrant IV
			theta = 360 - theta
	else:
		if y > 0: # quadrant II
			theta = 180 - theta
		else: # quadrant III
			theta = 180 + theta
	return theta

def get_orientation(position, idx, orientation_steps):
	if idx < 1:
		raise Exception("Orientation cannot be calculated for the first position")
	position = np.array(position)
	delta = position[idx] - position[idx - orientation_steps]
	theta = degrees2radians((calculate_vector_theta(delta)))
	return theta

def calculate_vector_length(vector):
	return np.sqrt(vector[0]**2 + vector[1]**2)

def calculate_vector_lengths(vectors):
	return np.sqrt(np.sum(vectors**2, axis=1))

def theta2orientation_vector(theta):
	return np.concatenate([
		np.cos(theta).reshape(-1, 1), 
		np.sin(theta).reshape(-1, 1)], axis=1
	)

def theta_to_bearing(theta):
	return (450 - theta) % 360

def bearing_to_theta(bearing):
	return (450 - bearing) % 360

def rotate_vector(vector, theta):
	theta = theta / 180 * np.pi
	return np.concatenate([
		(vector[:, 0] * np.cos(theta) - vector[:, 1] * np.sin(theta)).reshape(-1, 1),
		(vector[:, 0] * np.sin(theta) + vector[:, 1] * np.cos(theta)).reshape(-1, 1)
	], axis=1)

def find_closest_ground_truth_indices(true_timestamps, measured_timestamps, by_measured):
	#true_timestamps -= true_timestamps[0]

	# if not by_measured:
	# 	measured_timestamps, true_timestamps = true_timestamps, measured_timestamps
	
	# find index of min timestamp
	min_measured_ts, max_measured_ts = measured_timestamps[0], measured_timestamps[-1]
	min_true_ts, max_true_ts = true_timestamps[0], true_timestamps[-1]
	
	true_idxs = np.array(list(range(len(true_timestamps))))
	measured_idxs = np.array(list(range(len(measured_timestamps))))
	# if min_measured_ts > min_true_ts:
	# 	true_timestamps = np.array(true_timestamps)
	# 	if not by_measured:
	# 		true_idxs = true_idxs[true_timestamps >= min_measured_ts]
	# 	true_timestamps = true_timestamps[true_timestamps >= min_measured_ts]
		
	# else:
	# 	measured_timestamps = np.array(measured_timestamps)
	# 	if by_measured:
	# 		measured_idxs = measured_idxs[measured_timestamps >= min_true_ts]
	# 	measured_timestamps = measured_timestamps[measured_timestamps >= min_true_ts]
		
	# # find index of max timestamp
	# if max_measured_ts < max_true_ts:
	# 	true_timestamps = np.array(true_timestamps)
	# 	if not by_measured:
	# 		true_idxs = true_idxs[true_timestamps <= max_measured_ts]
	# 	true_timestamps = true_timestamps[true_timestamps <= max_measured_ts]
	# else:
	# 	measured_timestamps = np.array(measured_timestamps)
	# 	if by_measured:
	# 		measured_idxs = measured_idxs[measured_timestamps <= max_true_ts]
	# 	measured_timestamps = measured_timestamps[measured_timestamps <= max_true_ts]
		
	if by_measured:
		diffs = np.abs(true_timestamps - measured_timestamps[0])
		start_idx = np.argmin(diffs)
		temp = true_timestamps[start_idx:]
		idxs = [start_idx]
		
		for i in range(1, len(measured_timestamps)):
			diffs = np.abs(temp - measured_timestamps[i])
			idx = np.argmin(diffs)
			assert (max(idxs) <= start_idx + idx) and idx >= 0, print(idxs)
			# Ensure the index doesn't exceed array bounds
			final_idx = min(start_idx + idx, len(true_timestamps) - 1)
			idxs.append(final_idx)
		
		return idxs, measured_idxs
	else:
		diffs = np.abs(measured_timestamps - true_timestamps[0])
		start_idx = np.argmin(diffs)
		temp = measured_timestamps[start_idx:]
		idxs = [start_idx]
		
		for i in range(1, len(true_timestamps)):
			diffs = np.abs(temp - true_timestamps[i])
			idx = np.argmin(diffs)
			assert (max(idxs) <= start_idx + idx) and idx >= 0, print(idxs)
			# Ensure the index doesn't exceed array bounds
			final_idx = min(start_idx + idx, len(measured_timestamps) - 1)
			idxs.append(final_idx)

	return true_idxs, idxs

def calculate_angles_between_vectors(v1, v2):
	return np.arccos(
		(
			v1[:, 0] * v2[:, 0] + 
			v1[:, 1] * v2[:, 1]
		) / 
		(np.sqrt(
			(
				v1[:, 0]**2 + 
				v1[:, 1]**2
			) * 
			(
				v2[:, 0]**2 + 
				v2[:, 1]**2
			)
		))
	) / np.pi * 180

def get_vector_directions(pivots, vectors):
	# Preserve the original azimuth convention used throughout the dataset:
	# positive cross-product means the target is to the left, which maps to a negative azimuth.
	dot_products = pivots[:, 0] * (-vectors[:, 1]) + pivots[:, 1] * vectors[:, 0]
	signs = np.ones((len(dot_products),))
	signs[dot_products < 0] = -1
	return signs

def calculate_azimuths(orientations, vectors):
	return get_vector_directions(orientations, vectors) * calculate_angles_between_vectors(orientations, vectors)

def azimuths_to_directions(azimuths, mode):
	directions = azimuths.copy()

	if mode == "front only":
		neg_idxs = azimuths < -10
		neutral_idxs = np.abs(azimuths) <= 10
		pos_idxs = azimuths > 10

		directions[neg_idxs] = -1
		#print("After neg update:", azimuths)
		directions[neutral_idxs] = 0
		#print("After neutral update:", azimuths)
		directions[pos_idxs] = 1
		#print("After positive update:", azimuths)
	elif mode == "all": 
		front_left_idxs = (azimuths <= -10) & (azimuths >= -90)
		front_right_idxs = (azimuths <= 90) & (azimuths >= 10)
		back_left_idxs = (azimuths < -90) & (azimuths >= -170)
		back_right_idxs = (azimuths <= 170) & (azimuths > 90)
		front_idxs = np.abs(azimuths) < 10
		back_idxs = (azimuths >= 170) | (azimuths <= -170)

		directions[front_left_idxs] = -1
		directions[front_idxs] = 0
		directions[front_right_idxs] = 1
		directions[back_left_idxs] = -2
		directions[back_right_idxs] = 2
		directions[back_idxs] = 3
	elif mode == "front back only":
		front_idxs = np.abs(azimuths) <= 90
		back_idxs = (azimuths > 90) | (azimuths < -90)
		directions[front_idxs] = 0
		directions[back_idxs] = 3
	elif mode == "all without front and back":
		front_left_idxs = (azimuths < 0) & (azimuths >= -90)
		front_right_idxs = (azimuths <= 90) & (azimuths > 0)
		back_left_idxs = (azimuths < -90) & (azimuths >= -180)
		back_right_idxs = (azimuths <= 180) & (azimuths > 90)
		directions[front_left_idxs] = -1
		directions[front_right_idxs] = 1
		directions[back_left_idxs] = -2
		directions[back_right_idxs] = 2
	return directions

def fuse_motion_and_uwb_azimuths(motion_azimuths, uwb_azimuths, method="uwb"):
	if method == "uwb":
		predicted_azimuths = uwb_azimuths.copy()
		# rely more on motion azimuths for azimuths over 90
		inconsistent_idxs = ((np.abs(uwb_azimuths) < 90) & (np.abs(motion_azimuths) > 90)) #| ((np.abs(uwb_azimuths) > 90) & (np.abs(motion_azimuths) < 90))
		predicted_azimuths[inconsistent_idxs] = motion_azimuths[inconsistent_idxs]
		# rely more on uwb if the azimuth is under 90   
		close_to_90 = (np.abs(uwb_azimuths) < 90)
		predicted_azimuths[close_to_90] = uwb_azimuths[close_to_90]
		# replace nil azimuths with motion azimuths
		nan_mask = np.isnan(predicted_azimuths)
		predicted_azimuths[nan_mask] = motion_azimuths[nan_mask]
	elif method == "mean":
		#print("motion azimuths:", motion_azimuths[:5], "uwb azimuths:", uwb_azimuths[:5])
		predicted_azimuths = azimuth_means(motion_azimuths, uwb_azimuths)
		nan_mask = np.isnan(predicted_azimuths)
		predicted_azimuths[nan_mask] = motion_azimuths[nan_mask]
		#print("motion azimuths:", motion_azimuths[:5], "uwb azimuths:", uwb_azimuths[:5], "predicted azimuths:", predicted_azimuths[:5])
	elif method == "motion":
		predicted_azimuths = motion_azimuths.copy()
		inconsistent_idxs = ((uwb_azimuths < 0) & (motion_azimuths > 0)) | ((uwb_azimuths > 0) & (motion_azimuths < 0))
		predicted_azimuths[inconsistent_idxs] *= -1
	else:
		raise Exception("Unknown fusion method:", method)
	return predicted_azimuths

def estimate_motion_azimuths(uwb_azimuths, delta_thetas, azimuth_signs, add):
	estimated_azimuths = uwb_azimuths.copy() 
	estimated_azimuths[add] = np.abs(np.abs(uwb_azimuths[add]) + delta_thetas[add]) * azimuth_signs[add] # delta_thetas[add]
	estimated_azimuths[~add] = np.abs(np.abs(uwb_azimuths[~add]) - delta_thetas[~add]) * azimuth_signs[~add] # delta_thetas[~add]
	#print("before fix:", estimated_azimuths)
	neg_idxs = (estimated_azimuths < 0) & (np.abs(estimated_azimuths) > 180)
	estimated_azimuths[np.abs(estimated_azimuths) > 180] = (360 - np.abs(estimated_azimuths[np.abs(estimated_azimuths) > 180]))
	estimated_azimuths[neg_idxs] *= -1
	#print("after fix:", estimated_azimuths)
	return estimated_azimuths
	
def calculate_distance_between_vectors(v1, v2):
	return np.sqrt(np.sum((v1 - v2)**2, axis=1))
