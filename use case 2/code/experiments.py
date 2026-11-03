import os
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np 
import math 
import random

try:
	from scipy.stats import wilcoxon
except ImportError:
	wilcoxon = None

device_id_to_shop_id = {
	"1F3A93B2": "S1",
	"E7F89769": "S2",
	"6CD63B91": "S3",
	"C6E37C6E": "S4",
	"1B34D23B": "S5",
	"ECC49AC2": "S6",
	"0EE362F4": "S7",
	"CF0B9842": "S8",
	"3D2DC339": "S9"
}
shop_id_to_device_id = {v: k for (k, v) in device_id_to_shop_id.items()}

def get_two_circles_intersections(x1,y1,r1, x2,y2,r2):
	centerdx = x1 - x2
	centerdy = y1 - y2
	R = math.sqrt(centerdx * centerdx + centerdy * centerdy)
	if not (abs(r1 - r2) <= R and (R <= r1 + r2)): 
		return []
	
	R2 = R*R
	R4 = R2*R2
	a = (r1*r1 - r2*r2) / (2 * R2)
	r2r2 = (r1*r1 - r2*r2)
	try:
		c = math.sqrt(2 * (r1*r1 + r2*r2) / R2 - (r2r2 * r2r2) / R4 - 1)
	except:
		print("shop 1:", (x1, y1), "shop 2:", (x2, y2))
		print(2, "*", f"({r1}*{r1} + {r2}*{r2}) / {R2} - ({r2r2} * {r2r2}) / {R4} - 1")
		return []

	fx = (x1+x2) / 2 + a * (x2 - x1)
	gx = c * (y2 - y1) / 2
	ix1 = fx + gx
	ix2 = fx - gx

	fy = (y1+y2) / 2 + a * (y2 - y1)
	gy = c * (x1 - x2) / 2
	iy1 = fy + gy
	iy2 = fy - gy

	return np.array([[ix1, iy1], [ix2, iy2]])

def rotate_vector(vector, angle):
	angle = angle / 180 * math.pi
	return np.array([vector[0] * math.cos(angle) - vector[1] * math.sin(angle), vector[0] * math.sin(angle) + vector[1] * math.cos(angle)])

def load_data(dir_name, adjust_angles=False):
	data = []
	adjusted_angle_errors = []
	angle_errors = []
	for f in os.listdir(dir_name):
		if f[-4:] != ".csv":
			continue 
		rows = {}
		parts = f[:-4].split("; ")
		d1 = d2 = int(parts[0][:-1])
		s1, a1 = parts[1].split(" ")
		s2, a2 = parts[2].split(" ")
		a1, a2 = int(a1), int(a2)
		if d1 != 10:
			if d1 == 5 and a1 == 0:
				d1 = d2 * np.cos(abs(a2) / 180 * np.pi)
				
			elif (a1 > 0 and a2 > 0) or (a1 < 0 and a2 < 0):
				d1 = d1 / np.cos(abs(a1) / 180 * np.pi)
				d2 = d2 / np.cos(abs(a2) / 180 * np.pi) 
		rows[s1] = {"actual distance": d1, "actual angle": a1}   
		rows[s2] = {"actual distance": d2, "actual angle": a2}   
		
		s1 = shop_id_to_device_id[s1]
		s2 = shop_id_to_device_id[s2]

		df = pd.read_csv(os.path.join(dir_name, f))
		df = df.iloc[1:, :]
		
		for device_id in df["DeviceID"].unique():
			device_df = df[df["DeviceID"] == device_id]
			shop_id = device_id_to_shop_id[device_id]
			rows[shop_id]["measured distance"] = device_df["Distance"].mean()
			device_df = device_df[device_df["Azimuth"] != "nil"]
			device_df["Azimuth"] = device_df["Azimuth"].astype(float)
			rows[shop_id]["measured angle"] = device_df["Azimuth"].mean()
			if adjust_angles:
				measured_angle = rows[shop_id]["measured angle"]
				angle_errors.append(abs(measured_angle - rows[shop_id]["actual angle"]))
				err = 0.002 * measured_angle**2 - 0.0071 * measured_angle + 6.1348
				rows[shop_id]["measured angle"] = measured_angle + err if measured_angle < 0 else measured_angle - err
				adjusted_angle_errors.append(abs(rows[shop_id]["measured angle"] - rows[shop_id]["actual angle"]))
				
		row = {}
		for i, shop in enumerate(rows):
			row[f"actual distance {i + 1}"] = rows[shop]["actual distance"] 
			row[f"actual angle {i + 1}"] = rows[shop]["actual angle"] 
			row[f"measured distance {i + 1}"] = rows[shop]["measured distance"] 
			row[f"measured angle {i + 1}"] = rows[shop]["measured angle"] 
			
		data.append(row)
	df = pd.DataFrame(data)
	df["distance error 1"] = np.abs(df["actual distance 1"] - df["measured distance 1"])
	df["distance error 2"] = np.abs(df["actual distance 2"] - df["measured distance 2"])
	df["angle error 1"] = np.abs(df["actual angle 1"] - df["measured angle 1"])
	df["angle error 2"] = np.abs(df["actual angle 2"] - df["measured angle 2"])
	df["angle 1 signs equal?"] = df["actual angle 1"] * df["measured angle 1"] > 0
	df["angle 2 signs equal?"] = df["actual angle 2"] * df["measured angle 2"] > 0
	print(df)
	df.to_csv("data.csv")

	if adjust_angles:
		fig, ax = plt.subplots()
		plot_histogram(angle_errors, "angle error", "CDF", "original", ax, color="blue", show=False)
		plot_histogram(adjusted_angle_errors, "angle error", "CDF", "adjusted", ax, color="green", show=False)
		plt.show()
	return df 

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

def calculate_vector_length(vector):
	return np.sqrt(vector[0]**2 + vector[1]**2)

def calc_angle_between_vectors(v1, v2):
	return np.arccos(
		(
			v1[0] * v2[0] + 
			v1[1] * v2[1]
		) / 
		np.sqrt(
			(
				v1[0]**2 + 
				v1[1]**2
			) * 
			(
				v2[0]**2 + 
				v2[1]**2
			)
		)
	) / np.pi * 180

def wilson_ci(successes, n, z=1.96):
	if n == 0:
		return (np.nan, np.nan)
	p = successes / n
	den = 1 + (z * z) / n
	center = (p + (z * z) / (2 * n)) / den
	half = (z / den) * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
	return (center - half, center + half)

def binom_one_sided_greater(k, n, p0=0.5):
	"""
	Exact one-sided binomial p-value: P[X >= k], X~Bin(n, p0).
	"""
	if n <= 0:
		return np.nan
	k = int(k)
	tail = 0.0
	for i in range(k, n + 1):
		tail += math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
	return min(max(tail, 0.0), 1.0)

def summarize_baseline_vs_dilateration(df):
	valid = df.dropna(subset=["positioning error", "random positioning error"]).copy()
	if len(valid) == 0:
		print("No valid intersection cases for baseline comparison")
		return

	n = len(valid)
	dil_mean_err = float(valid["positioning error"].mean())
	rand_mean_err = float(valid["random positioning error"].mean())
	dil_median_err = float(valid["positioning error"].median())
	rand_median_err = float(valid["random positioning error"].median())

	# Positive diff means random baseline has higher error than dilateration.
	diff = valid["random positioning error"].to_numpy() - valid["positioning error"].to_numpy()
	mean_improvement = float(np.mean(diff))
	median_improvement = float(np.median(diff))

	# Exact paired sign test (one-sided): H0 median(diff)=0 vs H1 median(diff)>0.
	n_pos = int(np.sum(diff > 0))
	n_neg = int(np.sum(diff < 0))
	n_ties = int(np.sum(diff == 0))
	n_eff = n_pos + n_neg
	p_pos = binom_one_sided_greater(n_pos, n_eff, p0=0.5) if n_eff > 0 else np.nan

	# Wilcoxon signed-rank sensitivity test using Pratt zero handling.
	w_stat = np.nan
	w_p = np.nan
	if wilcoxon is not None:
		try:
			w_res = wilcoxon(diff, zero_method="pratt", alternative="greater", method="approx")
			w_stat = float(w_res.statistic)
			w_p = float(w_res.pvalue)
		except Exception:
			pass

	# Secondary endpoint: intersection-level accuracy.
	dil_correct = int(valid["correct intersection?"].sum())
	rand_correct = int(valid["random correct intersection?"].sum())
	dil_acc = dil_correct / n
	rand_acc = rand_correct / n
	dil_ci = wilson_ci(dil_correct, n)
	rand_ci = wilson_ci(rand_correct, n)

	# One-sided exact binomial test against chance for dilateration accuracy.
	p_vs_chance = binom_one_sided_greater(dil_correct, n, p0=0.5)

	print("\n===== Dilateration vs Random-Intersection Baseline =====")
	print(f"Valid two-intersection cases: {n}")
	print("\nPrimary endpoint: paired positioning error")
	print(f"Mean positioning error (dilateration): {dil_mean_err:.4f} m")
	print(f"Mean positioning error (random): {rand_mean_err:.4f} m")
	print(f"Median positioning error (dilateration): {dil_median_err:.4f} m")
	print(f"Median positioning error (random): {rand_median_err:.4f} m")
	print(f"Mean improvement (random - dilateration): {mean_improvement:.4f} m")
	print(f"Median improvement (random - dilateration): {median_improvement:.4f} m")
	print(
		f"Paired sign test (one-sided, diff>0): n_pos={n_pos}, n_neg={n_neg}, "
		f"ties={n_ties}, n_eff={n_eff}, p-value={p_pos:.6g}"
	)
	if wilcoxon is None:
		print("Wilcoxon signed-rank test unavailable: scipy is not installed")
	elif not np.isfinite(w_p):
		print(
			"Wilcoxon signed-rank test returned a non-finite p-value. "
			"Require scipy>=1.11 with Pratt zero handling and method='approx'."
		)
	else:
		print(
			"Wilcoxon signed-rank (Pratt zeros, one-sided diff>0, method=approx): "
			f"W={w_stat:.4f}, p-value={w_p:.6g}"
		)

	print("\nSecondary endpoint: intersection selection accuracy")
	print(
		f"Dilateration accuracy: {dil_acc:.4f} "
		f"(95% Wilson CI [{dil_ci[0]:.4f}, {dil_ci[1]:.4f}])"
	)
	print(
		f"Random baseline accuracy: {rand_acc:.4f} "
		f"(95% Wilson CI [{rand_ci[0]:.4f}, {rand_ci[1]:.4f}])"
	)
	print(f"Accuracy gain (dilateration - random): {dil_acc - rand_acc:.4f}")
	print(
		"Exact binomial one-sided p-value "
		f"for dilateration > chance (p0=0.5): {p_vs_chance:.6g}"
	)

def display_pos_error(data, title):
	x = data["actual angle 1"]
	y = data["actual angle 2"]
	z = data["positioning error"]
	fig = plt.figure()

	# syntax for 3-D plotting
	ax = plt.axes(projection='3d')

	# syntax for plotting
	ax.scatter3D(x, y, z, cmap='cool')

	ax.zaxis.set_rotate_label(False) 
	ax.set_ylabel("${\\theta^\prime_1}$ (${\degree}$)", rotation=0, labelpad=20)
	ax.set_xlabel("${\\theta^\prime_2}$ (${\degree}$)", rotation=0, labelpad=20)
	ax.set_zlabel("Positioning error (m)", rotation=90)
	plt.title(title)
	plt.show()

def get_positions(df, verbose=False):
	errors = []
	errors2 = []
	random_errors = []
	random_correct_flags = []
	estimated_oris11 = []
	estimated_oris12 = []
	estimated_oris21 = []
	estimated_oris22 = []
	intersection_ids = []
	c = 0
	for i, r in df.iterrows():
		ad1, aa1 = r["actual distance 1"], (90 - r["actual angle 1"]) / 180 * np.pi
		ad2, aa2 = r["actual distance 2"], (90 - r["actual angle 2"]) / 180 * np.pi
		if verbose:
			print("CASE: d1 =", ad1, "a1 =", aa1 / np.pi * 180, "d2 =", ad2, "a2 =", aa2 / np.pi * 180)
		shop1_coords = np.array([ad1 * np.cos(aa1), ad1 * np.sin(aa1)])
		shop2_coords = np.array([ad2 * np.cos(aa2), ad2 * np.sin(aa2)])
		md1, ma1 = r["measured distance 1"], (90 - r["measured angle 1"]) 
		md2, ma2 = r["measured distance 2"], (90 - r["measured angle 2"]) #/ 180 * np.pi
		if verbose:
			print("measured distance 1:", md1, "measured distance 2:", md2, "measured angle 1:", ma1, "measured angle 2:", ma2)
		
		intersections = get_two_circles_intersections(shop1_coords[0], shop1_coords[1], md1, shop2_coords[0], shop2_coords[1], md2)
				
		if len(intersections) == 0:
			print("!!!!!!! -------- NO INTERSECTIONS ------- !!!!!!!!")
			print("CASE: d1 =", ad1, "a1 =", aa1 / np.pi * 180, "d2 =", ad2, "a2 =", aa2 / np.pi * 180, "shop 1:", shop1_coords, "shop 2:", shop2_coords)
			errors.append(np.nan)
			errors2.append(np.nan)
			random_errors.append(np.nan)
			random_correct_flags.append(np.nan)
			estimated_oris11.append(np.nan)
			estimated_oris12.append(np.nan)
			estimated_oris21.append(np.nan)
			estimated_oris22.append(np.nan)
			intersection_ids.append(np.nan)
			c += 1
			continue

		dist_between_inter = calculate_vector_length(intersections[0] - intersections[1])
		distance_to_i1 = calculate_vector_length(intersections[0])
		distance_to_i2 = calculate_vector_length(intersections[1])
		if verbose:
			print("\tintersection 1:", intersections[0], "intersection 2:", intersections[1], "ground truth: [0, 0]", "distance between intersections:", dist_between_inter)
			print("\tdistance to intersection 1:", distance_to_i1, "distance to intersection 2:", distance_to_i2)
			print("\tCorrect intersection:", 1 if distance_to_i1 < distance_to_i2 else 2)
		i1s1 = shop1_coords - intersections[0]
		i2s1 = shop1_coords - intersections[1]
		i1s2 = shop2_coords - intersections[0]
		i2s2 = shop2_coords - intersections[1]

		i1_ori1 = rotate_vector(i1s1, -ma1)
		i1_ori2 = rotate_vector(i1s2, -ma2)
		i2_ori1 = rotate_vector(i2s1, -ma1)
		i2_ori2 = rotate_vector(i2s2, -ma2)
		estimated_oris11.append(calculate_vector_theta(i1_ori1))
		estimated_oris12.append(calculate_vector_theta(i1_ori2))
		estimated_oris21.append(calculate_vector_theta(i2_ori1))
		estimated_oris22.append(calculate_vector_theta(i2_ori2))
		
		i1_error = calculate_vector_length(i1_ori1 - i1_ori2)
		i2_error = calculate_vector_length(i2_ori1 - i2_ori2)

		if verbose:
			print("i1s1 theta:", calculate_vector_theta(i1s1), "actual theta to s1:", aa1 / np.pi * 180)
			print("i1s2 theta:", calculate_vector_theta(i1s2), "actual theta to s2:", aa2 / np.pi * 180)
			print("i2s1 theta:", calculate_vector_theta(i2s1), "actual theta to s1:", aa1 / np.pi * 180)
			print("i2s2 theta:", calculate_vector_theta(i2s2), "actual theta to s2:", aa2 / np.pi * 180)
			print("\ti1s1 orientation theta:", calculate_vector_theta(i1_ori1), "actual orientation theta:", 90)
			print("\ti1s2 orientation theta:", calculate_vector_theta(i1_ori2), "actual orientation theta:", 90)
			print("\ti2s1 orientation theta:", calculate_vector_theta(i2_ori1), "actual orientation theta:", 90)
			print("\ti2s2 orientation theta:", calculate_vector_theta(i2_ori2), "actual orientation theta:", 90)
			
			print("\t\tAngle between orientations for intersection 1:", i1_error)
			print("\t\tAngle between orientations for intersection 2:", i2_error)
		
		if not np.isnan(i1_error) and not np.isnan(i2_error):
			# Ground-truth correct candidate is the one closer to the origin [0, 0].
			correct_intersection_idx = 1 if distance_to_i1 < distance_to_i2 else 2

			if i1_error > i2_error:
				if verbose:
					print(f"\t\t\tRESULT: intersection 2")
				errors.append(distance_to_i2)
				errors2.append(distance_to_i1)
				intersection_ids.append(2)
			else:
				if verbose:
					print(f"\t\t\tRESULT: intersection 1")
				errors.append(distance_to_i1)
				errors2.append(distance_to_i2)
				intersection_ids.append(1)

			# Random-intersection baseline on the same case.
			random_choice = random.choice([1, 2])
			random_err = distance_to_i1 if random_choice == 1 else distance_to_i2
			random_errors.append(random_err)
			random_correct_flags.append(random_choice == correct_intersection_idx)
			
			if errors[-1] > errors2[-1]:
				print("WRONG INTERSECTION:", "S1:", shop1_coords, "S2:", shop2_coords, "intersection 1:", intersections[0], "intersection 2:", intersections[1])
				print("i1s1:", i1s1, "i1s2:", i1s2, "i2s1:", i2s1, "i2s2:", i2s2)
				print("i1_ori1:", i1_ori1, "i1_ori2:", i1_ori2, "i2_ori1:", i2_ori1, "i2_ori2:", i2_ori2)
				print("ma1:", ma1, "ma2:", ma2)
		else:
			if np.isnan(i1_error):
				print("intersection 1: couldn't calculate the angle between orientations")
			elif np.isnan(i2_error):
				print("intersection 2: couldn't calculate the angle between orientations")

			errors.append(np.nan)
			errors2.append(np.nan)
			random_errors.append(np.nan)
			random_correct_flags.append(np.nan)
		
	df["positioning error"] = errors
	df["positioning error 2"] = errors2
	df["random positioning error"] = random_errors
	
	df["correct intersection?"] = df["positioning error"] < df["positioning error 2"]
	df["random correct intersection?"] = random_correct_flags
	df.to_csv("data.csv")

	print("number of invalid cases:", c)

def plot_histogram(data, xlabel, ylabel, label, ax, line=False, color=None, show=True):
	if show:
		fig, ax = plt.subplots()
	if line:
		count, bins_count = np.histogram(data, bins=5) 
		pdf = count / sum(count) 
		cdf = np.cumsum(pdf) 
		ax.plot(bins_count[1:], cdf, label=label)
	else:
		ax.hist(data, label=label, cumulative=True, weights=np.ones_like(data) * 100 / len(data), alpha=.3, linewidth=1, edgecolor="black", bins=10, color=color if color else "black")
    
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.legend()
	if show:
		plt.show()

df = load_data("data")
print("original length:", df.shape[0])
random.seed(42)
get_positions(df)

summarize_baseline_vs_dilateration(df)

df = df.dropna()

df["max distance"] = np.max(df[["actual distance 2", "actual distance 1"]], axis=1)

indices = set()

fig, ax = plt.subplots()
separate = True
line = False
for color, start, end in [("blue", 1, 2), ("gray", 5, 8), ("green", 10, 10)]:  # [("blue", 1, 6), ("gray", 6, 10)]
	temp = df[(df["max distance"] >= start) & (df["max distance"] <= end)]#.reset_index(drop=True)
	print(len(temp["positioning error"]))#, temp.index)
	indices = indices.union(set(temp.index))
	label = f"{start} - {end}" if start != end else f"{start}"
	max_label = "Maximum distance to sensors:" if start != end else "Distance to sensors:"
	print(temp["positioning error"].describe())
	
print(len(df.index), len(indices), set(df.index) - indices)
if not separate:
	plt.show()