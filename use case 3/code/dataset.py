import collections
import time
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import operator 
import datetime as dt
from matplotlib.ticker import PercentFormatter
from colour import Color
import util 
import constants
import seaborn as sns 
from matplotlib.ticker import PercentFormatter
import uwb_imu_fusion as te 
import random
import orientation_fusion 
BLUE_PHONE_OFFSETS = (-.19, -.155)
SHOP_OFFSETS = (-0.09, .15)
S7_OFFSETS = (.07, -.365)
load_raw_iphone_data = True 
use_cached_trilateration = False


def _estimate_object_nbytes(obj):
	"""Recursively estimate memory usage in bytes for numpy-heavy containers."""
	if obj is None:
		return 0
	if isinstance(obj, np.ndarray):
		return int(obj.nbytes)
	if isinstance(obj, pd.DataFrame):
		return int(obj.memory_usage(deep=True).sum())
	if isinstance(obj, dict):
		return int(sum(_estimate_object_nbytes(v) for v in obj.values()))
	if isinstance(obj, (list, tuple)):
		return int(sum(_estimate_object_nbytes(v) for v in obj))
	return 0


def _estimate_method_memory_mb(method_path):
	"""Estimate in-memory footprint (MB) for one method trajectory structure."""
	if not isinstance(method_path, dict):
		return np.nan
	bytes_used = 0
	bytes_used += _estimate_object_nbytes(method_path.get("positions"))
	bytes_used += _estimate_object_nbytes(method_path.get("orientations"))
	bytes_used += _estimate_object_nbytes(method_path.get("timestamps"))
	bytes_used += _estimate_object_nbytes(method_path.get("azimuths"))
	bytes_used += _estimate_object_nbytes(method_path.get("true azimuths"))
	return bytes_used / (1024.0 * 1024.0)

def load_data(
	shops_filename,
	consolidate_by_timestamp,
	lidar_data_dir, 
	lidar_orientations,
	iphone_data_dir,
	drop_risky_measured_azimuths, 
	drop_risky_true_azimuths,
	drop_risky_elevations, 
	drop_risky_true_distances,
	check_vectors_for_sanity,
	check_abrupt_angle_changes,
	by_measured,
	start=None,
	finish=None,
	cache_paths_on_disk=True
):
	shop_locations, true_paths, lidar_first_timestamps = load_lidar_data(
		lidar_data_dir, 
		lidar_orientations=lidar_orientations
	)

	save_shops(shop_locations, shops_filename)
	iphone_paths, iphone_first_timestamps, imu_data = load_iphone_data(
		iphone_data_dir, 
		shop_locations, 
		true_paths, 
		by_measured,
		drop_risky_measured_azimuths=drop_risky_measured_azimuths, 
		drop_risky_true_azimuths=drop_risky_true_azimuths,
		drop_risky_elevations=drop_risky_elevations, 
		drop_risky_true_distances=drop_risky_true_distances,
		check_vectors_for_sanity=check_vectors_for_sanity,
		check_abrupt_angle_changes=check_abrupt_angle_changes,
		start=start,
		finish=finish,
		cache_paths_on_disk=cache_paths_on_disk
	)

	#consolidate_shop_measurements_by_timestamp(iphone_paths, "unfiltered")
	for data_type in consolidate_by_timestamp:
		consolidate_shop_measurements_by_timestamp(iphone_paths, data_type)
	#deduce_azimuths(shop_locations, iphone_paths, time_interval)
	#consolidate_shop_measurements_by_timestamp(iphone_paths, "deduced")

	return shop_locations, true_paths, iphone_paths, lidar_first_timestamps, iphone_first_timestamps, imu_data

def azimuths2float(azimuths):	
	if azimuths.dtype != np.float32 and azimuths.dtype != np.float64 and azimuths.dtype != float:
		#print("type:", azimuths.dtype)
		azimuths = azimuths.astype(str).str.replace("nil", f"{util.DUMMY}")
		#azimuths = azimuths.str.replace("nil", f"{util.DUMMY}")

	return azimuths.astype(np.float32)

def quaternion_to_orientation(q):
	yaw = np.arctan2(2.0*(q[1]*q[2] + q[3]*q[0]), q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2])
	pitch = np.arcsin(-2.0*(q[0]*q[2] - q[3]*q[1]))
	roll = np.arctan2(2.0*(q[0]*q[1] + q[3]*q[2]), q[3]*q[3] + q[0]*q[0] - q[1]*q[1] - q[2]*q[2])
	return roll

def timestamp_to_number(timestamp):
	return dt.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f').timestamp()

def get_perpendicular_vector(vector):
	return np.array([-vector[1], vector[0]])

def get_phone_position_from_lidar(lidar_position, lidar_orientation):
	orientation_vector = np.array([np.cos(lidar_orientation), np.sin(lidar_orientation)]) 
	orientation_vector /= util.calculate_vector_length(orientation_vector)
	perp = get_perpendicular_vector(orientation_vector) 
	perp /= util.calculate_vector_length(perp)
	left = lidar_position - .1 * perp
	right = lidar_position + .1 * perp 
	mid = (left + right) / 2
	res = mid - orientation_vector * constants.DISTANCE_FROM_LIDAR_TO_PHONE
	return res

def offset_position(position, orientation_vector, offset_x, offset_y):
	para_vector = util.rotate_vector(orientation_vector, -90)
	#print("positions shape:", position.shape, "para vector shape:", para_vector.shape, "orientation vec shape:", orientation_vector.shape, "offsets:", offset_x, offset_y)
	position = position + offset_y * orientation_vector / util.calculate_vector_lengths(orientation_vector).reshape(-1, 1)
	position = position + offset_x * para_vector / util.calculate_vector_lengths(para_vector).reshape(-1, 1)
	return position

def load_lidar_data(
	dir_name, 
	lidar_orientations=True
):
	shops = {}
	paths = {}
	edge_timestamps = {}

	for batch_name in os.listdir(dir_name): 
		if not batch_name.startswith("batch"):
			continue
		batch_id = int(batch_name[-1])
		path_to_dir = os.path.join(dir_name, batch_name)
		for filename in sorted(os.listdir(path_to_dir)):
			if filename[-4:] != ".log":
				continue
			positions = []
			orientations = []
			timestamps = []
			lc = 0
			is_shop = filename[0] == "S"
			is_path = filename[0] == "P"
			path_id = filename[:-4] 
			if batch_id == 1 and path_id == "S2":
				continue
			
			if not is_shop and not is_path:
				continue
			
			# if os.path.exists(f"ground-truth/{path_id}.csv") or os.path.exists(f"ground-truth/{path_id}-{batch_id}.csv"):
			# 	if os.path.exists(f"ground-truth/{path_id}.csv"):
			# 		ground_truth_df = pd.read_csv(f"ground-truth/{path_id}.csv") 
			# 	else:
			# 		ground_truth_df = pd.read_csv(f"ground-truth/{path_id}-{batch_id}.csv")
				
			# 	positions = ground_truth_df.loc[:, ["PositionX", "PositionY"]].values
			# 	orientations = ground_truth_df["Orientation"].values
			# 	timestamps = ground_truth_df["Timestamp"].values 
			# else:
			with open(os.path.join(path_to_dir, filename), "r") as f:
				init_ts = None
				for line in f.readlines()[::-1] if is_shop else f.readlines():
					if "Sent" in line:
						line = line.split(", \"")
						stop = False

						for i, data in enumerate(line):
							data = data.split(": ")
							if i == 0: # timestamp
								timestamp = data[1][1:-1]
								timestamp = timestamp_to_number(timestamp) #- timestamp_to_number(constants.REF_DATE)
								
								if is_path:
									if not init_ts:
										init_ts = timestamp
										edge_timestamps[path_id] = [init_ts]
									
									if timestamp > init_ts + constants.MEASUREMENT_DURATION:
										stop = True 
										break 
									if timestamp < init_ts + constants.MEASUREMENT_START_OFFSET:
										break 
								
								timestamps.append(timestamp)
							elif i == 1: # position
								position = list(map(float, data[1][1:-1].split(", ")[:2]))
								positions.append(position)
							elif i == 2: # orientation
								if lidar_orientations:
									# orientation = list(map(float, data[1][1:-4].split(", ")))
									# orientations.append(quaternion_to_orientation(orientation))
									orientation = float(data[1][:-3])#quaternion_to_orientation(list(map(float, data[1][1:-4].split(", "))))
									orientations.append(orientation)
								elif len(positions) > constants.ORIENTATION_STEPS:
									orientation = util.get_orientation(positions, len(positions) - 1, constants.ORIENTATION_STEPS)
									orientations.append(orientation)
								
								# if len(orientations) > 0:
								# 	positions[-1] = get_phone_position_from_lidar(positions[-1], orientations[-1])
								
						lc += 1
						if (is_shop and lc == constants.NUM_LIDAR_LINES) or (is_path and stop):
							break 
				
				# if is_path:	
				# 	edge_timestamps[path_id].append(timestamp)
			
			positions, orientations, timestamps = np.array(positions), np.array(orientations), np.array(timestamps) # add 90 degrees to all lidar orientations if the phone was facing to the front because the Velodyne text was facing to the right 
		
			ground_truth_df = pd.DataFrame({
				"Timestamp": timestamps,
				"PositionX": positions[:, 0],
				"PositionY": positions[:, 1],
				"Orientation": orientations
			})
			ground_truth_df.to_csv(f"ground-truth/{(path_id if not is_shop else f'{path_id}-{batch_id}')}.csv", index=False)

			
			if is_shop: # rotate orientations for blue phone and shops
				orientations = (orientations + 180) % 360
			
			orientations = util.degrees2radians(orientations)
					
			if is_shop:
				position = positions.mean(axis=0)
				orientation = np.mean(orientations) 
				orientation_vector = util.theta2orientation_vector(orientation) #* -1
				position = offset_position(position, orientation_vector, SHOP_OFFSETS[0], SHOP_OFFSETS[1])
				if path_id == "S7":
					position = offset_position(position, orientation_vector, S7_OFFSETS[0], S7_OFFSETS[1])
					
				shops[f"{path_id}-{batch_id}"] = position.reshape(-1,)
			elif is_path:
				if not lidar_orientations:
					paths[path_id] = {
						"positions": positions[constants.ORIENTATION_STEPS:], 
						"orientations": orientations, 
						"timestamps": timestamps[constants.ORIENTATION_STEPS:]
					}
				else:
					paths[path_id] = {
						"positions": positions, 
						"orientations": orientations, 
						"timestamps": timestamps
					}

	return shops, paths, edge_timestamps

def calculate_true_speeds(timestamps, positions):
	idxs = np.argsort(timestamps)
	timestamps = timestamps[idxs]
	positions = positions[idxs]
	distances, delta_ts = [], []

	max_dist = -1
	max_dist_idx = -1
	max_delta_t = -1
	
	for i in range(1, len(positions)):
		delta_t = timestamps[i] - timestamps[i - 1]
		distance = util.calculate_vector_length(positions[i] - positions[i - 1])
		distances.append(distance)
		delta_ts.append(delta_t)
		if distance > max_dist:
			max_dist = distance 
			max_delta_t = delta_t
			max_dist_idx = i
	print("max dist:", max_dist, "delta_t of max_dist:", max_delta_t, "speed:", max_dist / max_delta_t, "max dist idx:", max_dist_idx, "out of", len(timestamps))
	
	speeds = np.array(distances) / np.array(delta_ts)
	# print("Min distance:", np.min(distances))
	# print("Max distance:", np.max(distances))
	# print("Min delta_t:", np.min(delta_ts))
	# print("Max delta_t:", np.max(delta_ts))
	# print("Min speed:", np.min(speeds))
	# print("Max speed:", np.max(speeds))
	df = pd.DataFrame([speeds, distances, delta_ts])
	df = df.T 
	df.columns=["Speed", "Distance", "Delta t"]
	#df.to_csv("speeds.csv", index=False)
	#util.plot_histogram(speeds, "Speed (m/s)", "PDF", None, None, line=False, color=None, show=True, marker=None, cumul=False, bins=50)
	return df 

def average_df_into_intervals(df, time_interval):
	def helper(df):
		data = {col: [] for col in df.columns}
		for i, r in df.iterrows():
			for col in data:
				if r[col] != "nil" and not np.isnan(r[col]):
					data[col].append(r[col])
	df = df.sort_values("Timestamp")

	i = 0
	while i < len(df):
		temp_dist = {}
		start_idx = curr_idx = i
		record = []
		while curr_idx < len(df) and (df.loc[curr_idx, "Timestamp"] - df.loc[start_idx, "Timestamp"]) < time_interval:
			curr_idx += 1
		i += 1

def load_iphone_data(
	dir_name, 
	shop_locations, 
	true_paths, 
	by_measured,
	drop_risky_measured_azimuths=True, 
	drop_risky_true_azimuths=True, 
	drop_risky_elevations=True, 
	drop_risky_true_distances=True,
	check_vectors_for_sanity=True, 
	check_abrupt_angle_changes=True,
	start=None,
	finish=None,
	cache_paths_on_disk=True,
	num_batches=2
):
	
	if not start:
		start = 1
	if not finish:
		finish = len(true_paths)

	iphone_paths = {}
	closest = {}
	edge_timestamps = {}
	imu_data = {}
	total_duration = 0 
	for i in range(start, finish + 1):
		for batch_id in range(1, num_batches + 1):
			batch_name = f"batch{batch_id}"
			path_id = f"P{i}"
			uwb_filename = f"B{i}_UWB.csv"
			imu_filename = f"B{i}_IMU.csv"
			uwb_file_path = os.path.join(dir_name, batch_name, uwb_filename)
			imu_file_path = os.path.join(dir_name, batch_name, imu_filename)
			if not os.path.exists(uwb_file_path) or path_id not in true_paths:
				continue

			
			iphone_paths[path_id] = {}
			closest[path_id] = {}
			
			if not load_raw_iphone_data:
				continue
			
			print("loading data:", path_id)
			df = pd.read_csv(uwb_file_path).sort_values("Timestamp")
			
			# remove the first record that contains the start time and the trailing records
			iphone_start_time = df.loc[0, "Timestamp"] + constants.MEASUREMENT_START_OFFSET
			iphone_end_time = df.loc[df.shape[0] - 1, "Timestamp"] - constants.MEASUREMENT_END_OFFSET
			lidar_start_time = true_paths[path_id]["timestamps"][0]
			lidar_end_time = true_paths[path_id]["timestamps"][-1]
			start_time = max(iphone_start_time, lidar_start_time)
			end_time = min(iphone_end_time, lidar_end_time)
			total_duration += (end_time - start_time)

			df_imu = pd.read_csv(imu_file_path).sort_values("Timestamp")
			df_imu = df_imu[df_imu["Timestamp"] >= start_time]
			df_imu = df_imu[df_imu["Timestamp"] <= end_time]
			df_imu = df_imu.dropna().reset_index(drop=True)
			imu_data[path_id] = df_imu


			print("Lidar start time:", lidar_start_time, "iphone start time:", iphone_start_time, "difference:", iphone_start_time - lidar_start_time)
			
			df = df.iloc[1:, :]
			df = df[df["Timestamp"] >= start_time]
			df = df[df["Timestamp"] <= end_time]
			df = df.dropna().reset_index()
			if not df.empty:
				print("iPhone start time:", start_time, "iPhone end time:", end_time, "measurement duration:", end_time - start_time)
			else:
				print(path_id, "df is empty")
			
			# sort out the lidar timestamps too
			temp_ts = true_paths[path_id]["timestamps"]
			ts_idxs = (temp_ts >= start_time) & (temp_ts <= end_time)
			for k in true_paths[path_id]:
				true_paths[path_id][k] = true_paths[path_id][k][ts_idxs]

			true_path = true_paths[path_id]
			true_timestamps = true_path["timestamps"]
			true_orientations = true_path["orientations"]

			print("lidar start time:", true_paths[path_id]["timestamps"][0], "lidar end time:", true_paths[path_id]["timestamps"][-1], "measurement duration:", true_paths[path_id]["timestamps"][-1] - true_paths[path_id]["timestamps"][0])

			for device_id in df["DeviceID"].unique():
				shop_id = f"S{device_id.split(' ')[1]}-{batch_id}"
				if shop_id == "S2-1":
					continue # someone moved shop 2 during experiments 
				unfiltered_path_shop_csv_name = f"path_files/unfiltered_{path_id}_{shop_id}.csv"
				unfiltered_df = None
				if cache_paths_on_disk and os.path.exists(unfiltered_path_shop_csv_name):
					unfiltered_df = pd.read_csv(unfiltered_path_shop_csv_name)
				else:
					device_df = df[(df["DeviceID"] == device_id)].reset_index(drop=True)
					
					# calculate the vector to this anchor
					true_tas = shop_locations[shop_id] - true_path["positions"]
					
					# match timestamps between the ground truth and measurements
					measured_timestamps = device_df["Timestamp"].values
					if by_measured:
						true_idxs, measured_idxs = util.find_closest_ground_truth_indices(true_timestamps, measured_timestamps, True)
						
					else:
						true_idxs, measured_idxs = util.find_closest_ground_truth_indices(true_timestamps, measured_timestamps, False)
						
					device_df = device_df.iloc[measured_idxs, :].reset_index(drop=True)
					
					time_aligned_orientations = true_orientations[true_idxs] 
					time_aligned_tas = true_tas[true_idxs]
					time_aligned_timestamps = true_timestamps[true_idxs]
					time_aligned_positions = true_path["positions"][true_idxs]

					device_df["Azimuth"] = azimuths2float(device_df["Azimuth"])
					device_df["Actual Distance"] = util.calculate_vector_lengths(time_aligned_tas)
					device_df["Actual Azimuth"] = util.calculate_azimuths(util.theta2orientation_vector(time_aligned_orientations), time_aligned_tas)
					device_df["Actual Timestamp"] = time_aligned_timestamps
					device_df["Actual Orientation"] = time_aligned_orientations
					device_df["Actual X"] = time_aligned_positions[:, 0]
					device_df["Actual Y"] = time_aligned_positions[:, 1]
					
					# calculate bearings
					device_df["DeviceBearing"] = util.theta_to_bearing(util.radians2degrees(time_aligned_orientations))
					

					# calculate unfiltered positions (with nil angles)
					unfiltered_df = iphone_data_to_coordinates(
						device_df,
						shop_locations[shop_id], 
						check_vectors_for_sanity=False,
						check_abrupt_angle_changes=False,
						drop_risky_measured_azimuths=False,
						drop_risky_true_azimuths=False,
						drop_risky_elevations=False,
						drop_risky_true_distances=False,
						filter_invalid_angles=False
					)
					unfiltered_df = unfiltered_df.reset_index(drop=True)

					unfiltered_df.to_csv(unfiltered_path_shop_csv_name, index=False)
				
				if unfiltered_df is not None:
					unfiltered_df["Path"] = path_id
					unfiltered_df["Shop"] = shop_id

					iphone_paths[path_id][shop_id] = {
						"unfiltered": {
							"lidar start time": lidar_start_time,
							"time-aligned actual timestamps": unfiltered_df["Actual Timestamp"].values,
							"measured distances": unfiltered_df.loc[:, "Distance"].values, 
							"measured azimuths": unfiltered_df.loc[:, "Azimuth"].values,
							"bearings": unfiltered_df.loc[:, "DeviceBearing"].values,
							"time-aligned actual orientations": (unfiltered_df["Actual Orientation"].values),
							"timestamps": unfiltered_df["Timestamp"].values,
							"time-aligned actual distances": unfiltered_df["Actual Distance"].values,
							"time-aligned actual azimuths": unfiltered_df["Actual Azimuth"].values,
							"time-aligned actual positions": unfiltered_df.loc[:, ["Actual X", "Actual Y"]].values,
							"positions": unfiltered_df.loc[:, ["X", "Y"]].values,
							"orientations": util.degrees2radians(unfiltered_df["Actual Orientation"].values)
						}
					}

	print("Total duration:", total_duration)
	return iphone_paths, edge_timestamps, imu_data

def consolidate_shop_measurements_by_timestamp(iphone_paths, data_type):
	consolidated = {}
	for path_id in iphone_paths:
		# average all devices
		filtered_data = {}
		
		for shop_id in iphone_paths[path_id]:
			if shop_id[0] != "S":
				continue
			if data_type not in iphone_paths[path_id][shop_id]:
				continue
			
			if path_id not in consolidated:
				consolidated[path_id] = {}

			if data_type not in consolidated[path_id]:
				consolidated[path_id][data_type] = []

			curr_shop = iphone_paths[path_id][shop_id][data_type]
			fd = []
			for i, ts in enumerate(curr_shop["timestamps"]):
				fd.append({
					"timestamp": ts,
					"actual timestamp": curr_shop["time-aligned actual timestamps"][i],
					"position": curr_shop["positions"][i] if "positions" in curr_shop else None, 
					"orientation": curr_shop["orientations"][i] if "orientations" in curr_shop else None,
					"distance": curr_shop["measured distances"][i],
					"actual azimuth": curr_shop["time-aligned actual azimuths"][i],
					"azimuth": curr_shop["measured azimuths"][i],
					"actual orientation": curr_shop["time-aligned actual orientations"][i],
					"actual position": curr_shop["time-aligned actual positions"][i],
					"actual distance": curr_shop["time-aligned actual distances"][i],
					"shop": shop_id,
					"index": i
				})
			consolidated[path_id][data_type] += fd
		
		# make sure the timestamps are sorted for each path 
		if path_id in consolidated and data_type in consolidated[path_id]:
			consolidated[path_id][data_type] = sorted(consolidated[path_id][data_type], key=lambda x: x["timestamp"])
			
	for path_id in consolidated:
		if "consolidated" not in iphone_paths[path_id]:
			iphone_paths[path_id]["consolidated"] = {}
		for data_type in consolidated[path_id]:
			iphone_paths[path_id]["consolidated"][data_type] = consolidated[path_id][data_type]
			
def average_iphone_positions(
	iphone_paths,
	shop_locations,
	time_interval,
	max_num_anchors,
	methods,
	imu_data=None,
	dilateration_enabled=True,
	use_cached_outputs=True,
):
	anchor_conn_stats = {}
	pas = full_pas = 0
	for path_id in iphone_paths:
			if "trilateration" in methods or "ekf" in methods or "ukf" in methods:
				# trilateration
				trilateration_positions, trilateration_orientations, trilateration_timestamps, uwb_azimuths, true_uwb_azimuths, measurement_type = [], [None], [], collections.defaultdict(list), collections.defaultdict(list), []
				tr_rate = num_windows = dr_rate = 0
				
				# see if there is cached data 	
				if use_cached_outputs and os.path.exists(f"path_files/trilateration_{path_id}_MaxAPs={max_num_anchors}_TI={time_interval}.csv"):
					trilateration_df = pd.read_csv(f"path_files/trilateration_{path_id}_MaxAPs={max_num_anchors}_TI={time_interval}.csv")
					trilateration_positions = trilateration_df[["PositionX", "PositionY"]].values
					trilateration_orientations = trilateration_df["Orientation"].values
					trilateration_timestamps = trilateration_df["Timestamp"].values
					if "measurement_type" in trilateration_df.columns:
						measurement_type = trilateration_df["measurement_type"].fillna("trilateration").values.tolist()
					else:
						measurement_type = ["trilateration"] * len(trilateration_df)
					if "InferenceTimePerUpdateMs" in trilateration_df.columns:
						inference_time = float(np.mean(trilateration_df["InferenceTimePerUpdateMs"].values))
					else:
						# Backward compatibility with old cached files.
						inference_time = float(np.mean(trilateration_df["InferenceTime"].values))
					uwb_azimuths = {}
					true_uwb_azimuths = {}
					for shop_id in shop_locations:
						if f"{shop_id}_Azimuth" in trilateration_df.columns:
							uwb_azimuths[shop_id] = trilateration_df[f"{shop_id}_Azimuth"].values
							true_uwb_azimuths[shop_id] = trilateration_df[f"{shop_id}_Actual_Azimuth"].values

					
				else:
					if "consolidated" not in iphone_paths[path_id]:
						consolidate_shop_measurements_by_timestamp(iphone_paths, "unfiltered")
					if not ("consolidated" in iphone_paths[path_id] and "unfiltered" in iphone_paths[path_id]["consolidated"]):
						continue 
					data = iphone_paths[path_id]["consolidated"]["unfiltered"]
					if len(data) == 0:
						continue 

					i = 0
					trilateration_start_time = time.time()
					while i < len(data):
						temp_dist = {}
						temp_uwb_azimuths = {}
						temp_actual_uwb_azimuths = {}
						start_idx = curr_idx = i
						interval_last_timestamp = None 
						while curr_idx < len(data) and (data[curr_idx]["timestamp"] - data[start_idx]["timestamp"]) < time_interval:
							curr_shop_id = data[curr_idx]["shop"]
							if curr_shop_id not in temp_dist:
								temp_dist[curr_shop_id] = []
								temp_uwb_azimuths[curr_shop_id] = []
								temp_actual_uwb_azimuths[curr_shop_id] = []

							temp_dist[curr_shop_id].append(data[curr_idx]["distance"])
							if "azimuth" in data[curr_idx] and data[curr_idx]["azimuth"] != util.DUMMY:
								temp_uwb_azimuths[curr_shop_id].append(data[curr_idx]["azimuth"])
								temp_actual_uwb_azimuths[curr_shop_id].append(data[curr_idx]["actual azimuth"])
							interval_last_timestamp = data[curr_idx]["timestamp"]
							
							curr_idx += 1
							
						for curr_shop_id in temp_dist:
							temp_dist[curr_shop_id] = np.array(temp_dist[curr_shop_id]).mean(axis=0)
							if len(temp_uwb_azimuths[curr_shop_id]) > 0:
								temp_uwb_azimuths[curr_shop_id] = util.azimuth_mean(temp_uwb_azimuths[curr_shop_id]) 
								temp_actual_uwb_azimuths[curr_shop_id] = util.azimuth_mean(temp_actual_uwb_azimuths[curr_shop_id]) 
							else:
								temp_uwb_azimuths[curr_shop_id] = np.nan 
								temp_actual_uwb_azimuths[curr_shop_id] = np.nan
						anchor_conn_stats[len(temp_dist)] = anchor_conn_stats.get(len(temp_dist), 0) + 1
						
						# trilateration	
						if len(temp_dist) > 2:
							temp_pos = trilateration(shop_locations, temp_dist, max_num_anchors)
							est_dist = None 
							if len(trilateration_positions) > 0:
								time_diff = interval_last_timestamp - trilateration_timestamps[-1]
								est_dist = time_diff * 2 # assume max speed is 2 m/s
							
							if True or est_dist is None or (util.calculate_vector_length(np.array(temp_pos) - trilateration_positions[-1]) < est_dist):
								trilateration_positions.append(np.array(temp_pos))
								trilateration_timestamps.append(interval_last_timestamp)
								measurement_type.append("trilateration")

								for shop_id in shop_locations:
									if shop_id in temp_uwb_azimuths:
										uwb_azimuths[shop_id].append(temp_uwb_azimuths[shop_id])
										true_uwb_azimuths[shop_id].append(temp_actual_uwb_azimuths[shop_id])
									else:
										uwb_azimuths[shop_id].append(np.nan)
										true_uwb_azimuths[shop_id].append(np.nan)
								

							tr_rate += 1
						# dilateration
						elif dilateration_enabled and len(temp_dist) > 1 and len(temp_uwb_azimuths) > 1:
							# check if there's a trilateration benchmark position within this time window
							if len(trilateration_timestamps) > 0:
								
								dists = []
								azimuths = []
								shop_coords = []
								for curr_shop_id in temp_dist:
									dists.append(temp_dist[curr_shop_id])
									azimuths.append(temp_uwb_azimuths[curr_shop_id])
									shop_coords.append(shop_locations[curr_shop_id])
								temp_pos = dilateration(dists[0], dists[1], azimuths[0], azimuths[1], shop_coords[0], shop_coords[1])
								if temp_pos is not None:
									time_diff = data[curr_idx - 1]["timestamp"] - trilateration_timestamps[-1]
									est_dist = time_diff * 2 # assume max speed is 2 m/s
									if util.calculate_vector_length(np.array(temp_pos) - trilateration_positions[-1]) < est_dist:
										trilateration_positions.append(np.array(temp_pos))
										trilateration_timestamps.append(data[start_idx]["timestamp"])
										measurement_type.append("dilateration")
										dr_rate += 1

										for shop_id in shop_locations:
											if shop_id in temp_uwb_azimuths:
												uwb_azimuths[shop_id].append(temp_uwb_azimuths[shop_id])
												true_uwb_azimuths[shop_id].append(temp_actual_uwb_azimuths[shop_id])
											else:
												uwb_azimuths[shop_id].append(np.nan)
												true_uwb_azimuths[shop_id].append(np.nan)
						
						num_windows += 1
						i = curr_idx
					
					total_runtime_s = time.time() - trilateration_start_time
					n_updates = max(1, len(trilateration_timestamps))
					inference_time = (total_runtime_s / n_updates) * 1000.0  # ms per update
					trilateration_orientations = util.positions2orientations(trilateration_positions) 
					trilateration_positions = np.array(trilateration_positions)
					trilateration_timestamps = np.array(trilateration_timestamps)
					
					# cache trilateration results for testing 
					trilateration_df = pd.DataFrame({
						"Timestamp": trilateration_timestamps,
						"PositionX": trilateration_positions[:, 0],
						"PositionY": trilateration_positions[:, 1],
						"Orientation": trilateration_orientations,
						"measurement_type": measurement_type,
						"InferenceTimePerUpdateMs": inference_time,
						"InferenceTime": inference_time
					})
					for shop_id in uwb_azimuths:
						trilateration_df[f"{shop_id}_Azimuth"] = uwb_azimuths[shop_id]
						trilateration_df[f"{shop_id}_Actual_Azimuth"] = true_uwb_azimuths[shop_id]
					trilateration_df.to_csv(f"path_files/trilateration_{path_id}_MaxAPs={max_num_anchors}_TI={time_interval}.csv", index=False)

				if len(trilateration_positions) > 0:
					iphone_paths[path_id]["trilateration"] = {
						"positions": trilateration_positions,
						"orientations": trilateration_orientations,
						"timestamps": trilateration_timestamps,
						"azimuths": uwb_azimuths,
						"true azimuths": true_uwb_azimuths,
						"inference time": inference_time
					}

					if "ekf" in methods:
						if use_cached_outputs and os.path.exists(f"path_files/ekf_{path_id}_MaxAPs={max_num_anchors}_TI={time_interval}.csv"):
							ekf_df = pd.read_csv(f"path_files/ekf_{path_id}_MaxAPs={max_num_anchors}_TI={time_interval}.csv")
							
							fused_path = ekf_df[["PositionX", "PositionY"]].values
							ekf_timestamps = ekf_df["Timestamp"].values
							ekf_orientations = ekf_df["Orientation"].values
							if "InferenceTimePerUpdateMs" in ekf_df.columns:
								ekf_inference_time = float(np.mean(ekf_df["InferenceTimePerUpdateMs"].values))
							else:
								# Backward compatibility with old cached files.
								ekf_inference_time = float(np.mean(ekf_df["InferenceTime"].values))
							ekf_azimuths = {}
							ekf_true_azimuths = {}
							for shop_id in shop_locations:
								if f"{shop_id}_Azimuth" in ekf_df.columns:
									ekf_azimuths[shop_id] = ekf_df[f"{shop_id}_Azimuth"].values
									ekf_true_azimuths[shop_id] = ekf_df[f"{shop_id}_Actual_Azimuth"].values
						else:
							if imu_data is not None:
								# kalman filter on trilateration and dilateration positions
								
								curr_trilateration_df = pd.DataFrame({
									"PositionX": trilateration_positions[:, 0], 
									"PositionY": trilateration_positions[:, 1], 
									"Timestamp": trilateration_timestamps, 
									"measurement_type": measurement_type
								})

								# azimuths 
								uwb_rows = [f"{shop_id}_Azimuth" for shop_id in uwb_azimuths] + [f"{shop_id}_Actual_Azimuth" for shop_id in true_uwb_azimuths] + ["PositionX", "PositionY"]
								for shop_id in uwb_azimuths:
									curr_trilateration_df[f"{shop_id}_Azimuth"] = uwb_azimuths[shop_id]
									curr_trilateration_df[f"{shop_id}_Actual_Azimuth"] = true_uwb_azimuths[shop_id]
								
								curr_imu_df = imu_data[path_id]
								trilateration_imu_df = te.merge_uwb_imu_data(curr_trilateration_df, curr_imu_df, uwb_rows)
								
								# EKF fusion
								if "ekf" in methods:
									
									ekf, initial_state = te.setup_ekf(curr_trilateration_df)

									# Optional ML heading fusion: if a column named 'MLHeadingRad' exists (radians),
									# pass it as a pseudo-measurement. Otherwise, regular EKF runs unchanged.
									heading_col = 'MLHeadingRad' if 'MLHeadingRad' in trilateration_imu_df.columns else None
									ekf_start_time = trilateration_start_time
									fused_path, ekf_timestamps, ekf_idxs, ekf_diag = te.run_ekf(
										ekf,
										trilateration_imu_df,
										initial_state,
										smooth=True,
										use_ml_heading=(heading_col is not None),
										heading_column=heading_col,
										heading_noise_deg=10.0,
										return_diagnostics=True,
									)

									ekf_orientations = util.positions2orientations(fused_path)
									ekf_runtime_s = time.time() - ekf_start_time
									n_ekf_updates = max(1, len(ekf_timestamps))
									ekf_inference_time = (ekf_runtime_s / n_ekf_updates) * 1000.0  # ms per update
									
									ekf_azimuths = {f"{shop_id}": trilateration_imu_df.loc[ekf_idxs, f"{shop_id}_Azimuth"].values for shop_id in shop_locations if f"{shop_id}_Azimuth" in trilateration_imu_df.columns}
									ekf_true_azimuths = {f"{shop_id}": trilateration_imu_df.loc[ekf_idxs, f"{shop_id}_Actual_Azimuth"].values for shop_id in shop_locations if f"{shop_id}_Actual_Azimuth" in trilateration_imu_df.columns}

									# cache ekf results for testing 
									ekf_df = pd.DataFrame({
										"Timestamp": ekf_timestamps,
										"PositionX": fused_path[:, 0],
										"PositionY": fused_path[:, 1],
										"Orientation": ekf_orientations,
										"InferenceTimePerUpdateMs": ekf_inference_time,
										"InferenceTime": ekf_inference_time,
										"InnovationResidualX_m": ekf_diag["innovation_residual_x"],
										"InnovationResidualY_m": ekf_diag["innovation_residual_y"],
										"InnovationResidualNorm_m": ekf_diag["innovation_residual_norm"],
										"PredictedPositionX_m": ekf_diag["predicted_position_x"],
										"PredictedPositionY_m": ekf_diag["predicted_position_y"],
										"UWBMeasurementX_m": ekf_diag["uwb_measurement_x"],
										"UWBMeasurementY_m": ekf_diag["uwb_measurement_y"],
									})

									for shop_id in ekf_azimuths:
										ekf_df[f"{shop_id}_Azimuth"] = ekf_azimuths[shop_id]
										ekf_df[f"{shop_id}_Actual_Azimuth"] = ekf_true_azimuths[shop_id]
									ekf_df.to_csv(f"path_files/ekf_{path_id}_MaxAPs={max_num_anchors}_TI={time_interval}.csv", index=False)


						iphone_paths[path_id]["ekf"] = {
							"positions": fused_path,
							"orientations": ekf_orientations,
							"timestamps": ekf_timestamps, 
							"inference time": ekf_inference_time
						}

						# now let's calculate positioning availability 
						pa, full_pa = util.calculate_position_availability(iphone_paths[path_id]["ekf"]["timestamps"], time_interval)
						pas += pa 
						full_pas += full_pa
						iphone_paths[path_id]["ekf"]["azimuths"] = ekf_azimuths
						iphone_paths[path_id]["ekf"]["true azimuths"] = ekf_true_azimuths
									
	return iphone_paths

def evaluate_motion_errors(iphone_paths, true_paths, shop_locations, methods, directions_mode, azimuth_fusion_methods):
	motion_azimuth_errors = {method: [] for method in methods}
	uwb_azimuth_errors = [] 
	uwb_to_motion_azimuth_errors = [] 
	fused_azimuth_errors = {method: {} for method in methods}
	distance_errors = {method: [] for method in methods}
	position_errors = {method: [] for method in methods}
	inference_times = {method: [] for method in methods}
	memory_footprints_mb = {method: [] for method in methods}
	dir_strs = []
	if directions_mode == "all":
		dir_strs = ["back left", "front left", "front", "front right", "back right", "back"]
	elif directions_mode == "front only":
		dir_strs = ["front left", "front", "front right"]
	elif directions_mode == "front back only":
		dir_strs = ["front", "back"]
	elif directions_mode == "all without front and back":
		dir_strs = ["back left", "front right", "back right", "front left"]
	else:
		raise ValueError("Invalid directions mode:", directions_mode)
	
	direction_errors = {}
	dir_str_dict = {
		dir: [] for dir in dir_strs
	}
	predicted_labels = {method: [] for method in methods}
	true_labels = {method: [] for method in methods}
	predicted_labels["UWB only"] = []
	true_labels["UWB only"] = []
	predicted_timestamps = {method: [] for method in methods}
	predicted_timestamps["UWB only"] = []

	# Collect per-position paired correctness for temporal dependence MBB CI
	paired_correctness = {'trilateration_ml': [], 'trilateration_uwb': [],
	                      'ekf_ml': [], 'ekf_uwb': []}

	# Collect per-method C_{A,pos} = fraction of shops correct at each position
	per_method_c_at = {'trilateration': [], 'ekf': [], 'UWB only': []}

	# Collect per-position mean azimuth and distance errors
	per_method_azimuth = {'trilateration': [], 'ekf': [], 'UWB only': []}
	per_method_azimuth_ts = {'trilateration': [], 'ekf': [], 'UWB only': []}
	per_method_distance = {'trilateration': [], 'ekf': []}
	per_method_distance_ts = {'trilateration': [], 'ekf': []}

	for path_id in iphone_paths:
		if path_id not in true_paths:
			continue
		true_path = true_paths[path_id]
		uwb_azimuth_error_calculated = False
		for method in methods:
			if method not in iphone_paths[path_id]:
				continue

			iphone_path = iphone_paths[path_id][method]
		
			true_timestamps = true_path["timestamps"]
			motion_timestamps = iphone_path["timestamps"]
			
			true_idxs, motion_idxs = util.find_closest_ground_truth_indices(true_timestamps, motion_timestamps, True)
			true_positions = true_path["positions"][true_idxs]
			true_orientations = true_path["orientations"][true_idxs]
			true_motion_orientations = util.positions2orientations(true_positions) # the first orientation is invalid
			motion_positions = iphone_path["positions"][motion_idxs]
			motion_orientations = iphone_path["orientations"][motion_idxs]
			true_timestamps = true_timestamps[true_idxs]
			motion_timestamps = motion_timestamps[motion_idxs]
			inference_time = iphone_path["inference time"]
			inference_times[method].append(inference_time)
			memory_footprint_mb = _estimate_method_memory_mb(iphone_path)
			if not np.isnan(memory_footprint_mb):
				memory_footprints_mb[method].append(memory_footprint_mb)
			valid_idxs_mask = ~np.isnan(true_motion_orientations) & ~np.isnan(motion_orientations)
			true_positions = true_positions[valid_idxs_mask]
			true_motion_orientations = true_motion_orientations[valid_idxs_mask]
			motion_positions = motion_positions[valid_idxs_mask]
			motion_orientations = motion_orientations[valid_idxs_mask]
			true_orientations = true_orientations[valid_idxs_mask]

			# Per-position C_{A,pos} accumulation: sum across shops at each position
			n_positions = len(motion_positions)
			method_per_pos_correct = np.zeros(n_positions)
			method_per_pos_total = np.zeros(n_positions)
			method_per_pos_az_sum = np.zeros(n_positions)
			method_per_pos_az_cnt = np.zeros(n_positions)
			method_per_pos_d_sum = np.zeros(n_positions)
			method_per_pos_d_cnt = np.zeros(n_positions)
			if method == 'trilateration':
				uwb_per_pos_correct = np.zeros(n_positions)
				uwb_per_pos_total = np.zeros(n_positions)
				uwb_per_pos_az_sum = np.zeros(n_positions)
				uwb_per_pos_az_cnt = np.zeros(n_positions)

			position_errors[method].append(util.calculate_distance_between_vectors(motion_positions, true_positions))

			for shop_id in shop_locations:
				# check the batch id 
				batch_id = int(shop_id.split("-")[1])
				if (batch_id == 1 and path_id > "P40") or (batch_id == 2 and path_id < "P40"):
					continue 
				true_tas = shop_locations[shop_id] - true_positions
				motion_tas = shop_locations[shop_id] - motion_positions
				
				true_distances = util.calculate_vector_lengths(true_tas)
				motion_distances = util.calculate_vector_lengths(motion_tas)

				distance_errors[method].append(np.abs(motion_distances - true_distances))
				
				true_motion_azimuths = util.calculate_azimuths(util.theta2orientation_vector(true_motion_orientations), true_tas)
				motion_azimuths = util.calculate_azimuths(util.theta2orientation_vector(motion_orientations), motion_tas)
				
				motion_azimuth_diffs = util.azimuth_error(true_motion_azimuths, motion_azimuths)
				motion_azimuth_diffs = motion_azimuth_diffs[~np.isnan(motion_azimuth_diffs)]
				if len(motion_azimuth_diffs) > 0:
					motion_azimuth_errors[method].append(motion_azimuth_diffs)
				
				if shop_id not in iphone_path["azimuths"] or shop_id not in iphone_path["true azimuths"]:
					continue
				uwb_azimuths = np.array(iphone_path["azimuths"][shop_id])[valid_idxs_mask]
				true_uwb_azimuths = np.array(iphone_path["true azimuths"][shop_id])[valid_idxs_mask]

				delta_thetas = util.calculate_angles_between_vectors(util.theta2orientation_vector(motion_orientations), util.theta2orientation_vector(true_orientations))
				azimuth_signs = util.get_vector_directions(util.theta2orientation_vector(motion_orientations), motion_tas) # true_motion_orientations, true_tas
				front_back_relative_to_motion = util.calculate_angles_between_vectors(util.theta2orientation_vector(motion_orientations), motion_tas) > 90 # true_motion_orientations, true_tas
				uwb_to_motion_azimuths = util.estimate_motion_azimuths(uwb_azimuths, delta_thetas, azimuth_signs, add=front_back_relative_to_motion) # true_uwb_azimuths
				fused_azimuths = {azimuth_fusion_method: util.fuse_motion_and_uwb_azimuths(motion_azimuths, uwb_to_motion_azimuths, method=azimuth_fusion_method) for azimuth_fusion_method in azimuth_fusion_methods}
				
				uwb_azimuth_diffs_raw = util.azimuth_error(true_uwb_azimuths, uwb_azimuths)
				uwb_azimuth_diffs = uwb_azimuth_diffs_raw[~np.isnan(uwb_azimuth_diffs_raw)]
				
				if len(uwb_azimuth_diffs) > 0 and not uwb_azimuth_error_calculated:
					uwb_azimuth_errors.append(uwb_azimuth_diffs)

				uwb_to_motion_azimuth_diffs = util.azimuth_error(true_motion_azimuths, uwb_to_motion_azimuths)
				uwb_to_motion_azimuth_diffs = uwb_to_motion_azimuth_diffs[~np.isnan(uwb_to_motion_azimuth_diffs)]
				if len(uwb_to_motion_azimuth_diffs) > 0 and not uwb_azimuth_error_calculated:
					uwb_to_motion_azimuth_errors.append(uwb_to_motion_azimuth_diffs)

				fused_azimuth_diffs = {azimuth_fusion_method: util.azimuth_error(true_motion_azimuths, fused_azimuths[azimuth_fusion_method]) for azimuth_fusion_method in fused_azimuths}
				fused_azimuth_diffs = {azimuth_fusion_method: fused_azimuth_diffs[azimuth_fusion_method][~np.isnan(fused_azimuth_diffs[azimuth_fusion_method])] for azimuth_fusion_method in fused_azimuth_diffs}
				for azimuth_fusion_method in fused_azimuth_diffs:
					#print(method, azimuth_fusion_method, path_id, shop_id, "number of fused diffs:", len(fused_azimuth_diffs[azimuth_fusion_method]))
					if len(fused_azimuth_diffs[azimuth_fusion_method]) > 0:
						if azimuth_fusion_method not in fused_azimuth_errors[method]:
							fused_azimuth_errors[method][azimuth_fusion_method] = []
						fused_azimuth_errors[method][azimuth_fusion_method].append(fused_azimuth_diffs[azimuth_fusion_method])

				true_motion_directions = util.azimuths_to_directions(true_motion_azimuths, directions_mode)
				true_uwb_directions = util.azimuths_to_directions(true_uwb_azimuths, directions_mode)
				motion_directions = util.azimuths_to_directions(motion_azimuths, directions_mode)
				fused_directions = {azimuth_fusion_method: util.azimuths_to_directions(fused_azimuths[azimuth_fusion_method], directions_mode) for azimuth_fusion_method in fused_azimuths}
				uwb_to_motion_directions = util.azimuths_to_directions(uwb_to_motion_azimuths, directions_mode)
				uwb_directions = util.azimuths_to_directions(uwb_azimuths, directions_mode)

				# Accumulate per-position C_{A,pos} for method (fraction correct across shops)
				valid_pos = ~np.isnan(motion_directions) & ~np.isnan(true_motion_directions)
				method_per_pos_correct[valid_pos] += (motion_directions[valid_pos] == true_motion_directions[valid_pos]).astype(int)
				method_per_pos_total[valid_pos] += 1

				# Accumulate per-position azimuth and distance errors
				az_valid = ~np.isnan(motion_azimuth_diffs)
				method_per_pos_az_sum[az_valid] += motion_azimuth_diffs[az_valid]
				method_per_pos_az_cnt[az_valid] += 1
				dist_diffs = np.abs(motion_distances - true_distances)
				d_valid = ~np.isnan(dist_diffs)
				method_per_pos_d_sum[d_valid] += dist_diffs[d_valid]
				method_per_pos_d_cnt[d_valid] += 1

				# Accumulate UWB per-position once per path (use UWB's own ground truth)
				if not uwb_azimuth_error_calculated:
					uwb_valid = ~np.isnan(uwb_directions) & ~np.isnan(true_uwb_directions)
					uwb_per_pos_correct[uwb_valid] += (uwb_directions[uwb_valid] == true_uwb_directions[uwb_valid]).astype(int)
					uwb_per_pos_total[uwb_valid] += 1
					# UWB per-position azimuth error
					uwb_az_valid = ~np.isnan(uwb_azimuth_diffs_raw)
					uwb_per_pos_az_sum[uwb_az_valid] += uwb_azimuth_diffs_raw[uwb_az_valid]
					uwb_per_pos_az_cnt[uwb_az_valid] += 1

				# Collect per-position paired correctness
				paired_mask = (~np.isnan(motion_directions) & ~np.isnan(true_motion_directions) &
				               ~np.isnan(uwb_directions) & ~np.isnan(true_uwb_directions))
				if paired_mask.sum() > 0:
					pc_d = true_motion_directions[paired_mask]
					mkey = method  # 'trilateration' or 'ekf'
					paired_correctness[f'{mkey}_ml'].extend((motion_directions[paired_mask] == pc_d).astype(int).tolist())
					# UWB uses its own ground truth for correctness
					uwb_pc_d = true_uwb_directions[paired_mask]
					paired_correctness[f'{mkey}_uwb'].extend((uwb_directions[paired_mask] == uwb_pc_d).astype(int).tolist())
				for dir_str in dir_strs:
					if dir_str == "back left":
						d = -2
					elif dir_str == "front left":
						d = -1
					elif dir_str == "front":
						d = 0
					elif dir_str == "front right":
						d = 1
					elif dir_str == "back right":
						d = 2
					elif dir_str == "back":
						d = 3
					dir_mask = (true_motion_directions == d) 
					# tmd = true_motion_directions[td_idxs]
					# md = motion_directions[td_idxs]
					# fd = fused_directions[td_idxs]
					# umd = uwb_to_motion_directions[td_idxs]
	
					# td_idxs = true_uwb_directions == d 
					# tud = true_uwb_directions[td_idxs]
					# ud = uwb_directions[td_idxs]

					
					dir_nan_mask = dir_mask & ~np.isnan(motion_directions)
					md = motion_directions[dir_nan_mask] 
					if len(md) > 0:
						if method not in direction_errors: 
							direction_errors[method] = {dir: [] for dir in dir_strs}
						direction_errors[method][dir_str].append(md == true_motion_directions[dir_nan_mask])
						
						predicted_labels[method].extend(md)
						true_labels[method].extend(true_motion_directions[dir_nan_mask])
						predicted_timestamps[method].extend(
							motion_timestamps[valid_idxs_mask][dir_nan_mask].tolist()
						)

					for azimuth_fusion_method in fused_directions:
						dir_nan_mask = dir_mask & ~np.isnan(fused_directions[azimuth_fusion_method])
						fd = fused_directions[azimuth_fusion_method][dir_nan_mask]
						if len(fd) > 0:
							if method + f" ({azimuth_fusion_method})" not in direction_errors:
								direction_errors[method + f" ({azimuth_fusion_method})"] = {dir: [] for dir in dir_strs}
							direction_errors[method + f" ({azimuth_fusion_method})"][dir_str].append(fd == true_motion_directions[dir_nan_mask])
							
					# dir_nan_mask = dir_mask & ~np.isnan(uwb_to_motion_directions)
					# umd = uwb_to_motion_directions[dir_nan_mask]
					# if len(umd) > 0:
					# 	if "uwb" not in direction_errors:
					# 		direction_errors["uwb"] = {dir: [] for dir in dir_strs}
					# 	direction_errors["uwb"][dir_str].append(umd == true_motion_directions[dir_nan_mask])

				dir_mask = true_uwb_directions == d 
				dir_nan_mask = dir_mask & ~np.isnan(uwb_directions)
				ud = uwb_directions[dir_nan_mask]
				if len(ud) > 0:
					if "UWB only" not in direction_errors:
						direction_errors["UWB only"] = {dir: [] for dir in dir_strs}
					direction_errors["UWB only"][dir_str].append(ud == true_uwb_directions[dir_nan_mask])

					predicted_labels["UWB only"].extend(ud)
					true_labels["UWB only"].extend(true_uwb_directions[dir_nan_mask])

			uwb_azimuth_error_calculated = True

			# Extract C_{A,pos} for this method (fraction correct across shops at each position)
			mask = method_per_pos_total > 0
			if mask.sum() > 0:
				c_at = method_per_pos_correct[mask] / method_per_pos_total[mask]
				per_method_c_at[method].extend(c_at.tolist())

			# Extract per-position mean azimuth error
			az_mask = method_per_pos_az_cnt > 0
			if az_mask.sum() > 0:
				mean_az = method_per_pos_az_sum[az_mask] / method_per_pos_az_cnt[az_mask]
				per_method_azimuth[method].extend(mean_az.tolist())
				method_ts = motion_timestamps[valid_idxs_mask][az_mask]
				per_method_azimuth_ts[method].extend(method_ts.tolist())

			# Extract per-position mean distance error
			d_mask = method_per_pos_d_cnt > 0
			if d_mask.sum() > 0:
				mean_d = method_per_pos_d_sum[d_mask] / method_per_pos_d_cnt[d_mask]
				per_method_distance[method].extend(mean_d.tolist())
				method_ts_d = motion_timestamps[valid_idxs_mask][d_mask]
				per_method_distance_ts[method].extend(method_ts_d.tolist())

			# Extract UWB C_{A,pos} and per-position azimuth error after trilat iteration
			if method == 'trilateration':
				uwb_valid = uwb_per_pos_total > 0
				if np.sum(uwb_valid) > 0:
					uwb_c_at = uwb_per_pos_correct[uwb_valid] / uwb_per_pos_total[uwb_valid]
					per_method_c_at['UWB only'].extend(uwb_c_at.tolist())
				uwb_az_valid = uwb_per_pos_az_cnt > 0
				if np.sum(uwb_az_valid) > 0:
					uwb_mean_az = uwb_per_pos_az_sum[uwb_az_valid] / uwb_per_pos_az_cnt[uwb_az_valid]
					per_method_azimuth['UWB only'].extend(uwb_mean_az.tolist())
				uwb_ts = motion_timestamps[valid_idxs_mask][uwb_az_valid]
				per_method_azimuth_ts['UWB only'].extend(uwb_ts.tolist())
	stats_dfs = []
	# Save raw error sequences for per-metric MBB CIs
	_error_sequences = {}
	for method in methods:
		if len(distance_errors[method]) == 0:
			continue
		
		distance_errors[method] = np.concatenate(distance_errors[method], axis=0)
		_error_sequences[(method, 'distance')] = distance_errors[method].copy()
		distance_errors[method] = pd.DataFrame(distance_errors[method], columns=[f"Distance error ({method})"])
		# print("Distance error statistics for method:", method)
		# print(distance_errors[method].describe())
		distance_df = distance_errors[method].describe(percentiles=[.75, .9]) 
		stats_dfs.append(distance_df)
		
		# plt.hist(distance_errors[method], bins=30)
		# plt.xlabel("Distance error (m)")
		# plt.ylabel("Frequency")
		# plt.title(f"Distance error distribution ({method})")
		# plt.show()

		if len(position_errors[method]) == 0:
			continue
		position_errors[method] = np.concatenate(position_errors[method], axis=0)
		_error_sequences[(method, 'position')] = position_errors[method].copy()
		position_errors[method] = pd.DataFrame(position_errors[method], columns=[f"Position error ({method})"])
		# print("Position error statistics for method:", method)
		# print(position_errors[method].describe())
		position_df = position_errors[method].describe(percentiles=[.75, .9])
		stats_dfs.append(position_df)

		# plt.hist(position_errors[method], bins=30)
		# plt.xlabel("Position error (m)")
		# plt.ylabel("Frequency")
		# plt.title(f"Position error distribution ({method})")
		# plt.show()

		if len(motion_azimuth_errors[method]) == 0:
			continue
		motion_azimuth_errors[method] = np.concatenate(motion_azimuth_errors[method], axis=0)
		_error_sequences[(method, 'azimuth')] = motion_azimuth_errors[method].copy()
		motion_azimuth_errors[method] = pd.DataFrame(motion_azimuth_errors[method], columns=[f"Method azimuth error ({method})"])
		
		# if method == "ekf":
		# 	print(motion_azimuth_errors[method])
		# 	print("Azimuth error statistics for method:", method)
		# 	print(motion_azimuth_errors[method].describe())
		azimuth_df = motion_azimuth_errors[method].describe(percentiles=[.75, .9])
		stats_dfs.append(azimuth_df)

		for azimuth_fusion_method in azimuth_fusion_methods:
			if azimuth_fusion_method not in fused_azimuth_errors[method]:
				print(fused_azimuth_errors[method])
				continue
			fused_azimuth_errors[method][azimuth_fusion_method] = np.concatenate(fused_azimuth_errors[method][azimuth_fusion_method], axis=0)
			fused_azimuth_errors[method][azimuth_fusion_method] = pd.DataFrame(fused_azimuth_errors[method][azimuth_fusion_method], columns=[f"Fused azimuth error ({method}, {azimuth_fusion_method})"])
			fused_azimuth_df = fused_azimuth_errors[method][azimuth_fusion_method].describe(percentiles=[.75, .9])
			stats_dfs.append(fused_azimuth_df)

	
	if len(uwb_azimuth_errors) > 0:
		uwb_azimuth_errors = np.concatenate(uwb_azimuth_errors, axis=0)
		_error_sequences[('UWB only', 'azimuth')] = uwb_azimuth_errors.copy()
		uwb_azimuth_errors = pd.DataFrame(uwb_azimuth_errors, columns=[f"UWB azimuth error"])
		uwb_azimuth_df = uwb_azimuth_errors.describe(percentiles=[.75, .9])
		stats_dfs.append(uwb_azimuth_df)

	if len(uwb_to_motion_azimuth_errors) > 0:
		uwb_to_motion_azimuth_errors = np.concatenate(uwb_to_motion_azimuth_errors, axis=0)
		uwb_to_motion_azimuth_errors = pd.DataFrame(uwb_to_motion_azimuth_errors, columns=[f"UWB to motion azimuth error"])
		uwb_to_motion_azimuth_df = uwb_to_motion_azimuth_errors.describe(percentiles=[.75, .9])
		stats_dfs.append(uwb_to_motion_azimuth_df)
		#print(method, "mean azimuth error:", np.mean(np.abs(azimuth_errors[method])), "median azimuth error:", np.median(np.abs(azimuth_errors[method])), "max azimuth error:", np.max(np.abs(azimuth_errors[method])))
		# plt.hist(azimuth_errors[method], bins=30)
		# plt.xlabel("Azimuth error (degrees)")
		# plt.ylabel("Frequency")
		# plt.title(f"Azimuth error distribution ({method})")
		# plt.show()

		#print("fused direction errors:", fused_direction_errors["ekf"])
		# direction accuracy 
	
	for method in inference_times:
		inference_times[method] = np.mean(inference_times[method])
		inference_time_df = pd.DataFrame({f"Inference time ({method})": [inference_times[method]]}, index=["mean"])
		stats_dfs.append(inference_time_df)
		if len(memory_footprints_mb[method]) > 0:
			mean_mem_mb = float(np.mean(memory_footprints_mb[method]))
			memory_df = pd.DataFrame({f"Memory footprint ({method})": [mean_mem_mb]}, index=["mean"])
			stats_dfs.append(memory_df)
		
	for method in direction_errors:
		temp = []
		for dir_str in direction_errors[method]:
			if len(direction_errors[method][dir_str]) == 0:
				direction_errors[method][dir_str] = np.nan
				continue
			temp.append(np.concatenate(direction_errors[method][dir_str], axis=0))
			direction_errors[method][dir_str] = np.mean(np.concatenate(direction_errors[method][dir_str], axis=0)) * 100 
		temp = np.array(np.concatenate(temp, axis=0))

		direction_errors_df = pd.DataFrame({f"Direction accuracy ({method})": [np.mean(temp), len(temp)]}, index=["mean", "count"])
		stats_dfs.append(direction_errors_df)

		direction_errors[method]['average'] = np.mean(temp) * 100

		# fused_direction_errors[method][dir_str] = np.mean(np.concatenate(fused_direction_errors[method][dir_str], axis=0)) * 100
		# #print("Direction accuracy for", dir_str, "for method", method, ":", direction_errors[method][dir_str])
		# uwb_direction_errors[method][dir_str] = np.mean(np.concatenate(uwb_direction_errors[method][dir_str], axis=0)) * 100
	
	dir_strs.append("average")
	dir_accs = {method.capitalize().replace("Ekf", "EKF").replace("Uwb", "UWB"): [direction_errors[method][dir_str] for dir_str in dir_strs] for method in direction_errors.keys()}
	
	stats_dfs = pd.concat(stats_dfs, axis=1).T
	print(stats_dfs)

	classification_rows = []
	available_methods = []
	for method in predicted_labels:
		predicted_labels[method] = np.array(predicted_labels[method])
		true_labels[method] = np.array(true_labels[method])
		available_methods.append(method)
		metrics = util.calculate_classification_metrics(true_labels[method], predicted_labels[method])
		bootstrap_metrics = util.bootstrap_classification_metrics_mbb(
			true_labels[method],
			predicted_labels[method],
			n_bootstrap=2000,
			block_length={'trilateration': 120, 'ekf': 120, 'UWB only': 49}.get(method, 30),
		)

		classification_row = {"Samples": len(true_labels[method])}
		classification_row.update(metrics)
		for metric_name, values in bootstrap_metrics.items():
			classification_row[f"{metric_name} CI Lower"] = values["ci_lower"]
			classification_row[f"{metric_name} CI Upper"] = values["ci_upper"]
		classification_rows.append(pd.DataFrame(classification_row, index=[f"Classification ({method})"]))

		print(f"\nClassification report for {method}:")
		print(f"Total samples: {len(true_labels[method])}")
		print(metrics)
		print("95% bootstrap confidence intervals:")
		for metric_name, values in bootstrap_metrics.items():
			print(
				f"  {metric_name}: {values['mean']:.4f} "
				f"[{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]"
			)

	if len(classification_rows) > 0:
		stats_dfs = pd.concat([stats_dfs] + classification_rows, axis=0)

	display_names = {
		"trilateration": "Multilateration",
		"ekf": "EKF",
		"UWB only": "UWB only",
	}
	methods_with_labels = [method for method in ["UWB only", "trilateration", "ekf"] if method in true_labels and len(true_labels[method]) > 0]
	if len(methods_with_labels) >= 2:
		correct_counts = [int(np.sum(predicted_labels[method] == true_labels[method])) for method in methods_with_labels]
		total_counts = [int(len(true_labels[method])) for method in methods_with_labels]
		incorrect_counts = [total_counts[idx] - correct_counts[idx] for idx in range(len(methods_with_labels))]

		omnibus_result = util.chi_square_test_independence(correct_counts, total_counts)
		omnibus_df = pd.DataFrame(
			{
				"Methods": [", ".join(display_names.get(method, method) for method in methods_with_labels)],
				"Correct": [int(sum(correct_counts))],
				"Incorrect": [int(sum(incorrect_counts))],
				"Chi-square": [omnibus_result["statistic"]],
				"Degrees of freedom": [omnibus_result["degrees_of_freedom"]],
				"P Value": [omnibus_result["p_value"]],
			},
			index=["Chi-square omnibus"],
		)
		print("\nStatistical significance table (omnibus):")
		print(omnibus_df)
		stats_dfs = pd.concat([stats_dfs, omnibus_df], axis=0)

		pairwise_specs = []
		if "UWB only" in methods_with_labels:
			for comparison_method in ["trilateration", "ekf"]:
				if comparison_method in methods_with_labels:
					pairwise_specs.append((comparison_method, "UWB only"))

		pairwise_results = []
		for method_a, method_b in pairwise_specs:
			correct_a = int(np.sum(predicted_labels[method_a] == true_labels[method_a]))
			total_a = int(len(true_labels[method_a]))
			correct_b = int(np.sum(predicted_labels[method_b] == true_labels[method_b]))
			total_b = int(len(true_labels[method_b]))
			result = util.two_proportion_z_test(correct_a, total_a, correct_b, total_b)
			pairwise_results.append((method_a, method_b, correct_a, total_a, correct_b, total_b, result))

		if len(pairwise_results) > 0:
			familywise_alpha = 0.05
			n_tests = len(pairwise_results)
			bonferroni_alpha = familywise_alpha / n_tests
			adjusted_p_values = util.bonferroni_correction([item[6]["p_value"] for item in pairwise_results])
			pairwise_rows = []
			for item, adjusted_p_value in zip(pairwise_results, adjusted_p_values):
				method_a, method_b, correct_a, total_a, correct_b, total_b, result = item
				pairwise_rows.append({
					"Comparison": f"{display_names.get(method_a, method_a)} vs {display_names.get(method_b, method_b)}",
					f"{display_names.get(method_a, method_a)} correct": correct_a,
					f"{display_names.get(method_a, method_a)} total": total_a,
					f"{display_names.get(method_b, method_b)} correct": correct_b,
					f"{display_names.get(method_b, method_b)} total": total_b,
					"Accuracy difference": result["difference"],
					"Z statistic": result["z_statistic"],
					"P Value": result["p_value"],
					"Bonferroni alpha": bonferroni_alpha,
					"Bonferroni-adjusted P Value": adjusted_p_value,
					"Significant after Bonferroni": adjusted_p_value < familywise_alpha,
				})
			pairwise_df = pd.DataFrame(pairwise_rows).set_index("Comparison")
			print("\nStatistical significance table (pairwise z-tests):")
			print(pairwise_df)
			stats_dfs = pd.concat([stats_dfs, pairwise_df], axis=0)

	# Plot and save confusion matrices
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	# Override UWB labels with standalone 0.5 s aggregated predictions if available
	standalone_uwb_path = os.path.join("stats", "uwb_standalone_labels.npz")
	if os.path.exists(standalone_uwb_path):
		uwb_data = np.load(standalone_uwb_path)
		true_labels['UWB only'] = list(uwb_data['dir_true'])
		predicted_labels['UWB only'] = list(uwb_data['dir_pred'])
		print(f"[Confusion matrix] Loaded standalone UWB labels: {len(true_labels['UWB only'])} predictions")

	ordered = [(l, true_labels.get(k, []), predicted_labels.get(k, []))
	           for k, l in [('UWB only', 'UWB only'), ('trilateration', 'Multilateration'), ('ekf', 'EKF')]
	           if len(true_labels.get(k, [])) > 0]
	if ordered:
		all_classes = np.unique(np.concatenate([np.concatenate([np.array(t), np.array(p)]) for _, t, p in ordered]))
		fig, axes = plt.subplots(1, len(ordered), figsize=(5.8 * len(ordered), 5.8), squeeze=False, constrained_layout=True)
		image = None
		for idx, (title, tl, pl) in enumerate(ordered):
			cm, classes = util._build_confusion_matrix(tl, pl, classes=all_classes)
			image = util._plot_confusion_matrix_on_axis(axes[0, idx], cm, classes, title, show_ylabel=(idx == 0))
		if image is not None:
			fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
		fig.supxlabel('Predicted label')
		fig.supylabel('True label')
		out_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'latex', 'images', 'confusion_matrices2.png')
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		fig.savefig(out_path, dpi=150, bbox_inches='tight')
		plt.close(fig)
		print(f"\nConfusion matrices saved to: {out_path}")

	# Save paired correctness for temporal dependence MBB analysis
	paired_path = os.path.join("stats", "paired_correctness.npz")
	os.makedirs("stats", exist_ok=True)

	# Save FULL per-method binary correctness (for per-method ACF/block_length)
	full_path = os.path.join("stats", "per_method_correctness.npz")
	full_data = {}
	for method_name, display in [('trilateration', 'Multilateration'), ('ekf', 'EKF'), ('UWB only', 'UWB')]:
		if method_name in predicted_labels and len(predicted_labels[method_name]) > 0:
			pl = np.array(predicted_labels[method_name])
			tl = np.array(true_labels[method_name])
			full_data[display] = (pl == tl).astype(int)
	np.savez(full_path, **full_data)

	# Compute and print per-method ACF from FULL sequences
	print("\n" + "=" * 70)
	print("PER-METHOD ACF FROM FULL CORRECTNESS SEQUENCES")
	print("=" * 70)
	for display, seq in full_data.items():
		seq = seq.astype(float)
		n = len(seq)
		a = util._compute_acf(seq, max_lag=120)
		dl = util._decorrelation_lag(a)
		# ESS
		tau = 0.5
		max_k = min(14, n // 2 - 1)
		for k in range(1, max_k + 1):
			if 2*k >= len(a): break
			s = a[2*k - 1] + a[2*k]
			if s < 0: break
			tau += s
		ess = max(1, n / (2 * tau)) if tau > 0 else n
		# MBB CI on mean accuracy
		bl = max(1, int(dl))
		nblk = max(2, int(np.ceil(n / bl)))
		rng = np.random.default_rng(42)
		means = np.array([np.mean(np.concatenate(
			[seq[i:i+bl] for i in rng.integers(0, max(1, n - bl + 1), size=nblk)]
		)[:n]) for _ in range(2000)])
		lo = float(np.quantile(means, 0.025))
		hi = float(np.quantile(means, 0.975))
		print(f"\n{display}: n={n}, acc={np.mean(seq):.4f}")
		print(f"  lag-1={a[1]:.3f}, block={dl}, ESS={ess:.0f} ({ess/n*100:.1f}%)")
		print(f"  MBB CI: [{lo:.4f}, {hi:.4f}]")

	# Compute per-method C_{A,pos} ACF
	print("\n" + "=" * 70)
	print("PER-METHOD C_{A,pos} ACF (fraction of shops correct per position)")
	print("=" * 70)
	ca_results = {}
	for method, display in [('trilateration', 'Multilateration'), ('ekf', 'EKF'), ('UWB only', 'UWB')]:
		if method not in per_method_c_at or len(per_method_c_at[method]) == 0:
			continue
		seq = np.array(per_method_c_at[method], dtype=float)
		n = len(seq)
		mean_val = np.mean(seq)
		a = util._compute_acf(seq, max_lag=60)
		dl = util._decorrelation_lag(a)
		tau = 0.5
		for k in range(1, min(14, n // 2 - 1) + 1):
			if 2*k >= len(a): break
			s = a[2*k - 1] + a[2*k]
			if s < 0: break
			tau += s
		ess = max(1, n / (2 * tau)) if tau > 0 else n
		bl = max(1, int(dl))
		nblk = max(2, int(np.ceil(n / bl)))
		rng = np.random.default_rng(42)
		means = np.array([np.mean(np.concatenate(
			[seq[i:i+bl] for i in rng.integers(0, max(1, n - bl + 1), size=nblk)]
		)[:n]) for _ in range(2000)])
		lo = float(np.quantile(means, 0.025))
		hi = float(np.quantile(means, 0.975))
		print(f"\n{display}: n={n}, mean C_A,pos={mean_val:.4f}")
		print(f"  lag-1={a[1]:.3f}, block={dl}, ESS={ess:.0f} ({ess/n*100:.1f}%)")
		print(f"  MBB CI: [{lo:.4f}, {hi:.4f}]")
		ca_results[method] = {'n': n, 'mean': mean_val, 'lag1': a[1],
		                      'block': dl, 'ess': ess, 'ci_lo': lo, 'ci_hi': hi}

	np.savez(paired_path,
	         trilateration_ml=paired_correctness['trilateration_ml'],
	         trilateration_uwb=paired_correctness['trilateration_uwb'],
	         ekf_ml=paired_correctness['ekf_ml'],
	         ekf_uwb=paired_correctness['ekf_uwb'])

	# Compute paired MBB CI inline using util
	print("\n" + "=" * 70)
	print("PAIRED MBB CI FOR PAIRWISE ACCURACY DIFFERENCES")
	print("=" * 70)
	for mkey, mlabel in [('trilateration', 'Multilateration'), ('ekf', 'EKF')]:
		ml_seq = np.array(paired_correctness[f'{mkey}_ml'], dtype=int)
		uwb_seq = np.array(paired_correctness[f'{mkey}_uwb'], dtype=int)
		if len(ml_seq) < 10:
			continue
		acc_ml = np.mean(ml_seq)
		acc_uwb = np.mean(uwb_seq)
		diff_seq = ml_seq - uwb_seq
		mean_diff = np.mean(diff_seq)

		# Estimate block_length from paired-diff ACF
		a = util._compute_acf(diff_seq, max_lag=30)
		dl = util._decorrelation_lag(a)
		bl = max(1, int(dl))

		print(f"\n{mlabel} vs UWB only (paired, shared positions):")
		print(f"  n={len(ml_seq)}, ML acc={acc_ml:.4f}, UWB acc={acc_uwb:.4f}")
		print(f"  diff ACF lag-1={a[1]:.3f}, block_length={bl}")
		print(f"  point diff={mean_diff:.4f} ({mean_diff*100:.2f} pp)")

		# Per-method ACF
		ml_acf = util._compute_acf(ml_seq.astype(float), max_lag=30)
		ml_dl = util._decorrelation_lag(ml_acf)
		uwb_acf = util._compute_acf(uwb_seq.astype(float), max_lag=30)
		uwb_dl = util._decorrelation_lag(uwb_acf)
		print(f"  per-method ACF lag-1: ML={ml_acf[1]:.3f} (bl={ml_dl}), UWB={uwb_acf[1]:.3f} (bl={uwb_dl})")

		# Paired MBB CI (Bonferroni m=2: 97.5% confidence)
		result = util.paired_accuracy_block_bootstrap(
			ml_seq.astype(float), np.ones(len(ml_seq)),
			uwb_seq.astype(float), np.ones(len(ml_seq)),
			block_length=bl, confidence=0.975)
		pdiff_val, lo, hi, bl_used = result
		print(f"  MBB CI (block={bl_used}): [{lo:.4f}, {hi:.4f}] → {'SIG' if lo > 0 or hi < 0 else 'NOT SIG'}")

	print(f"\nPaired data saved to: {paired_path}")

	# Paired MBB CI for azimuth errors (ML vs UWB at shared trilat positions)
	# EKF vs UWB uses standalone 0.5s-aggregated UWB data matched by nearest GT timestamp
	print("\n" + "=" * 70)
	print("PAIRED MBB CI FOR AZIMUTH ERROR DIFFERENCES")
	print("=" * 70)
	if 'UWB only' in per_method_azimuth and len(per_method_azimuth['UWB only']) > 0:
		uwb_az = np.array(per_method_azimuth['UWB only'])
		uwb_ts = np.array(per_method_azimuth_ts['UWB only'])
		for mkey, mlabel in [('trilateration', 'Multilateration'), ('ekf', 'EKF')]:
			if mkey not in per_method_azimuth or len(per_method_azimuth[mkey]) == 0:
				continue
			ml_az = np.array(per_method_azimuth[mkey])
			ml_ts = np.array(per_method_azimuth_ts[mkey])
			if mkey == 'trilateration':
				# Both at trilat positions — naturally paired (same positions)
				n_shared = min(len(ml_az), len(uwb_az))
				ml_paired = ml_az[:n_shared]
				uwb_paired = uwb_az[:n_shared]
			else:
				# EKF vs UWB: load standalone 0.5s-aggregated UWB per-position data
				standalone_path = os.path.join("stats", "uwb_standalone_labels.npz")
				if not os.path.exists(standalone_path):
					continue
				standalone = np.load(standalone_path)
				uwb_standalone_az = standalone['az_per_pos']
				uwb_standalone_ts = standalone['c_at_ts']
				# Match each EKF position to nearest standalone UWB position by GT timestamp
				idx = np.searchsorted(uwb_standalone_ts, ml_ts)
				idx = np.clip(idx, 1, len(uwb_standalone_ts) - 1)
				dl = np.abs(ml_ts - uwb_standalone_ts[idx - 1])
				dr = np.abs(ml_ts - uwb_standalone_ts[idx])
				use_left = dl <= dr
				idx[use_left] = idx[use_left] - 1
				uwb_paired = uwb_standalone_az[idx]
				ml_paired = ml_az
			diff_seq = uwb_paired - ml_paired  # positive = UWB worse
			mean_diff = np.mean(diff_seq)
			a_az = util._compute_acf(diff_seq, max_lag=60)
			dl_az = util._decorrelation_lag(a_az)
			bl_az = max(1, int(dl_az))
			n_shared = len(diff_seq)
			nblk = max(2, int(np.ceil(n_shared / bl_az)))
			rng = np.random.default_rng(42)
			means_az = np.array([np.mean(np.concatenate(
				[diff_seq[i:i+bl_az] for i in rng.integers(0, max(1, n_shared - bl_az + 1), size=nblk)]
			)[:n_shared]) for _ in range(2000)])
			lo_az = float(np.quantile(means_az, 0.0125))  # 98.75% for Bonferroni m=2
			hi_az = float(np.quantile(means_az, 0.9875))
			print(f"\n{mlabel} vs UWB azimuth error (paired, shared positions):")
			print(f"  n={n_shared}, ML mean az={np.mean(ml_paired):.2f}, UWB mean az={np.mean(uwb_paired):.2f}")
			print(f"  diff ACF lag-1={a_az[1]:.3f}, block_length={bl_az}")
			print(f"  point diff={mean_diff:.2f} deg (UWB - ML)")
			print(f"  Bonferroni MBB CI (97.5%): [{lo_az:.2f}, {hi_az:.2f}] deg → {'SIG' if lo_az > 0 or hi_az < 0 else 'NOT SIG'}")

	return (distance_errors, motion_azimuth_errors, position_errors, stats_dfs, dir_accs, 
	        true_labels, predicted_labels, per_method_azimuth, per_method_azimuth_ts,
	        predicted_timestamps)

def save_shops(shop_locations, shops_filename):
	df = pd.DataFrame(shop_locations).T
	df["Shop"] = df.index 
	df = df.reset_index(drop=True)
	df.columns = ["X", "Y", "Shop"]
	df.to_csv(shops_filename, index=False)

def load_shops(shops_filename):
	df = pd.read_csv(shops_filename)
	shops = {}
	for i, r in df.iterrows():
		shops[r["Shop"]] = np.array([r["X"], r["Y"]])
	return shops 

def azimuth2direction(iphone_paths, time_interval, shop_locations):			
	correct_directions = num_intervals = num_shops_detected = all_possible_shops = 0
	angle_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
	distance_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
	direction_counts = {"front": {"correct": 0, "total": 0}, "back": {"correct": 0, "total": 0}, "left": {"correct": 0, "total": 0}, "right": {"correct": 0, "total": 0}}
	vector_errors = []
	vector_errors2 = []
	broken_count = total_broken_count = 0
	for path_id in iphone_paths:
		# average all devices
		#for shop_id in iphone_paths[path_id]:
		if "consolidated" not in iphone_paths[path_id]:
			consolidate_shop_measurements_by_timestamp(iphone_paths, "unfiltered")
			consolidate_shop_measurements_by_timestamp(iphone_paths, "filtered")
		for data_type in iphone_paths[path_id]["consolidated"]:
			data = iphone_paths[path_id]["consolidated"][data_type] # it's an array of dicts
			
			if len(data) > 0 and data_type == "unfiltered":
				i = 0
				while i < len(data):
					temp_dist = {}
					start_idx = curr_idx = i
					while curr_idx < len(data) and (data[curr_idx]["timestamp"] - data[start_idx]["timestamp"]) < time_interval:
						curr_shop_id = data[curr_idx]["shop"]#data[curr_ts]["shop"]
						if curr_shop_id not in temp_dist:
							temp_dist[curr_shop_id] = {"distance": [], "azimuth": [], "actual azimuth": [], "actual distance": [], "compass value": [], "actual compass value": [], "actual orientation": [], "actual position": [], "timestamp": [], "actual timestamp": []}
						
						temp_dist[curr_shop_id]["distance"].append(data[curr_idx]["distance"])
						temp_dist[curr_shop_id]["actual distance"].append(data[curr_idx]["actual distance"])
						temp_dist[curr_shop_id]["actual azimuth"].append(data[curr_idx]["actual azimuth"])
						# temp_dist[curr_shop_id]["compass value"].append(data[curr_idx]["compass value"])
						# temp_dist[curr_shop_id]["actual compass value"].append(data[curr_idx]["actual compass value"])
						temp_dist[curr_shop_id]["actual orientation"].append(data[curr_idx]["actual orientation"])
						temp_dist[curr_shop_id]["actual position"].append(data[curr_idx]["actual position"])
						temp_dist[curr_shop_id]["timestamp"].append(data[curr_idx]["timestamp"])
						temp_dist[curr_shop_id]["actual timestamp"].append(data[curr_idx]["actual timestamp"])
						if data[curr_idx]["azimuth"] != "nil" and not np.isnan(data[curr_idx]["azimuth"]):
							temp_dist[curr_shop_id]["azimuth"].append(data[curr_idx]["azimuth"])
							
						curr_idx += 1
						#print("curr_idx:", curr_idx)
					
					assert temp_dist[curr_shop_id]["timestamp"][-1] - temp_dist[curr_shop_id]["timestamp"][0] <= time_interval

					angles = distances = 0
					for curr_shop_id in temp_dist:
						temp = np.array(temp_dist[curr_shop_id]["azimuth"])
						# if temp.dtype != np.float32:
						# 	print(temp.dtype, path_id)
						assert (temp.dtype == np.float32) or (temp.dtype == np.float64)
						temp = temp[temp != util.DUMMY]
						temp_dist[curr_shop_id]["azimuth"] = temp[~np.isnan(temp)]

						if len(temp_dist[curr_shop_id]["azimuth"]) > 0:
							#for key in temp_dist[curr_shop_id]:

							#print("nan angles:", len(temp[np.isnan(temp)]))
							curr_dist = np.array(temp_dist[curr_shop_id]["distance"]).mean(axis=0)
							curr_ori_theta = np.array(temp_dist[curr_shop_id]["actual orientation"]).mean(axis=0)
							#_val = np.array(temp_dist[curr_shop_id]["compass value"]).mean(axis=0)
							curr_azimuth = np.array(temp_dist[curr_shop_id]["azimuth"]).mean()
							curr_act_dist = np.array(temp_dist[curr_shop_id]["actual distance"]).mean(axis=0)
							#curr_act_compass_val = np.array(temp_dist[curr_shop_id]["actual compass value"]).mean(axis=0)
							curr_act_azimuth = np.array(temp_dist[curr_shop_id]["actual azimuth"]).mean(axis=0)
							# measured_pos = util.calculate_position(curr_compass_val, curr_azimuth, curr_dist)
							# measured_pos_with_act_ori = util.calculate_position(curr_act_compass_val, curr_azimuth, curr_dist)
							# actual_pos = util.calculate_position(curr_act_compass_val, curr_act_azimuth, curr_act_dist)
							# vector_errors.append(util.calculate_distance_between_vectors(measured_pos.reshape(-1, 1), actual_pos.reshape(-1, 1)))
							# vector_errors2.append(util.calculate_distance_between_vectors(measured_pos_with_act_ori.reshape(-1, 1), actual_pos.reshape(-1, 1)))
							
							util.discretize_azimuths(curr_azimuth, curr_act_azimuth, direction_counts)
							# calculate the shop's position from the phone's perspective
							all_azimuths = util.get_azimuths_to_other_shops(np.degrees(curr_ori_theta), curr_dist, curr_azimuth, shop_locations, curr_shop_id)
							all_act_azimuths = util.get_azimuths_to_other_shops(np.degrees(curr_ori_theta), curr_act_dist, curr_act_azimuth, shop_locations, curr_shop_id)
							
							# if abs(all_azimuths[curr_shop_id] - curr_azimuth) > 1e-3:
							# 	print(all_azimuths[curr_shop_id], curr_azimuth)
							# assert abs(all_azimuths[curr_shop_id] - curr_azimuth) < 1e-3

							# broken = False 	
							# for other_shop_id in all_azimuths:
							# 	#correct_directions += helper(all_azimuths[other_shop_id], all_act_azimuths[other_shop_id])
							# 	all_possible_shops += 1
							
							# 	# if i == 0:
							# 	# 	if curr_shop_id == other_shop_id:
							# 	# 		print("measured azimuth:", all_azimuths[curr_shop_id], "actual azimuth:", all_act_azimuths[curr_shop_id])
									
							# 	if other_shop_id in temp_dist:
							# 		if abs(all_act_azimuths[other_shop_id] - np.array(temp_dist[other_shop_id]["actual azimuth"]).mean()) > 6:#1e-3:
							# 			print("source shop:", curr_shop_id, "actual azimuth of source shop:", curr_act_azimuth, "actual distance of source shop:", curr_act_dist, other_shop_id, "transformed azimuth:", all_act_azimuths[other_shop_id], "actual azimuth:", np.array(temp_dist[other_shop_id]["actual azimuth"]).mean(), "actual distance:", np.array(temp_dist[other_shop_id]["actual distance"]).mean(), "timestamp:", np.array(temp_dist[other_shop_id]["timestamp"]).mean(), "actual timestamp:", np.array(temp_dist[other_shop_id]["actual timestamp"]).mean())
							# 			print("all averaged azimuths of this shop:", np.array(temp_dist[curr_shop_id]["actual azimuth"]))
							# 			print("all averaged orientations of this shop:", np.array(temp_dist[curr_shop_id]["actual orientation"]))
							# 			print("all averaged azimuths of other shop:", np.array(temp_dist[other_shop_id]["actual azimuth"]))
							# 			print("all averaged orientations of other shop:", np.array(temp_dist[other_shop_id]["actual orientation"]))
							# 			broken = True 
							# 			broken_count += 1
							# 			act_pos = np.array(temp_dist[other_shop_id]["actual position"]).mean(axis=0)
							# 			act_ori = np.array(temp_dist[other_shop_id]["actual orientation"]).mean()
							# 			for temp_shop in shop_locations:
							# 				plt.scatter(shop_locations[temp_shop][0], shop_locations[temp_shop][1], color="red")
							# 				plt.annotate(temp_shop + " (a)", shop_locations[temp_shop], color="red")
										
							# 			plt.scatter(act_pos[0], act_pos[1], color="green")
							# 			plt.quiver(act_pos[0], act_pos[1], np.cos(act_ori), np.sin(act_ori), color="green")
							# 			plt.annotate("T", (act_pos[0], act_pos[1]), color="green")
										
							# 			ta = util.get_vector(curr_act_dist, 90 - curr_act_azimuth - util.theta_to_bearing(np.degrees(curr_ori_theta)))
							# 			print("orientation:", np.degrees(curr_ori_theta), "azimuth:", curr_act_azimuth, "bearing:", util.theta_to_bearing(np.degrees(curr_ori_theta)))
							# 			plt.quiver(act_pos[0], act_pos[1], ta[0], ta[1], color="purple")
							# 		# 	#plt.legend()
									
							# 		total_broken_count += 1 # unindent later
							# if broken:
							# 	plt.show()
							# 	plt.show()
							# if i == 0:
							# 	plt.legend()
							# 	plt.show()
							angles += 1

						if len(temp_dist[curr_shop_id]["distance"]) > 0:
							distances += 1
						
						num_shops_detected += 1
					
					if angles > 0:
						angle_counts[angles] += 1
					
					if distances > 0:
						distance_counts[distances] += 1
					num_intervals += 1
					i += 100
					#print("i:", i, "len data:", len(data))
	
	print("Number of intervals:", num_intervals, "num shops detected:", num_shops_detected, "angle counts:", angle_counts, "distance counts:", distance_counts, "all possible shops:", all_possible_shops, "correct dirs:", correct_directions, "broken:", broken_count, "total broken count:", total_broken_count)#, "%:", 100 * correct_directions / num_intervals)
	
	print("direction counts:", direction_counts)
	distance_counts = np.array([(k, v) for (k, v) in distance_counts.items()])
	angle_counts = np.array([(k, v) for (k, v) in angle_counts.items()])
	angle_counts = pd.DataFrame(angle_counts, columns=["Number of chips", "Frequency"])
	angle_counts["Type"] = "Angle"
	distance_counts = pd.DataFrame(distance_counts, columns=["Number of chips", "Frequency"])
	distance_counts["Type"] = "Distance"
	
	counts = pd.concat([angle_counts, distance_counts], axis=0)
	percents = {"Distance": {}, "Angle": {}}
	for num_chips in counts["Number of chips"].unique():
		for feat in ["Distance", "Angle"]:
			percents[feat][num_chips] = 100 * counts.loc[(counts["Number of chips"] == num_chips) & (counts["Type"] == feat), "Frequency"].values / counts["Frequency"].sum()
	
	print(percents)
	for feat in percents:
		temp = percents[feat]
		temp = np.array(list(temp.items()))
		print(temp)
		labels = [f"{temp[i, 0]} {'sensor' if temp[i, 0] == 1 else 'sensors'} ({round(temp[i, 1][0], 1)}%)" for i in range(len(temp))]
		plt.pie(temp[:, 1], labels=labels, colors=sns.color_palette('viridis'))
		plt.show()

	direction_counts = np.array([(k, round(100 * v["correct"] / v["total"], 2)) for k, v in direction_counts.items()])
	direction_counts = pd.DataFrame(direction_counts, columns=["Direction", "Accuracy"])
	sns.barplot(x=direction_counts["Direction"], y=direction_counts["Accuracy"], palette="cool")
	plt.ylabel("Accuracy (%)")
	plt.xlabel("Direction")
	plt.show()

def dist_angle_dist(iphone_paths, cumul, bins):
	distances = []
	azimuths = []
	unfiltered_num_records = filtered_num_records = 0
	for path_id in iphone_paths:
		for method in iphone_paths[path_id]:
			if method[0] != "S":
				continue 
			
			if "filtered" in iphone_paths[path_id][method]:
				distances.append(iphone_paths[path_id][method]["filtered"]["time-aligned actual distances"])
				azimuths.append(iphone_paths[path_id][method]["filtered"]["time-aligned actual azimuths"])
				
				#unfiltered_num_records += len(distances)
				filtered_num_records += len(iphone_paths[path_id][method]["filtered"]["time-aligned actual azimuths"])
	distances = np.concatenate(distances, axis=0)
	azimuths = np.concatenate(azimuths, axis=0)

	print("min distance:", np.min(distances), "max distance:", np.max(distances))
	print("min angle:", np.min(azimuths), "max angle:", np.max(azimuths))
	print("number of filtered records:", filtered_num_records)
	util.plot_histogram(distances, "Distance to anchors (m)", "CDF" if cumul else "Density", label=None, ax=None, line=False, color=None, show=True, marker=None, cumul=cumul, bins=bins)
	util.plot_histogram(azimuths, "Relative angle to anchors (degrees)", "CDF" if cumul else "Density", label=None, ax=None, line=False, color=None, show=True, marker=None, cumul=cumul, bins=bins)

def generate_vectors(iphone_paths, time_interval, shop_locations):
	def update_direction_counts(measured_azimuth, actual_azimuth, direction_counts):
		#correct_directions = 0
		if actual_azimuth > 90:
			if measured_azimuth > 90:
				direction_counts["back"]["correct"] += 1
			direction_counts["back"]["total"] += 1
		elif actual_azimuth < -90:
			if measured_azimuth < -90:
				direction_counts["back"]["correct"] += 1
			direction_counts["back"]["total"] += 1
		elif actual_azimuth > 10 and actual_azimuth < 90:
			if measured_azimuth > 10 and measured_azimuth < 90:
				direction_counts["right"]["correct"] += 1
			direction_counts["right"]["total"] += 1
		elif actual_azimuth < -10 and actual_azimuth > -90:
			if measured_azimuth < -10 and measured_azimuth > -90:
				direction_counts["left"]["correct"] += 1
			direction_counts["left"]["total"] += 1
		elif actual_azimuth > -10 and actual_azimuth < 10:
			if measured_azimuth > -10 and measured_azimuth < 10:
				direction_counts["front"]["correct"] += 1
			direction_counts["front"]["total"] += 1
	
	def generate(curr_shop_id, distance, azimuth, orientation, shop_locations, true_pos):
		vectors = {}
		# construct the vectors first 
		ta = util.get_vector(distance, 90 - azimuth - util.theta_to_bearing(np.degrees(orientation)))
		target_ori_vector = util.get_vector(1, np.degrees(orientation))
		ot = shop_locations[curr_shop_id] - ta
		vectors["O"] = {"T": ot}
		vectors["T"] = {curr_shop_id: ta, "O": -ot}
		vectors[curr_shop_id] = {"T": -ta}
		
		for other_shop_id in shop_locations:
			ta_i = shop_locations[other_shop_id] - ot
			vectors["T"][other_shop_id] = ta_i 
			vectors[other_shop_id] = {"T": -ta_i}
			vectors["O"][other_shop_id] = shop_locations[other_shop_id]
			vectors[other_shop_id]["O"] = -shop_locations[other_shop_id]

		for s1 in shop_locations:
			for s2 in shop_locations:
				if s1 == s2:
					continue 
				vectors[s1][s2] = vectors["O"][s2] - vectors["O"][s1]
				vectors[s2][s1] = vectors["O"][s1] - vectors["O"][s2]
		# generate vectors 
		generated_vectors = {}
		for p1 in vectors:
			for p2 in vectors:
				if p1 == p2:
					continue 
				if p1 not in generated_vectors:
					generated_vectors[p1] = {}
				if p2 not in generated_vectors[p1]:
					generated_vectors[p1][p2] = [vectors[p1][p2]]

				for p3 in vectors:
					if p2 == p3 or p1 == p3:
						continue
					
					for p4 in vectors:
						if p3 == p4 or p1 == p4 or p2 == p4:
							continue
						if p3 in vectors[p1] and p4 in vectors[p3] and p2 in vectors[p4]:
							
							generated_vectors[p1][p2].append(vectors[p1][p3] + vectors[p3][p4] + vectors[p4][p2])

				# if p1 == "O":
				# 	print(p2)
				# if p1 == "O" and p2 == "T":
				# 	print(p1, p2)
				# 	print("measured vector:", vectors[p1][p2])
				# 	print("generated vectors:", np.array(generated_vectors[p1][p2]))
				# 	print("=>", np.array(generated_vectors[p1][p2]).mean(axis=0))
					
		for p1 in generated_vectors:
			for p2 in generated_vectors[p1]:
				generated_vectors[p1][p2] = np.array(generated_vectors[p1][p2]).mean(axis=0)
		
		# get azimuths 
		azimuths = {}
		for shop_id in shop_locations:
			val = util.calculate_azimuths(target_ori_vector.reshape(1, 2), np.array(generated_vectors["T"][shop_id]).reshape(1, 2))
			azimuths[shop_id] = val[0]
		
		shop_ids = list(shop_locations.keys())
		coords = [shop_locations[shop_id] for shop_id in shop_ids]
		coords.append(ot)
		shop_ids.append("T")
		mv, gv = np.zeros((len(coords), len(coords), 2)), np.zeros((len(coords), len(coords), 2))
		
		for i in range(len(shop_ids)):
			for j in range(len(shop_ids)):
				s1 = shop_ids[i]
				s2 = shop_ids[j]
				if i != j:
					#print(generated_vectors[s1][s2])
					mv[i][j] = vectors[s1][s2]
					gv[i][j] = generated_vectors[s1][s2]

		# plt.scatter(true_pos[0], true_pos[1], color="green")
		# plt.scatter(ot[0], ot[1], color="blue")
		# plt.annotate("T (a)", true_pos, color="green")
		# plt.annotate("T (m)", ot, color="blue")
		# util.draw_vectors(np.array(coords), mv, "Measured", show=True)
		# plt.scatter(true_pos[0], true_pos[1], color="green")
		# plt.scatter(ot[0], ot[1], color="blue")
		# plt.annotate("T (a)", true_pos, color="green")
		# plt.annotate("T (m)", ot, color="blue")
		# util.draw_vectors(np.array(coords), gv, "Generated", show=True)
		
		return azimuths 
	
	direction_counts = {"front": {"correct": 0, "total": 0}, "back": {"correct": 0, "total": 0}, "left": {"correct": 0, "total": 0}, "right": {"correct": 0, "total": 0}}
	other_direction_counts = {"front": {"correct": 0, "total": 0}, "back": {"correct": 0, "total": 0}, "left": {"correct": 0, "total": 0}, "right": {"correct": 0, "total": 0}}
	#gen_direction_counts = {"front": {"correct": 0, "total": 0}, "back": {"correct": 0, "total": 0}, "left": {"correct": 0, "total": 0}, "right": {"correct": 0, "total": 0}}
							
	for path_id in iphone_paths:
		# average all devices
		#for shop_id in iphone_paths[path_id]:
		if "consolidated" not in iphone_paths[path_id]:
			consolidate_shop_measurements_by_timestamp(iphone_paths, "unfiltered")
			
		data = iphone_paths[path_id]["consolidated"]["unfiltered"] # it's an array of dicts
		
		if len(data) == 0:
			continue
		
		i = 0
		while i < len(data):
			temp_dist = {}
			start_idx = curr_idx = i
			while curr_idx < len(data) and (data[curr_idx]["timestamp"] - data[start_idx]["timestamp"]) < time_interval:
				curr_shop_id = data[curr_idx]["shop"]#data[curr_ts]["shop"]
				if curr_shop_id not in temp_dist:
					temp_dist[curr_shop_id] = {"distance": [], "azimuth": [], "actual azimuth": [], "actual distance": [], "compass value": [], "actual compass value": [], "actual orientation": [], "actual position": [], "timestamp": [], "actual timestamp": []}
				
				temp_dist[curr_shop_id]["distance"].append(data[curr_idx]["distance"])
				temp_dist[curr_shop_id]["actual distance"].append(data[curr_idx]["actual distance"])
				temp_dist[curr_shop_id]["actual azimuth"].append(data[curr_idx]["actual azimuth"])
				# temp_dist[curr_shop_id]["compass value"].append(data[curr_idx]["compass value"])
				# temp_dist[curr_shop_id]["actual compass value"].append(data[curr_idx]["actual compass value"])
				temp_dist[curr_shop_id]["actual orientation"].append(data[curr_idx]["actual orientation"])
				temp_dist[curr_shop_id]["actual position"].append(data[curr_idx]["actual position"])
				temp_dist[curr_shop_id]["timestamp"].append(data[curr_idx]["timestamp"])
				temp_dist[curr_shop_id]["actual timestamp"].append(data[curr_idx]["actual timestamp"])
				if data[curr_idx]["azimuth"] != "nil" and not np.isnan(data[curr_idx]["azimuth"]):
					temp_dist[curr_shop_id]["azimuth"].append(data[curr_idx]["azimuth"])
					
				curr_idx += 1
				#print("curr_idx:", curr_idx)
			
			assert temp_dist[curr_shop_id]["timestamp"][-1] - temp_dist[curr_shop_id]["timestamp"][0] <= time_interval

			for curr_shop_id in temp_dist:
				temp = np.array(temp_dist[curr_shop_id]["azimuth"])
				
				assert (temp.dtype == np.float32) or (temp.dtype == np.float64)
				temp = temp[temp != util.DUMMY]
				temp_dist[curr_shop_id]["azimuth"] = temp[~np.isnan(temp)]

				if len(temp_dist[curr_shop_id]["azimuth"]) > 0:
					curr_dist = np.array(temp_dist[curr_shop_id]["distance"]).mean(axis=0)
					curr_ori_theta = np.array(temp_dist[curr_shop_id]["actual orientation"]).mean(axis=0)
					#curr_compass_val = np.array(temp_dist[curr_shop_id]["compass value"]).mean(axis=0)
					curr_azimuth = np.array(temp_dist[curr_shop_id]["azimuth"]).mean()
					curr_act_dist = np.array(temp_dist[curr_shop_id]["actual distance"]).mean(axis=0)
					#curr_act_compass_val = np.array(temp_dist[curr_shop_id]["actual compass value"]).mean(axis=0)
					curr_act_azimuth = np.array(temp_dist[curr_shop_id]["actual azimuth"]).mean(axis=0)
					#actual_pos = util.calculate_position(curr_act_compass_val, curr_act_azimuth, curr_act_dist)
					
					#helper(curr_azimuth, curr_act_azimuth, direction_counts)
					# calculate the shop's position from the phone's perspective
					all_azimuths = util.get_azimuths_to_other_shops(np.degrees(curr_ori_theta), curr_dist, curr_azimuth, shop_locations, curr_shop_id)
					all_act_azimuths = util.get_azimuths_to_other_shops(np.degrees(curr_ori_theta), curr_act_dist, curr_act_azimuth, shop_locations, curr_shop_id)
					assert curr_shop_id in all_azimuths

					for other_shop_id in all_azimuths:
						if other_shop_id == curr_shop_id:
							update_direction_counts(all_azimuths[other_shop_id], all_act_azimuths[other_shop_id], direction_counts)
						else:
							update_direction_counts(all_azimuths[other_shop_id], all_act_azimuths[other_shop_id], other_direction_counts)
					
					# gen_azimuths = generate(curr_shop_id, curr_dist, curr_azimuth, curr_ori_theta, shop_locations, actual_pos)
					# for other_shop_id in gen_azimuths:
					# 	update_direction_counts(gen_azimuths[other_shop_id], all_act_azimuths[other_shop_id], gen_direction_counts)
					
			i += 100
			
	print("direction counts:", direction_counts)
	print("generated direction counts:", other_direction_counts)
	
	direction_counts = np.array([(k, round(100 * v["correct"] / v["total"], 2)) for k, v in direction_counts.items()])
	direction_counts = pd.DataFrame(direction_counts, columns=["Direction", "Accuracy"])
	direction_counts["Type"] = "Measured"
	other_direction_counts = np.array([(k, round(100 * v["correct"] / v["total"], 2)) for k, v in other_direction_counts.items()])
	other_direction_counts = pd.DataFrame(other_direction_counts, columns=["Direction", "Accuracy"])
	other_direction_counts["Type"] = "Generated"
	counts = pd.concat([direction_counts, other_direction_counts], axis=0).reset_index(drop=True)
	counts["Accuracy"] = counts["Accuracy"].astype(float)
	sns.barplot(x=counts["Direction"], y=counts["Accuracy"], hue=counts["Type"])
	plt.ylabel("Accuracy (%)")
	plt.xlabel("Direction")
	plt.show()

def fill_nil_azimuths(df):
	df = df.copy().reset_index()
	shift = 1
	curr_az, curr_ts = None, None  
	for i in range(1, df.shape[0]):
		# if i == len(df) - shift:
		#     break
		prev, curr = df.iloc[max(i - shift, 0), :], df.iloc[i, :]
		if curr["Azimuth"] == util.DUMMY:
			if prev["Azimuth"] != util.DUMMY:
				curr_az, curr_ts = prev["Azimuth"], prev["Timestamp"]
			elif curr_az != None and curr["Timestamp"] - curr_ts < 5:
				df.loc[i, "Azimuth"] = curr_az
	return df["Azimuth"]

def filter_iphone_measurements(df, drop_risky_measured_azimuths, drop_risky_true_azimuths, drop_risky_true_distances, fill_nill=False, drop_risky_elevations=True):
	# filter the df 
	df["Azimuth"] = azimuths2float(df["Azimuth"])

	if fill_nill:
		df["Azimuth"] = fill_nil_azimuths(df)
		#print("after fill nill:", df["Azimuth"])

	df = df[df["Azimuth"] != util.DUMMY]
	if (util.DUMMY in df["Azimuth"]):
		print(df["Azimuth"].sort_values())

	if drop_risky_measured_azimuths:
		df = df[abs(df["Azimuth"]) <= constants.RISKY_AZIMUTH_THRESHOLD]
	
	if drop_risky_true_azimuths:
		df = df[abs(df["Actual Azimuth"]) <= constants.RISKY_AZIMUTH_THRESHOLD]
	
	if drop_risky_true_distances:
		df = df[abs(df["Distance"] - df["Actual Distance"]) < constants.RISKY_DISTANCE_THRESHOLD]

	if drop_risky_elevations:
		to_del = []
		for i, r in df.iterrows():
			if r["Elevation"] == "nil" or float(r["Elevation"]) > constants.RISKY_ELEVATION_THRESHOLD:
				to_del.append(i)
			
		df = df.drop(to_del)
	return df 

def iphone_data_to_coordinates(
		original_df, 
		shop_location, 
		check_vectors_for_sanity, 
		check_abrupt_angle_changes, 
		filter_invalid_angles, 
		drop_risky_measured_azimuths, 
		drop_risky_true_azimuths, 
		drop_risky_elevations,
		drop_risky_true_distances
	):
	df = original_df.copy()
	if filter_invalid_angles:
		df = filter_iphone_measurements(df, drop_risky_measured_azimuths=drop_risky_measured_azimuths, drop_risky_true_azimuths=drop_risky_true_azimuths, drop_risky_true_distances=drop_risky_true_distances, drop_risky_elevations=drop_risky_elevations)
	df["DeviceTheta"] = util.bearing_to_theta(df["DeviceBearing"].values).astype(float)
	df["DeviceOrientationX"] = np.cos(util.degrees2radians(df["DeviceTheta"]))
	df["DeviceOrientationY"] = np.sin(util.degrees2radians(df["DeviceTheta"]))
	df["VectorBearing"] = (df["Azimuth"] + df["DeviceBearing"])
	df["VectorTheta"] = util.bearing_to_theta(df["VectorBearing"]) # convert to Cartesian angle from compass bearing
	df["Yaw"] = util.degrees2radians(df["VectorTheta"])
	df["TA_X"] = df["Distance"] * np.cos(df["Yaw"])
	df["TA_Y"] = df["Distance"] * np.sin(df["Yaw"])

	# drop vectors that are out of the device's range
	df["DeviceOrientation_TA_Angle"] = util.calculate_angles_between_vectors(df.loc[:, ["DeviceOrientationX", "DeviceOrientationY"]].values, df.loc[:, ["TA_X", "TA_Y"]].values)

	if check_vectors_for_sanity:
		df = df[df["DeviceOrientation_TA_Angle"] <= constants.MAX_ORIENTATION_TA_ANGLE].reset_index(drop=True)
		df = util.check_vector_sanity(df)

	if check_abrupt_angle_changes:
		df = util.drop_vectors_after_max(df)

	df["X"] = shop_location[0] - df["TA_X"]
	df["Y"] = shop_location[1] - df["TA_Y"]

	df["Azimuth error"] = abs(df["Azimuth"] - df["Actual Azimuth"])
	df["Distance error"] = abs(df["Distance"] - df["Actual Distance"])

	return df if df.shape[0] > 0 else None 

def trilateration(a, d, max_num_anchors):
	distances = []
	anchors = []
	shop_ids = sorted(list(d.keys()))
	
	is_batch2 = False 
	for shop_id in shop_ids[:min(max_num_anchors, len(d))]:
		distances.append(d[shop_id])
		anchors.append(a[shop_id])
		if "-2" in shop_id:
			is_batch2 = True

	if len(d) > 6:
		print("!!!!!FOUND MORE THAN 6 ANCHORS!!!!!", len(d), shop_ids)
	# Prepare the system of equations
	A = []
	b = []

	# Construct the linear equations by subtracting the first equation from the others
	for i in range(1, len(anchors)):
		x_diff = anchors[i][0] - anchors[0][0]
		y_diff = anchors[i][1] - anchors[0][1]
		d_diff = distances[0]**2 - distances[i]**2
		c_diff = anchors[i][0]**2 - anchors[0][0]**2 + anchors[i][1]**2 - anchors[0][1]**2
		
		A.append([2 * x_diff, 2 * y_diff])
		b.append(d_diff + c_diff)

	A = np.array(A)
	b = np.array(b)

	# Solve the system using the least squares method
	position, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
	return position 

def dilateration(md1, md2, ma1, ma2, shop1_coords, shop2_coords, verbose=False):
	errors = []
	errors2 = []
	estimated_oris11 = []
	estimated_oris12 = []
	estimated_oris21 = []
	estimated_oris22 = []
	intersection_ids = []
	c = 0

	if verbose:
		print("measured distance 1:", md1, "measured distance 2:", md2, "measured angle 1:", ma1, "measured angle 2:", ma2)
	
	intersections = get_two_circles_intersections(shop1_coords[0], shop1_coords[1], md1, shop2_coords[0], shop2_coords[1], md2)
			
	if len(intersections) == 0:
		if verbose:
			print("!!!!!!! -------- NO INTERSECTIONS ------- !!!!!!!!")
		errors.append(np.nan)
		errors2.append(np.nan)
		estimated_oris11.append(np.nan)
		estimated_oris12.append(np.nan)
		estimated_oris21.append(np.nan)
		estimated_oris22.append(np.nan)
		intersection_ids.append(np.nan)
		c += 1
		return None
	
	dist_between_inter = util.calculate_vector_length(intersections[0] - intersections[1])
	distance_to_i1 = util.calculate_vector_length(intersections[0])
	distance_to_i2 = util.calculate_vector_length(intersections[1])
	if verbose:
		print("\tintersection 1:", intersections[0], "intersection 2:", intersections[1], "ground truth: [0, 0]", "distance between intersections:", dist_between_inter)
		print("\tdistance to intersection 1:", distance_to_i1, "distance to intersection 2:", distance_to_i2)
		print("\tCorrect intersection:", 1 if distance_to_i1 < distance_to_i2 else 2)
	
	i1s1 = shop1_coords - intersections[0]
	i2s1 = shop1_coords - intersections[1]
	i1s2 = shop2_coords - intersections[0]
	i2s2 = shop2_coords - intersections[1]

	i1_ori1 = util.rotate_vector(i1s1.reshape(1, -1), -ma1).reshape(-1,)
	i1_ori2 = util.rotate_vector(i1s2.reshape(1, -1), -ma2).reshape(-1,)
	i2_ori1 = util.rotate_vector(i2s1.reshape(1, -1), -ma1).reshape(-1,)
	i2_ori2 = util.rotate_vector(i2s2.reshape(1, -1), -ma2).reshape(-1,)
	estimated_oris11.append(util.calculate_vector_theta(i1_ori1))
	estimated_oris12.append(util.calculate_vector_theta(i1_ori2))
	estimated_oris21.append(util.calculate_vector_theta(i2_ori1))
	estimated_oris22.append(util.calculate_vector_theta(i2_ori2))

	i1_error = util.calculate_vector_length(i1_ori1 - i1_ori2)
	i2_error = util.calculate_vector_length(i2_ori1 - i2_ori2)

	if not np.isnan(i1_error) and not np.isnan(i2_error):
		if i1_error > i2_error:
			if verbose:
				print(f"\t\t\tRESULT: intersection 2")
			errors.append(distance_to_i2)
			errors2.append(distance_to_i1)
			intersection_ids.append(2)
			return intersections[1]
		else:
			if verbose:
				print(f"\t\t\tRESULT: intersection 1")
			errors.append(distance_to_i1)
			errors2.append(distance_to_i2)
			intersection_ids.append(1)
			return intersections[0]
	else:
		if np.isnan(i1_error):
			print("intersection 1: couldn't calculate the angle between orientations")
		elif np.isnan(i2_error):
			print("intersection 2: couldn't calculate the angle between orientations")

		errors.append(np.nan)
		errors2.append(np.nan)
		return None 
	
	print("number of invalid cases:", c)

def get_two_circles_intersections(x1,y1,r1, x2,y2,r2):
	centerdx = x1 - x2
	centerdy = y1 - y2
	R = np.sqrt(centerdx * centerdx + centerdy * centerdy)
	if not (abs(r1 - r2) <= R and (R <= r1 + r2)): 
		return []
	
	R2 = R*R
	R4 = R2*R2
	a = (r1*r1 - r2*r2) / (2 * R2)
	r2r2 = (r1*r1 - r2*r2)
	try:
		c = np.sqrt(2 * (r1*r1 + r2*r2) / R2 - (r2r2 * r2r2) / R4 - 1)
	except:
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

# -------------------------- UWB Azimuth Matrix Prep ---------------------------

def estimate_azimuths_from_positions(iphone_paths, shop_locations):
	for path_id in iphone_paths: 
		anchor_list, timestamps, uwb_azimuths, uwb_mask, median_dt = prepare_uwb_azimuth_matrix(iphone_paths, path_id, shop_locations, method="trilateration")
		positions = iphone_paths[path_id]["trilateration"]["positions"] 
		true_azimuths = iphone_paths[path_id]["trilateration"]["true azimuths"] 
		res = orientation_fusion.run_full_azimuth_fusion(positions, median_dt, None, shop_locations, uwb_azimuths, uwb_mask, true_azimuths=true_azimuths)
		print(res)

def prepare_uwb_azimuth_matrix(iphone_paths, path_id, shop_locations, method: str = "trilateration"):
	"""
	Build matrices for UWB azimuth fusion.
	Returns (timestamps, positions, anchor_list, az_mat, mask_mat, true_rel_angles)
	- timestamps: (N,)
	- positions: (N,2) measured positions (method key under iphone_paths[path_id][method])
	- anchor_list: list[str] sorted anchor ids present in shop_locations
	- az_mat: (N,M) UWB relative angles to anchors (radians where possible)
	- mask_mat: (N,M) boolean availability mask
	- true_rel_angles: (N,M) ground-truth relative angles derived from true path if present; else None
	"""
	path = iphone_paths[path_id]
	if method not in path:
		raise KeyError(f"Path {path_id} has no method '{method}'")
	pdata = path[method]
	ts = pdata.get("timestamps")
	if ts is None:
		ts = pdata.get("timestamp")
	if ts is None:
		ts = []
	timestamps = np.asarray(ts, dtype=float)

	pos = pdata.get("positions")
	if pos is None:
		pos = []
	positions = np.asarray(pos, dtype=float)
	if positions.ndim != 2 or positions.shape[1] != 2:
		raise ValueError("positions must be (N,2)")

	# Collect anchors present in shop_locations; keep deterministic S-id order
	anchor_list = sorted([k for k in shop_locations.keys() if k.startswith('S')], key=lambda s: (int(s[1:]) if s[1:].isdigit() else 9999, s))
	M = len(anchor_list)
	N = positions.shape[0]
	az_mat = np.full((N, M), np.nan, dtype=float)
	mask_mat = np.zeros((N, M), dtype=bool)

	# Try measured UWB azimuths under each anchor id within pdata
	# Expect per-sample dict like pdata['azimuths'][i] = { 'S1': angle_deg, ... }
	per_sample_az = pdata.get("azimuths")
	if per_sample_az is not None and len(per_sample_az) == N:
		for i in range(N):
			row = per_sample_az[i]
			# Row may be a dict of anchor->angle or a mapping of anchor->sequence (N,)
			if isinstance(row, dict):
				for j, sid in enumerate(anchor_list):
					v = row.get(sid)
					if v is None:
						continue
					val = float(v)
					if abs(val) > np.pi + 0.1:
						val = np.deg2rad(val)
					az_mat[i, j] = val
					mask_mat[i, j] = True
			else:
				# Fallback: if per_sample_az is already a (N,M) ndarray aligned to anchor_list
				try:
					arr = np.asarray(per_sample_az, dtype=float)
					if arr.shape == (N, len(anchor_list)):
						az_mat[:, :] = arr
						mask_mat[:, :] = ~np.isnan(arr)
						break
				except Exception:
					pass

	# Optional: true relative angles
	# Priority 1: if method entry has 'true azimuths' per anchor, use them (likely degrees)
	true_rel_angles = None
	if isinstance(pdata.get("true azimuths"), dict) and N > 0:
		tru = pdata.get("true azimuths")
		arr = np.full((N, M), np.nan, dtype=float)
		for j, sid in enumerate(anchor_list):
			vals = tru.get(sid)
			if vals is None:
				continue
			v = np.asarray(vals, dtype=float)
			L = min(N, v.shape[0])
			arr[:L, j] = v[:L]
		# convert to radians if appears to be degrees
		deg_mask = np.abs(arr) > np.pi + 0.1
		arr[deg_mask] = np.deg2rad(arr[deg_mask])
		# wrap to [-pi, pi]
		true_rel_angles = (arr + np.pi) % (2*np.pi) - np.pi
	else:
		# Priority 2: derive from a 'true' path (positions/timestamps)
		if "true" in path and isinstance(path["true"], dict):
			tpos = np.asarray(path["true"].get("positions") or [], dtype=float)
			tt = np.asarray(path["true"].get("timestamps") or [], dtype=float)
			if len(tpos) >= 2 and len(tt) == len(tpos) and len(timestamps) == N:
				# align by closest indices
				idxs = util.find_closest_ground_truth_indices(tt, timestamps, by_measured=True)
				tpos_aligned = tpos[idxs]
				# compute true relative angles
				true_rel_angles = np.zeros((N, M), dtype=float)
				for i in range(N):
					pv = tpos_aligned[i]
					for j, sid in enumerate(anchor_list):
						av = np.asarray(shop_locations[sid][:2])
						bearing = np.arctan2(av[1]-pv[1], av[0]-pv[0])
						# motion heading from true positions
						if i > 0:
							dv = tpos_aligned[i] - tpos_aligned[i-1]
						else:
							dv = tpos_aligned[min(i+1, N-1)] - tpos_aligned[i]
						yaw = np.arctan2(dv[1], dv[0])
						true_rel_angles[i, j] = (bearing - yaw + np.pi) % (2*np.pi) - np.pi

	return timestamps, positions, anchor_list, az_mat, mask_mat, true_rel_angles

	"""Construct (uwb_azimuths, uwb_mask, median_dt) aligned to a consistent anchor order.

	Inputs
	- iphone_paths: dict produced by load_iphone_data/average_iphone_positions
	- path_id: key in iphone_paths, e.g., 'P26_green'
	- shop_locations: dict of anchor_id -> (x, y). Used to define consistent anchor ordering
	- method: only 'trilateration' currently supported (expects per-anchor azimuth series)

	Returns
	- anchor_list: ordered list of anchors (by numeric id)
	- timestamps: (N,) timestamps from the selected method
	- uwb_azimuths: (N, M) float array, device-frame relative azimuths (degrees as stored, NaN where missing)
	- uwb_mask: (N, M) bool array, True where measurement is present/finite
	- median_dt: scalar, median sampling interval from timestamps
	"""
	if path_id not in iphone_paths:
		raise KeyError(f"path_id {path_id} not in iphone_paths")
	if method not in iphone_paths[path_id]:
		raise KeyError(f"method '{method}' not found under iphone_paths[{path_id}]")

	entry = iphone_paths[path_id][method]
	if "timestamps" not in entry:
		raise KeyError(f"iphone_paths[{path_id}][{method}] has no 'timestamps'")
	if "azimuths" not in entry:
		raise KeyError(f"iphone_paths[{path_id}][{method}] has no per-anchor 'azimuths' dict")

	timestamps = np.asarray(entry["timestamps"], dtype=float)
	N = len(timestamps)
	# Consistent anchor order by numeric id (e.g., S1, S2, ... S12)
	def anchor_key(a):
		try:
			return int(str(a)[1:])
		except Exception:
			return str(a)
	anchor_list = sorted(list(shop_locations.keys()), key=anchor_key)
	M = len(anchor_list)
	
	entry_azimuths = {}

	for i in range(len(entry["azimuths"])):
		for shop_id in shop_locations:
			if shop_id not in entry_azimuths:
				entry_azimuths[shop_id] = []
			if shop_id not in entry["azimuths"][i]:
				entry_azimuths[shop_id].append(np.nan)
			else:
				entry_azimuths[shop_id].append(entry["azimuths"][i][shop_id])

	uwb_azimuths = np.full((N, M), np.nan, dtype=float)
	for j, aid in enumerate(anchor_list):
		series = entry_azimuths.get(aid, None)
		if series is None:
			continue
		vals = np.asarray(series, dtype=float)
		if vals.shape[0] != N:
			# be defensive: align by min length
			L = min(N, vals.shape[0])
			uwb_azimuths[:L, j] = vals[:L]
		else:
			uwb_azimuths[:, j] = vals

	uwb_mask = np.isfinite(uwb_azimuths)
	median_dt = float(np.median(np.diff(timestamps))) if N > 1 else 1.0
	return anchor_list, timestamps, uwb_azimuths, uwb_mask, median_dt
