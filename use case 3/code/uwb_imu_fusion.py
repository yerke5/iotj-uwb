import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from EKF_Fusion import EKF_Fusion_2D
from scipy.signal import butter, filtfilt
from scipy.stats import chi2
from typing import Dict, List, Tuple, Optional

def trilateration(anchors, distances):
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

	#print("A shape:", A.shape, "b shape:", b.shape)
	# Solve the system using the least squares method
	position, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
	return position 

def merge_uwb_imu_data(uwb_data, imu_data, uwb_rows):
    uwb_data = uwb_data.copy(deep=True)
    imu_data = imu_data.copy(deep=True)

    # Since gravity is already removed in the iPhone app and motion is purely horizontal,
    # we can use the accelerometer data directly but may want to apply some filtering
    # to reduce noise and vibrations
    
    # Optional: Apply light low-pass filter to reduce high-frequency noise and vibrations
    from scipy.signal import butter, filtfilt
    
    # Design low-pass filter to reduce noise (cutoff at 5 Hz)
    fs = 50  # Assumed sampling rate - adjust based on your data
    cutoff = 5  # Hz - this removes high-frequency vibrations while keeping motion
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(2, normalized_cutoff, btype='low')
    
    # Apply filter to reduce noise and vibrations
    imu_data['AccelerometerX'] = filtfilt(b, a, imu_data['AccelerometerX'])
    imu_data['AccelerometerZ'] = filtfilt(b, a, imu_data['AccelerometerZ'])
    
    # For horizontal-only motion, we can zero out Y-axis or use it as a quality check
    # Since motion should be horizontal, large Y accelerations indicate problems
    if 'AccelerometerY' not in imu_data.columns:
        imu_data['AccelerometerY'] = 0.0
    y_accel_std = imu_data['AccelerometerY'].std()
    #print(f"Y-axis acceleration std dev: {y_accel_std:.3f} m/s² (should be small for horizontal motion)")
    
    # You can choose to zero out Y or keep it for debugging
    # imu_data['AccelerometerY'] = 0  # Uncomment to force Y=0
    
    imu_data['type'] = 'imu'
    uwb_data['type'] = 'uwb'
    uwb_data['AccelerometerX'] = np.nan
    uwb_data['AccelerometerZ'] = np.nan
    uwb_data['GyroscopeY'] = np.nan
    # Ensure PositionX/PositionY columns exist for EKF updates
    for c in ['PositionX','PositionY']:
        if c not in uwb_data.columns:
            uwb_data[c] = np.nan
        if c not in imu_data.columns:
            imu_data[c] = np.nan
    for row in uwb_rows:
        imu_data[row] = np.nan

    first_uwb_row = uwb_data.iloc[0]

    base_cols = ['Timestamp', 'type', 'AccelerometerX', 'AccelerometerZ', 'GyroscopeY', 'PositionX', 'PositionY']
    all_cols = base_cols + [r for r in uwb_rows if r not in base_cols]
    all_data = pd.concat([
        imu_data[[c for c in all_cols if c in imu_data.columns]].assign(**{r: np.nan for r in uwb_rows if r not in imu_data.columns}),
        uwb_data[[c for c in all_cols if c in uwb_data.columns]].assign(**{r: uwb_data[r] if r in uwb_data.columns else np.nan for r in uwb_rows})
    ]).sort_values(by='Timestamp').reset_index(drop=True)

    # --- BEST PRACTICE: NORMALIZE TIMESTAMPS ---
    # This makes the timeline start at t=0 to prevent floating point issues.
    first_timestamp = all_data['Timestamp'].iloc[0]
    all_data['Timestamp'] = all_data['Timestamp'] #- first_timestamp
    # --- END NORMALIZATION ---

    all_data[['AccelerometerX', 'AccelerometerZ', 'GyroscopeY']] = all_data[['AccelerometerX', 'AccelerometerZ', 'GyroscopeY']].fillna(method='ffill')
    all_data.dropna(subset=['AccelerometerX'], inplace=True)

    # ==============================================================================
    # FIX: Start the fusion loop from the time of the first UWB measurement.
    # This prevents the filter from running on IMU data before its first position fix.
    # ==============================================================================
    first_uwb_timestamp = first_uwb_row['Timestamp'] #- first_timestamp # Use normalized timestamp
    all_data = all_data[all_data['Timestamp'] >= first_uwb_timestamp].reset_index(drop=True)
    # ==============================================================================

    return all_data

def _estimate_uwb_heading_rate(uwb_df):
    """Estimate heading and heading rate from UWB XY positions.
    Returns timestamps (t), heading (rad), and heading rate dpsi/dt aligned to UWB timestamps.
    """
    df = uwb_df.dropna(subset=['PositionX', 'PositionY']).copy()
    if len(df) < 3:
        return None, None, None
    xy = df[['PositionX', 'PositionY']].values.astype(float)
    t = df['Timestamp'].values.astype(float)
    dx = np.gradient(xy[:, 0], t)
    dy = np.gradient(xy[:, 1], t)
    psi = np.arctan2(dy, dx)
    psi_unwrapped = np.unwrap(psi)
    dpsi = np.gradient(psi_unwrapped, t)
    # light low-pass on dpsi to reduce noise
    if len(dpsi) >= 9:
        b, a = butter(2, 0.2)  # normalized cutoff ~0.2 Nyquist in UWB rate domain
        try:
            dpsi = filtfilt(b, a, dpsi)
        except Exception:
            pass
    return t, psi_unwrapped, dpsi

def calibrate_gyro_axis(all_data):
    """Pick gyro axis and sign that best matches UWB heading rate over a short window after first fix."""
    # build UWB-only dataframe
    uwb_only = all_data[all_data['type'] == 'uwb'][['Timestamp', 'PositionX', 'PositionY']].dropna()
    if len(uwb_only) < 5:
        return ('GyroscopeY', 1.0)  # default
    # limit to first few seconds to avoid long-run drift
    t0 = uwb_only['Timestamp'].iloc[0]
    uwb_cal = uwb_only[uwb_only['Timestamp'] <= t0 + 5.0]
    if len(uwb_cal) < 5:
        uwb_cal = uwb_only
    t_u, _, dpsi_u = _estimate_uwb_heading_rate(uwb_cal.rename(columns={'Timestamp': 'Timestamp', 'PositionX': 'PositionX', 'PositionY': 'PositionY'}))
    if dpsi_u is None:
        return ('GyroscopeY', 1.0)

    # interpolate UWB heading rate to IMU timestamps in same window
    imu_window = all_data[(all_data['Timestamp'] >= t_u[0]) & (all_data['Timestamp'] <= t_u[-1])]
    t_i = imu_window['Timestamp'].values.astype(float)
    if len(t_i) < 5:
        return ('GyroscopeY', 1.0)
    dpsi_i = np.interp(t_i, t_u, dpsi_u)

    candidates = ['GyroscopeX', 'GyroscopeY', 'GyroscopeZ']
    best = None
    best_corr = -np.inf
    corr_table = []
    for axis in candidates:
        if axis not in imu_window:
            continue
        g = imu_window[axis].values.astype(float)
        # basic outlier guard
        if not np.all(np.isfinite(g)):
            continue
        for sgn in (+1.0, -1.0):
            gg = sgn * g
            # remove NaNs
            mask = np.isfinite(gg) & np.isfinite(dpsi_i)
            if mask.sum() < 5:
                continue
            corr = np.corrcoef(gg[mask], dpsi_i[mask])[0, 1]
            if np.isnan(corr):
                continue
            corr_table.append((axis, int(sgn), float(corr)))
            if abs(corr) > best_corr:
                best_corr = abs(corr)
                best = (axis, np.sign(corr))  # prefer positive correlation
    if best is None:
        return ('GyroscopeY', 1.0)
    # Print a small summary to help manual verification
    try:
        print("Gyro axis calibration (axis, sign, corr):")
        for a, s, c in sorted(corr_table, key=lambda x: -abs(x[2])):
            print(f"  {a:>11}  {s:+d}  corr={c:+.3f}")
        print(f"Chosen axis: {best[0]} with sign {int(best[1])} (|corr|={best_corr:.3f})")
    except Exception:
        pass
    return best

def setup_ekf(uwb_data):
    # Initialize position from first UWB fix
    first_uwb_row = uwb_data.iloc[0]
    initial_pos = first_uwb_row[['PositionX', 'PositionY']].values.astype(float)

    # Estimate initial yaw from the first few positions if there is motion
    initial_yaw = 0.0
    if len(uwb_data) >= 3:
        p1 = uwb_data.iloc[0][['PositionX', 'PositionY']].values.astype(float)
        p2 = uwb_data.iloc[1][['PositionX', 'PositionY']].values.astype(float)
        p3 = uwb_data.iloc[2][['PositionX', 'PositionY']].values.astype(float)
        v1 = p2 - p1
        v2 = p3 - p2
        if np.linalg.norm(v1) > 0.05 and np.linalg.norm(v2) > 0.05:
            h1 = np.arctan2(v1[1], v1[0])
            h2 = np.arctan2(v2[1], v2[0])
            initial_yaw = np.arctan2(np.sin(h1) + np.sin(h2), np.cos(h1) + np.cos(h2))

    initial_state = np.hstack([initial_pos, [0.0, 0.0], initial_yaw])  # [px, py, vx, vy, yaw]

    # Conservative initial covariance
    initial_P = np.diag([0.5, 0.5, 1.0, 1.0, 1.0])

    # Tuned noise values (adjust to your hardware)
    r_uwb = 0.3   # meters std
    q_vel = 0.3   # per-second noise on velocity
    q_yaw = 0.15  # per-second yaw random walk
    q_bg = 1e-3   # gyro bias random walk

    ekf = EKF_Fusion_2D(initial_state, initial_P, q_vel=q_vel, q_yaw=q_yaw, r_uwb=r_uwb, q_bg=q_bg)
    return ekf, initial_state

def run_ekf(ekf, all_data, initial_state, smooth=True,
            use_ml_heading: bool = False,
            heading_column: Optional[str] = None,
            heading_noise_deg: float = 10.0,
            return_diagnostics: bool = False,
            return_state_history: bool = False):
    """Run fusion with dt capping/sub-steps, ZUPT, and chi-square gating. Optionally apply RTS smoothing."""
    states = []          # filtered states
    covs = []            # filtered covariances
    Fs = []              # transition matrices
    Qs = []              # process covariances

    states.append(np.hstack([initial_state, 0.0]))  # append bias for consistency
    covs.append(np.diag([0.5, 0.5, 1.0, 1.0, 1.0, 1.0]))

    innovation_diag = {
        "innovation_residual_x": [np.nan],
        "innovation_residual_y": [np.nan],
        "innovation_residual_norm": [np.nan],
        "predicted_position_x": [np.nan],
        "predicted_position_y": [np.nan],
        "uwb_measurement_x": [np.nan],
        "uwb_measurement_y": [np.nan],
    }

    last_timestamp = float(all_data['Timestamp'].iloc[0])
    last_uwb_update = float(all_data['Timestamp'].iloc[0])
    chi2_thresh = float(chi2.ppf(0.99, df=2))

    # Auto-calibrate gyro axis/sign from UWB heading-rate
    gyro_axis, gyro_sign = 'GyroscopeY', 1
    timestamps = []
    timestamps.append(last_timestamp)  # append initial timestamp to match initial state
    valid_idxs = [0]
    for i, row in all_data.iterrows():
        t = float(row['Timestamp'])
        dt = t - last_timestamp
        last_timestamp = t
        if dt <= 0:
            continue

        dt = min(dt, 0.2)
        n_steps = max(1, int(np.ceil(dt / 0.05)))
        dt_step = dt / n_steps

        accel_xy = row[['AccelerometerX', 'AccelerometerZ']].values.astype(float)
        gyro_val = row.get(gyro_axis, np.nan)
        try:
            gyro_z = float(gyro_val) * float(gyro_sign)
        except Exception:
            gyro_z = np.nan
        if np.any(~np.isfinite(accel_xy)) or not np.isfinite(gyro_z):
            continue

        F_last = None
        Q_last = None
        innov_x = np.nan
        innov_y = np.nan
        innov_norm = np.nan
        pred_x = np.nan
        pred_y = np.nan
        meas_x = np.nan
        meas_y = np.nan
        for _ in range(n_steps):
            F_last, Q_last = ekf.predict(accel_xy, gyro_z, dt_step)

        if np.linalg.norm(accel_xy) < 0.05 and abs(gyro_z) < 0.02:
            ekf.zero_velocity_update(sigma_v=0.03)

        if row['type'] == 'uwb' and np.isfinite(row['PositionX']) and np.isfinite(row['PositionY']):
            z = np.array([row['PositionX'], row['PositionY']], dtype=float)
            pred_xy = (ekf.H @ ekf.x).copy()
            y_innov = z - pred_xy
            innov_x = float(y_innov[0])
            innov_y = float(y_innov[1])
            innov_norm = float(np.linalg.norm(y_innov))
            pred_x = float(pred_xy[0])
            pred_y = float(pred_xy[1])
            meas_x = float(z[0])
            meas_y = float(z[1])
            if validate_position_update(ekf, z, chi2_thresh):
                ekf.update(z)
                last_uwb_update = t

        # Optional ML heading update (pseudo-measurement)
        if use_ml_heading and heading_column and heading_column in all_data.columns:
            try:
                zpsi = float(row[heading_column])
                if np.isfinite(zpsi):
                    ekf.update_heading(zpsi, np.deg2rad(float(heading_noise_deg)))
            except Exception:
                pass

        time_since_uwb = t - last_uwb_update
        if time_since_uwb > 3.0:
            ekf.q_vel = min(2.0, 0.3 + 0.2 * (time_since_uwb - 3.0))
            ekf.q_yaw = min(0.6, 0.15 + 0.05 * (time_since_uwb - 3.0))

        states.append(ekf.x.copy())
        covs.append(ekf.P.copy())
        Fs.append(F_last if F_last is not None else np.eye(6))
        Qs.append(Q_last if Q_last is not None else np.zeros((6, 6)))
        timestamps.append(t)
        valid_idxs.append(i)
        innovation_diag["innovation_residual_x"].append(innov_x)
        innovation_diag["innovation_residual_y"].append(innov_y)
        innovation_diag["innovation_residual_norm"].append(innov_norm)
        innovation_diag["predicted_position_x"].append(pred_x)
        innovation_diag["predicted_position_y"].append(pred_y)
        innovation_diag["uwb_measurement_x"].append(meas_x)
        innovation_diag["uwb_measurement_y"].append(meas_y)

        if np.any(np.abs(ekf.x[:2]) > 1e6):
            print("Filter diverged; stopping loop.")
            break


    Xf = np.vstack(states)
    if not smooth or len(Fs) < 2:
        state_history = Xf.copy()
        if return_diagnostics:
            if return_state_history:
                return Xf[:, :2], np.array(timestamps), np.array(valid_idxs), state_history, innovation_diag
            return Xf[:, :2], np.array(timestamps), np.array(valid_idxs), innovation_diag
        if return_state_history:
            return Xf[:, :2], np.array(timestamps), np.array(valid_idxs), state_history
        return Xf[:, :2], np.array(timestamps), np.array(valid_idxs)

    # RTS smoother
    n = len(Fs)
    Xs = Xf.copy()
    Ps = covs
    for k in range(n - 1, 0, -1):
        Pk = Ps[k]
        Fk = Fs[k - 1]
        Qk = Qs[k - 1]
        P_pred = Fk @ Ps[k - 1] @ Fk.T + Qk
        try:
            Ck = Ps[k - 1] @ Fk.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            Ck = np.zeros_like(Ps[k - 1])
        Xs[k - 1] = Xf[k - 1] + Ck @ (Xs[k] - (Fk @ Xf[k - 1]))
        Ps[k - 1] = Ps[k - 1] + Ck @ (Ps[k] - P_pred) @ Ck.T

    state_history = Xs.copy()
    if return_diagnostics:
        if return_state_history:
            return Xs[:, :2], np.array(timestamps), np.array(valid_idxs), state_history, innovation_diag
        return Xs[:, :2], np.array(timestamps), np.array(valid_idxs), innovation_diag
    if return_state_history:
        return Xs[:, :2], np.array(timestamps), np.array(valid_idxs), state_history
    return Xs[:, :2], np.array(timestamps), np.array(valid_idxs)

# ---------------------------------------------------------------
# A. Plotting Helper
# ---------------------------------------------------------------
def plot_filtered_path(states: np.ndarray, all_data: pd.DataFrame, diag: Optional[dict]=None,
                       anchors: Optional[Dict[str, np.ndarray]]=None, title:str="Filtered Trajectory",
                       show: bool=True, save_path: Optional[str]=None):
    """Visualize continuous filtered path, sparse UWB updates, velocity vectors.

    Parameters
    ----------
    states : (N, >=5) filtered state sequence from UKF/EKF (px, py, vx, vy, yaw, ...)
    all_data : merged dataframe (must contain type, Timestamp, PositionX/Y for UWB rows if available)
    diag : diagnostics dict to optionally overlay UWB update points (where innov_norm finite)
    anchors : dict name-> (2,) anchor coordinates
    title : plot title
    show : display figure
    save_path : optional filepath to save figure
    """
    plt.figure(figsize=(8,8))
    # Full continuous trajectory
    plt.plot(states[:,0], states[:,1], color='#4444aa', linewidth=1.0, label='Filtered path')

    # UWB updates (either from diag or directly from df rows)
    if diag is not None and 'innov_norm' in diag:
        mask = np.isfinite(diag['innov_norm'])
        if mask.any():
            # Align states indexing: states was appended each iteration (initial + per row)
            uwb_indices = np.where(mask)[0]
            uwb_pts = states[1:][uwb_indices, :2]  # skip initial
            plt.scatter(uwb_pts[:,0], uwb_pts[:,1], s=14, c='purple', label='UWB update (post)')
    else:
        uwb_rows = all_data[all_data['type']=='uwb'].dropna(subset=['PositionX','PositionY'])
        if len(uwb_rows):
            plt.scatter(uwb_rows['PositionX'], uwb_rows['PositionY'], c='purple', s=14, label='UWB meas')

    # Velocity direction arrows (subsample)
    if states.shape[0] > 10:
        step = max(1, states.shape[0]//40)
        for i in range(0, states.shape[0], step):
            px, py, vx, vy = states[i,0], states[i,1], states[i,2], states[i,3]
            plt.arrow(px, py, 0.4*vx, 0.4*vy, head_width=0.15, head_length=0.25, fc='orange', ec='orange', alpha=0.6)

    # Anchors
    if anchors:
        for name, pos in anchors.items():
            plt.scatter(pos[0], pos[1], marker='^', s=90, c='red')
            plt.text(pos[0]+0.1, pos[1]+0.1, name, color='red', fontsize=9)

    plt.title(title)
    plt.axis('equal')
    plt.legend(loc='best')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()

def validate_position_update(ekf, position, chi2_thresh):
    """Mahalanobis distance gating for UWB updates."""
    y = position.reshape(2,) - ekf.H @ ekf.x
    S = ekf.H @ ekf.P @ ekf.H.T + ekf.R
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return False
    m2 = float(y.T @ Sinv @ y)
    return m2 <= chi2_thresh
