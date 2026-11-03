import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pandas as pd

class EKF_Fusion:
    def __init__(self, initial_x, initial_P, q_pos, q_vel, q_orient, r_uwb):
        """
        Initializes the Extended Kalman Filter for IMU-UWB fusion.
        
        Args:
            initial_x (np.array): Initial state vector [px,py,pz, vx,vy,vz, qw,qx,qy,qz].
            initial_P (np.array): Initial covariance matrix (10x10).
            q_pos, q_vel, q_orient (float): Process noise parameters for position, velocity, and orientation.
            r_uwb (float): Measurement noise for the UWB sensor.
        """
        self.x = initial_x  # State vector
        self.P = initial_P  # State covariance
        
        # Process Noise Covariance (Q)
        # Represents uncertainty in the motion model
        self.Q = np.diag([
            q_pos, q_pos, q_pos,       # Position noise
            q_vel, q_vel, q_vel,       # Velocity noise
            q_orient, q_orient, q_orient, q_orient # Orientation noise
        ])
        
        # Measurement Noise Covariance (R)
        # Represents uncertainty in the UWB measurement
        self.R = np.diag([r_uwb, r_uwb, r_uwb])
        
        # Measurement Jacobian (H) is constant for UWB
        self.H = np.zeros((3, 10))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.q_vel = q_vel

    def predict(self, accel, gyro, dt):
        """
        Performs the prediction step using IMU data.
        
        Args:
            accel (np.array): 3-axis linear acceleration (m/s^2).
            gyro (np.array): 3-axis angular velocity (rad/s).
            dt (float): Time step.
        """
        # --- State Prediction ---
        pos = self.x[0:3]
        vel = self.x[3:6]
        q = self.x[6:10]
        
        # Get rotation matrix from quaternion
        rot_matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        
        # Rotate acceleration from body to world frame and remove gravity
        accel_world = rot_matrix @ accel
        # Note: Assuming accel is linear acceleration (gravity already removed)
        
        # Predict next state
        pos_new = pos + vel * dt + 0.5 * accel_world * dt**2
        vel_new = vel + accel_world * dt
        
        # Predict next orientation
        gyro_quat = np.array([0, gyro[0], gyro[1], gyro[2]]) * 0.5 * dt
        q_new = self.quaternion_multiply(q, self.quaternion_exp(gyro_quat))
        
        self.x = np.hstack([pos_new, vel_new, q_new])
        
        # --- Covariance Prediction ---
        F = np.eye(10)
        F[0:3, 3:6] = np.eye(3) * dt
        # More complex Jacobians for orientation and acceleration can be added here
        # but this is a common and effective simplification.
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_uwb):
        """
        Performs the update/correction step using UWB data.
        
        Args:
            z_uwb (np.array): UWB position measurement [x, y, z].
        """
        # Innovation (measurement residual)
        y = z_uwb - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(10) - K @ self.H) @ self.P
        
        # Normalize quaternion to prevent numerical drift
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

    # Helper functions for quaternion math
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quaternion_exp(self, v):
        norm = np.linalg.norm(v[1:])
        if norm > 0:
            w = np.cos(norm)
            vec = (v[1:] / norm) * np.sin(norm)
            return np.hstack([w, vec])
        return np.array([1.0, 0.0, 0.0, 0.0])

class EKF_Fusion_2D:
    def __init__(self, initial_x, initial_P, q_vel, q_yaw, r_uwb, q_bg=1e-3):
        """
        Robust 2D EKF with gyro bias state.

        State: [px, py, vx, vy, yaw, bg]
        Args:
            initial_x (np.array): Initial state vector [px, py, vx, vy, yaw] (or 6D with bg).
            initial_P (np.array): Initial covariance matrix (5x5 or 6x6).
            q_vel (float): Process noise on velocity (per second).
            q_yaw (float): Process noise on yaw (per second).
            r_uwb (float): UWB position noise std (meters) used on both axes.
            q_bg (float): Gyro bias random-walk noise (per second).
        """
        # Promote state to include gyro bias if needed
        if initial_x.shape[0] == 5:
            initial_x = np.hstack([initial_x, 0.0])
            P6 = np.zeros((6, 6))
            P6[:5, :5] = initial_P
            P6[5, 5] = 1.0
            initial_P = P6

        self.x = initial_x.astype(float)  # [px, py, vx, vy, yaw, bg]
        self.P = initial_P.astype(float)

        # Process noise parameters (per second). Will be scaled by dt in predict.
        self.q_vel = float(q_vel)
        self.q_yaw = float(q_yaw)
        self.q_bg = float(q_bg)

        # Measurement Noise Covariance (R) for 2D UWB
        self.R = np.diag([r_uwb, r_uwb]).astype(float)

        # Measurement Jacobian (H) for 2D UWB: position only
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _process_Q(self, dt):
        # White noise on velocity, yaw random walk, and gyro bias random walk
        Q = np.zeros((6, 6))
        Q[2, 2] = self.q_vel * dt
        Q[3, 3] = self.q_vel * dt
        Q[4, 4] = self.q_yaw * dt
        Q[5, 5] = self.q_bg * dt
        return Q

    def predict(self, accel_xy, gyro_z, dt):
        """
        Predict step using body-frame accel (x,y) and yaw-rate (z).
        accel_xy: shape (2,), gyro_z: float, dt: float
        Returns (F, Q) for optional smoothing.
        """
        if dt <= 0:
            return getattr(self, "_last_F", np.eye(6)), self._process_Q(0.0)

        px, py, vx, vy, yaw, bg = self.x

        # Rotate body accel to world frame
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        axw, ayw = (rot @ accel_xy.reshape(2, 1)).ravel()

        # Integrate
        px += vx * dt + 0.5 * axw * dt * dt
        py += vy * dt + 0.5 * ayw * dt * dt
        vx += axw * dt
        vy += ayw * dt
        yaw += (gyro_z - bg) * dt
        yaw = self._wrap_angle(yaw)

        self.x[:5] = [px, py, vx, vy, yaw]
        self.x[5] = bg

        # Jacobian F
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt

        # Sensitivity of accel world to yaw
        ax, ay = accel_xy
        daxw_dyaw = -ax * s - ay * c
        dayw_dyaw =  ax * c - ay * s
        F[2, 4] = daxw_dyaw * dt
        F[3, 4] = dayw_dyaw * dt
        # yaw depends on bias
        F[4, 5] = -dt

        Q = self._process_Q(dt)
        self.P = F @ self.P @ F.T + Q
        # Numerical stabilization
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-12 * np.eye(6)

        # expose last transition for smoothers
        self._last_F = F
        self._last_Q = Q
        return F, Q

    def update(self, z_uwb_xy):
        """UWB position update with Joseph-stabilized covariance update."""
        z = z_uwb_xy.reshape(2,)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        I = np.eye(6)
        KH = K @ self.H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-12 * np.eye(6)

        # wrap yaw
        self.x[4] = self._wrap_angle(self.x[4])

    def update_heading(self, z_yaw_rad: float, r_heading_std: float):
        """Heading (yaw) pseudo-measurement update: h(x)=yaw.

        Args:
            z_yaw_rad: measured absolute yaw in radians.
            r_heading_std: measurement noise std (radians).
        """
        # Residual with proper angle wrapping: z - yaw
        y = self._wrap_angle(float(z_yaw_rad) - float(self.x[4]))

        # Jacobian for yaw: d(yaw)/dx = [0 0 0 0 1 0]
        Hh = np.zeros((1, 6))
        Hh[0, 4] = 1.0

        Rh = np.array([[float(r_heading_std) ** 2]], dtype=float)
        S = Hh @ self.P @ Hh.T + Rh
        K = self.P @ Hh.T @ np.linalg.inv(S)

        self.x = self.x + (K @ np.array([y])).reshape(-1)

        # Joseph form for numerical stability
        I = np.eye(6)
        KH = K @ Hh
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ Rh @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-12 * np.eye(6)

        # wrap yaw back to [-pi, pi)
        self.x[4] = self._wrap_angle(self.x[4])

    def zero_velocity_update(self, sigma_v=0.05):
        """Optional pseudo-measurement when stationary: vx=vy=0."""
        H = np.zeros((2, 6))
        H[0, 2] = 1.0
        H[1, 3] = 1.0
        Rz = np.diag([sigma_v**2, sigma_v**2])
        y = -H @ self.x
        S = H @ self.P @ H.T + Rz
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ Rz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-12 * np.eye(6)

