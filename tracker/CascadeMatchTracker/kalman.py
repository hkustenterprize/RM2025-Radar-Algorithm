from filterpy.kalman import KalmanFilter
import numpy as np
from typing import List


class KalmanFilter3D:

    def __init__(self, initial_pos: List[float] = None, dt: float = 1.0):

        # (x, y, z, vx, vy, vz)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
        )
        self.reset(initial_pos)

    def predict(self) -> tuple[List[float], List[float]]:
        """
        Predict the next state of the Kalman filter
        Returns:
            A tuple containing the predicted position (x, y, z) and velocity (vx, vy, vz).
        """
        self.kf.predict()
        return self.kf.x[:3].tolist(), self.kf.x[3:].tolist()

    def update(self, pos: List[float]) -> tuple[List[float], List[float]]:
        """
        Update the Kalman filter with a new position measurement.
        Args:
            pos: A list containing the new position measurement [x, y, z].
        """
        self.kf.update(np.array(pos))
        return self.kf.x[:3].tolist(), self.kf.x[3:].tolist()

    def reset(self, initial_pos: List[float] = None) -> None:
        """
        Reset the Kalman filter with a new initial position.
        Args:
            initial_pos: A list containing the new initial position [x, y, z].
        """
        self.kf.x[:3] = np.array(initial_pos or [0.0, 0.0, 0.0]).reshape(3, -1)
        self.kf.x[3:] = np.array([0.0, 0.0, 0.0]).reshape(3, -1)
        self.kf.P *= 1.0
        self.kf.R *= 0.1
        self.kf.Q *= 0.01


class KalmanFilterBox:
    """
    Kalman Filter for bounding box tracking.
    State vector: [center_x, center_y, width, height, vx, vy, vw, vh]
    """

    def __init__(self, initial_bbox: List[float] = None, dt: float = 1.0):
        """
        Initialize Kalman filter for bounding box tracking.
        Args:
            initial_bbox: Initial bounding box [x, y, width, height]
            dt: Time step
        """
        # State: [center_x, center_y, width, height, vx, vy, vw, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],  # center_x
                [0, 1, 0, 0, 0, dt, 0, 0],  # center_y
                [0, 0, 1, 0, 0, 0, dt, 0],  # width
                [0, 0, 0, 1, 0, 0, 0, dt],  # height
                [0, 0, 0, 0, 1, 0, 0, 0],  # vx
                [0, 0, 0, 0, 0, 1, 0, 0],  # vy
                [0, 0, 0, 0, 0, 0, 1, 0],  # vw
                [0, 0, 0, 0, 0, 0, 0, 1],  # vh
            ]
        )

        # Measurement matrix (we observe center_x, center_y, width, height)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        # Process noise covariance
        self.kf.Q = np.eye(8)
        self.kf.Q[4:, 4:]

        # Measurement noise covariance
        self.kf.R = np.eye(4)

        # Error covariance matrix
        self.kf.P = np.eye(8)

        self.reset(initial_bbox)

    def predict(self) -> List[float]:
        """
        Predict the next state of the bounding box.
        Returns:
            Predicted bounding box [center_x, center_y, width, height]
        """
        self.kf.predict()
        return self.kf.x[:4].tolist()

    def update(self, bbox: List[float]) -> List[float]:
        """
        Update the Kalman filter with a new bounding box measurement.
        Args:
            bbox: Measured bounding box [center_x, center_y, width, height]
        Returns:
            Updated bounding box [center_x, center_y, width, height]
        """
        ## Detect jumping
        if abs(self.kf.x[0] - bbox[0]) > 100 or abs(self.kf.x[1] - bbox[1]) > 100:
            self.reset(bbox)
        self.kf.update(np.array(bbox))
        return self.kf.x[:4].tolist()

    def reset(self, initial_bbox: List[float] = None) -> None:
        """
        Reset the Kalman filter with a new initial bounding box.
        Args:
            initial_bbox: Initial bounding box [center_x, center_y, width, height]
        """
        if initial_bbox is None:
            initial_bbox = [0.0, 0.0, 1.0, 1.0]

        # Initialize state vector
        self.kf.x = np.zeros(8)
        self.kf.x[:4] = np.array(initial_bbox)
        self.kf.x[4:] = 0.0  # Initial velocities are zero

        # Reset covariance matrices
        self.kf.P = np.eye(8)

    def get_state(self) -> tuple[List[float], List[float]]:
        """
        Get the current state of the bounding box.
        Returns:
            A tuple containing (bbox, velocities) where:
            - bbox: [center_x, center_y, width, height]
            - velocities: [vx, vy, vw, vh]
        """
        return self.kf.x[:4].tolist(), self.kf.x[4:].tolist()

    def get_bbox_corners(self) -> List[float]:
        """
        Convert center-based representation to corner-based representation.
        Returns:
            Bounding box in format [x1, y1, x2, y2] (top-left, bottom-right)
        """
        center_x, center_y, width, height = self.kf.x[:4]
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        return [x1, y1, x2, y2]

    def set_bbox_from_corners(self, corners: List[float]) -> None:
        """
        Set bounding box from corner-based representation.
        Args:
            corners: Bounding box in format [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = corners
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        self.update([center_x, center_y, width, height])

class KalmanFilter2d:
    """
    2D Kalman Filter for position and velocity tracking.
    State vector: [x, y, vx, vy]
    """

    def __init__(self, initial_pos: List[float] = None, q_std:float = 2.0, r_std: float = 1.0, dt: float = 0.1):
        """
        Initialize 2D Kalman filter.
        Args:
            initial_pos: Initial position [x, y]
            dt: Time step
        """
        # State: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array(
            [
                [1, 0, dt, 0],   # x
                [0, 1, 0, dt],   # y  
                [0, 0, 1, 0],    # vx
                [0, 0, 0, 1],    # vy
            ]
        )
        
        # Measurement matrix (we observe x, y positions)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ]
        )
        
        # Process noise covariance matrix
        # Higher values = more process noise, more responsive to changes
        self.dt = dt
        self.kf.Q = np.array(
            [
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]
        ) * q_std**2
        self.q_std = q_std
        self.r_std = r_std
        
        # Measurement noise covariance matrix
        # Higher values = less trust in measurements
        self.kf.R = np.eye(2) * r_std**2
        
        # Error covariance matrix
        self.kf.P = np.eye(4) * 100.0  # Initial uncertainty
        
        self.reset(initial_pos)

    def predict(self, dt = None) -> tuple[List[float], List[float]]:
        """
        Predict the next state of the Kalman filter.
        Returns:
            A tuple containing the predicted position (x, y) and velocity (vx, vy).
        """
        if dt == None:
            dt = self.dt
        
        self.kf.Q = np.array(
            [
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]
        ) * self.q_std**2
            
        self.kf.predict()
        return self.kf.x[:2].tolist(), self.kf.x[2:].tolist()

    def update(self, pos: List[float]) -> tuple[List[float], List[float]]:
        """
        Update the Kalman filter with a new position measurement.
        Args:
            pos: A list containing the new position measurement [x, y].
        Returns:
            A tuple containing the updated position (x, y) and velocity (vx, vy).
        """        
        
        self.kf.update(np.array(pos))
        return self.kf.x[:2].tolist(), self.kf.x[2:].tolist()

    def reset(self, initial_pos: List[float] = None) -> None:
        """
        Reset the Kalman filter with a new initial position.
        Args:
            initial_pos: A list containing the new initial position [x, y].
        """
        if initial_pos is None:
            initial_pos = [0.0, 0.0]
            
        # Initialize state vector
        self.kf.x = np.zeros(4)
        self.kf.x[:2] = np.array(initial_pos)
        self.kf.x[2:] = 0.0  # Initial velocities are zero
        
        # Reset error covariance matrix
        self.kf.P = np.eye(4) * 100.0

    def get_state(self) -> tuple[List[float], List[float]]:
        """
        Get the current state of the filter.
        Returns:
            A tuple containing (position, velocity) where:
            - position: [x, y]
            - velocity: [vx, vy]
        """
        return self.kf.x[:2].tolist(), self.kf.x[2:].tolist()

    def get_position(self) -> List[float]:
        """
        Get the current position estimate.
        Returns:
            Current position [x, y]
        """
        return self.kf.x[:2].tolist()

    def get_velocity(self) -> List[float]:
        """
        Get the current velocity estimate.
        Returns:
            Current velocity [vx, vy]
        """
        return self.kf.x[2:].tolist()

    def set_process_noise(self, q_std: float) -> None:
        """
        Set the process noise covariance.
        Args:
            q_std: Standard deviation of process noise
        """
        dt = self.dt  # Assuming dt=1.0, could be made configurable
        self.kf.Q = np.array(
            [
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]
        ) * q_std**2

    def set_measurement_noise(self, r_std: float) -> None:
        """
        Set the measurement noise covariance.
        Args:
            r_std: Standard deviation of measurement noise
        """
        self.kf.R = np.eye(2) * r_std**2

    def predict_future(self, steps: int) -> List[List[float]]:
        """
        Predict future positions for multiple time steps ahead.
        Args:
            steps: Number of time steps to predict
        Returns:
            List of predicted positions [[x1, y1], [x2, y2], ...]
        """
        # Save current state
        current_x = self.kf.x.copy()
        current_P = self.kf.P.copy()
        
        predictions = []
        
        # Predict for each step
        for _ in range(steps):
            self.kf.predict()
            predictions.append(self.kf.x[:2].tolist())
        
        # Restore original state
        self.kf.x = current_x
        self.kf.P = current_P
        
        return predictions
