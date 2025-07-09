import numpy as np
from typing import Union, Optional, Tuple, Callable
from tqdm import tqdm


# class KFilter:

#     def __init__(
#             self,
#             state_update_func: Callable[[np.ndarray, Union[np.ndarray, float]],
#                                         np.ndarray], C: np.ndarray,
#             Q: np.ndarray, R: np.ndarray, x0: np.ndarray):
#         """
#         Initializes the Kalman filter with the given system matrices and initial state.

#         Parameters:
#             state_update_func (Callable): Function to update the state based on the current state and input.
#             C (np.ndarray): Observation matrix.
#             Q (np.ndarray): Process noise covariance matrix.
#             R (np.ndarray): Measurement noise covariance matrix.
#             x0 (Optional[np.ndarray]): Initial state estimate. If None, defaults to a zero vector.

#         Attributes:
#             state_update_func (Callable): Function to update the state based on the current state and input.
#             C (np.ndarray): Observation matrix.
#             Q (np.ndarray): Process noise covariance matrix.
#             R (np.ndarray): Measurement noise covariance matrix.
#             x0 (np.ndarray): Initial state estimate.
#             P (np.ndarray): Initial estimate error covariance matrix (initialized as identity).
#         """
#         self.state_update_func = state_update_func
#         self.C = C
#         self.Q = Q
#         self.R = R
#         self.x0 = x0
#         self.P = np.eye(A.shape[0])  # Initial covariance

#     def predict(self, u: Union[np.ndarray,
#                                float]) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Predict the next state and covariance.
#         """
#         u = u.reshape(-1, 1) if np.isscalar(u) else u
#         x_pred = self.A @ self.x0 + self.B @ u
#         P_pred = self.A @ self.P @ self.A.T + self.Q
#         return x_pred, P_pred

#     def update(self, z: Union[np.ndarray,
#                               float]) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Update the state with a new measurement.
#         """
#         z = z.reshape(-1, 1) if np.isscalar(z) else z
#         y_tilde = z - (self.C @ self.x0)
#         S = self.C @ self.P @ self.C.T + self.R
#         K = self.P @ self.C.T @ np.linalg.inv(S)

#         # Update state and covariance
#         x_updated = self.x0 + K @ y_tilde
#         I_KC = np.eye(self.P.shape[0]) - K @ self.C
#         P_updated = I_KC @ self.P

#         return x_updated, P_updated


class KalmanBase:
    """
    Base class for Kalman filters.
    """
    def __init__(
        self,
        x0: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        """
        Initializes the Kalman filter with the given parameters.

        Args:
            x0 (np.ndarray): Initial state estimate vector.
            H (np.ndarray): Observation matrix.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.

        Attributes:
            x0 (np.ndarray): Initial state estimate.
            H (np.ndarray): Observation matrix.
            Q (np.ndarray): Process noise covariance.
            R (np.ndarray): Measurement noise covariance.
            _x_pred (np.ndarray): Predicted state estimate, initialized to x0.
            _P_pred (np.ndarray): Predicted error covariance, initialized to identity matrix.
        """
        self.x0 = x0
        self.P0 = np.eye(x0.shape[0])
        self.H = H
        self.Q = Q
        self.R = R

        self._x_pred = x0
        self._P_pred = self.P0  # Initial covariance

    def state_update(self, x: np.ndarray, u) -> np.ndarray:
        """
        Update the state based on the current state and input.
        This method should be overridden by subclasses to implement 
        specific state update logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def covariance_update(self, P: np.ndarray) -> np.ndarray:
        """
        Update the covariance based on the current covariance and process noise.
        This method should be overridden by subclasses to implement 
        specific covariance update logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def observate(self, x: np.ndarray) -> np.ndarray:
        """
        Get the observation from the current state.
        """
        return self.H @ x

    def predict(self, x: np.ndarray, u,
                P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state based on the current state and input.
        """
        # u = u.reshape(-1, 1) if np.isscalar(u) else u
        # Reshape `u` into a column vector, not necessary here.
        self._x_pred = self.state_update(x, u)
        self._P_pred = self.covariance_update(P)
        return self._x_pred, self._P_pred

    def update(self, y, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Update the state with a new measurement.
        """
        y_tilde = y - self.observate(self._x_pred)
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        # Update state and covariance
        x_updated = self._x_pred + K @ y_tilde
        I_KC = np.eye(self._P_pred.shape[0]) - K @ self.H
        P_updated = I_KC @ self._P_pred

        return x_updated, P_updated

    def apply_filter(self, measure: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Apply the Kalman filter to a measurement with an input.

        Args:
            measure (np.ndarray): Measurement vector.
            u (Union[np.ndarray, float]): Input vector or scalar.

        Returns:
            np.ndarray: Updated state estimate after applying the filter.
        """
        self.predict(self._x_pred, u[0], self._P_pred)
        filtered = []
        with tqdm(total=len(measure), desc='Kalman Filter Progress') as pbar:
            for y, control in zip(measure, u):
                y = np.array([y], dtype=np.float64)
                x_updated, P_updated = self.update(y, self._P_pred)
                filtered.append(self.observate(x_updated).item() if self.observate(x_updated).size == 1 else self.observate(x_updated).flatten())
                self.predict(x_updated, control, P_updated)
                pbar.update(1)
        return np.array(filtered)
