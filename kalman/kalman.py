import numpy as np
from typing import Union, Optional, Tuple


class KFilter:

    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 C: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 x0: Optional[np.ndarray] = None):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x0 = x0 if x0 is not None else np.zeros((A.shape[0], 1))
        self.P = np.eye(A.shape[0])  # Initial covariance

    def predict(self, u: Union[np.ndarray,
                               float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state and covariance.
        """
        u = u.reshape(-1, 1) if np.isscalar(u) else u
        x_pred = self.A @ self.x0 + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q
        return x_pred, P_pred

    def update(self, z: Union[np.ndarray,
                              float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the state with a new measurement.
        """
        z = z.reshape(-1, 1) if np.isscalar(z) else z
        y_tilde = z - (self.C @ self.x0)
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        # Update state and covariance
        x_updated = self.x0 + K @ y_tilde
        I_KC = np.eye(self.P.shape[0]) - K @ self.C
        P_updated = I_KC @ self.P

        return x_updated, P_updated
