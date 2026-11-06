'''
Five state closed loop model of the MEMS accelerometer.

Everything in this module follows sections 12, 13 and 17 of the development
task sheet:

* the deterministic zero order hold discretisation uses one augmented matrix
  exponential, so ``A_m`` never has to be inverted;
* the mechanical process covariance is obtained with the Van Loan block
  exponential, which turns the continuous white force into a discrete
  covariance without pretending the spectral density is a variance;
* the observability and detectability checks act on the closed loop pair
  ``(F, H)`` with ``F = A + G H``, not on the two mechanical states alone.
'''
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.linalg


class ModelError(ValueError):
    '''
    Raised when the model cannot be built from the given parameters.
    '''


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    '''
    Remove rounding level asymmetry from a covariance matrix.
    '''
    return 0.5 * (matrix + matrix.T)


def check_covariance(matrix: np.ndarray, name: str, tolerance: float = 1e-10) -> np.ndarray:
    '''
    Symmetrise ``matrix`` and reject a significantly negative eigenvalue.

    Args:
        matrix: Candidate covariance matrix.
        name: Name used in error messages.
        tolerance: Relative tolerance below which a negative eigenvalue is
            treated as rounding noise.

    Returns:
        np.ndarray: The symmetrised matrix.

    Raises:
        ModelError: When the matrix is not positive semi definite.
    '''
    symmetric = symmetrize(np.asarray(matrix, dtype=np.float64))
    eigen_min = float(np.min(np.linalg.eigvalsh(symmetric)))
    scale = max(1., float(np.max(np.abs(symmetric))))
    if eigen_min < -tolerance * scale:
        raise ModelError(f'{name} is not positive semi definite '
                         f'(min eigenvalue {eigen_min:.3e}, scale {scale:.3e})')
    return symmetric


def continuous_mechanics(m: float, c: float, k: float, K_u: float) -> tuple:
    '''
    Continuous mechanical model of section 2.1.

    Args:
        m: Equivalent mass in kg.
        c: Damping coefficient in N*s/m.
        k: Spring constant in N/m.
        K_u: PID output to feedback force gain in N per control unit.

    Returns:
        tuple: ``(A_m, B_u, B_a)``, shapes ``(2, 2)``, ``(2, 1)``, ``(2, 1)``.
    '''
    A_m = np.array([[0., 1.], [-k / m, -c / m]], dtype=np.float64)
    B_u = np.array([[0.], [K_u / m]], dtype=np.float64)
    B_a = np.array([[0.], [1.]], dtype=np.float64)
    return A_m, B_u, B_a


def discretize_zero_order_hold(m: float, c: float, k: float, K_u: float, Ts: float) -> tuple:
    '''
    Exact zero order hold discretisation with one augmented matrix exponential.

    Section 12.1::

        M = [[A_m, B_u, B_a], [0, 0, 0], [0, 0, 0]]
        exp(M Ts) = [[Phi, Gamma_u, Gamma_a], [0, 1, 0], [0, 0, 1]]

    Args:
        m, c, k, K_u: Continuous mechanical parameters.
        Ts: Sampling period in seconds.

    Returns:
        tuple: ``(Phi, Gamma_u, Gamma_a)`` with shapes ``(2, 2)``, ``(2, 1)``,
        ``(2, 1)``.
    '''
    A_m, B_u, B_a = continuous_mechanics(m, c, k, K_u)
    augmented = np.zeros((4, 4), dtype=np.float64)
    augmented[:2, :2] = A_m
    augmented[:2, 2:3] = B_u
    augmented[:2, 3:4] = B_a
    exponential = scipy.linalg.expm(augmented * Ts)
    Phi = exponential[:2, :2]
    Gamma_u = exponential[:2, 2:3]
    Gamma_a = exponential[:2, 3:4]
    return Phi, Gamma_u, Gamma_a


def mechanical_process_noise(A_m: np.ndarray, q_b: float, Ts: float) -> tuple:
    '''
    Discrete mechanical process covariance through the Van Loan exponential.

    Section 12.2::

        W_m = B_a q_b B_a^T
        V   = [[A_m, W_m], [0, -A_m^T]]
        exp(V Ts) = [[Phi, Z], [0, Phi^-T]]
        Q_m = Z Phi^T

    Args:
        A_m: Continuous mechanical matrix ``(2, 2)``.
        q_b: Two sided PSD of the equivalent acceleration in (m/s^2)^2/Hz.
        Ts: Sampling period in seconds.

    Returns:
        tuple: ``(Phi, Q_m, W_m)`` with shapes ``(2, 2)``, ``(2, 2)`` and
        ``(2, 2)``.  ``W_m`` is returned for the verification module.

    Raises:
        ModelError: When ``q_b`` is negative.
    '''
    if q_b < 0:
        raise ModelError(f'equivalent acceleration PSD must not be negative, got {q_b}')
    B_a = np.array([[0.], [1.]], dtype=np.float64)
    W_m = B_a @ B_a.T * q_b

    van_loan = np.zeros((4, 4), dtype=np.float64)
    van_loan[:2, :2] = A_m
    van_loan[:2, 2:] = W_m
    van_loan[2:, 2:] = -A_m.T
    exponential = scipy.linalg.expm(van_loan * Ts)

    Phi = exponential[:2, :2]
    Z = exponential[:2, 2:]
    Q_m = Z @ Phi.T
    return Phi, Q_m, W_m


def continuous_acceleration_noise(m: float, q_F: float) -> float:
    '''
    Convert a two sided force PSD into the equivalent accelerance PSD.

    Section 12.2 states that ``q_b = q_F / m^2``; both are one and the same
    noise, expressed in two different units.
    '''
    return q_F / (m * m)


@dataclass
class ObservabilityReport:
    '''
    Numerical observability / detectability evidence for the closed loop pair.
    '''

    rank: int
    singular_values: np.ndarray
    condition: float
    scales: np.ndarray
    detectable: bool
    undetectable_modes: List[complex]
    accel_observable: bool
    accel_null_component: float
    notes: List[str]

    def as_dict(self) -> Dict[str, object]:
        '''
        JSON friendly view of the report.
        '''
        return {
            'rank': self.rank,
            'singular_values': [float(value) for value in self.singular_values],
            'condition': float(self.condition),
            'scales': [float(value) for value in self.scales],
            'detectable': bool(self.detectable),
            'undetectable_modes': [[float(mode.real), float(mode.imag)]
                                   for mode in self.undetectable_modes],
            'accel_observable': bool(self.accel_observable),
            'accel_null_component': float(self.accel_null_component),
            'notes': list(self.notes),
        }


class ClosedLoopModel:
    '''
    The five state model of section 13: ``X[k+1] = A X[k] + G u[k] + L n[k] + xi[k]``.

    Attributes:
        Phi, Gamma_u, Gamma_a: Mechanical discretisation, ``(2, 2)``, ``(2, 1)``,
            ``(2, 1)``.
        Q_m: Discrete mechanical process covariance, ``(2, 2)``.
        A, Q0, F: ``(5, 5)`` state, process covariance and closed loop matrix.
        G, L: ``(5, 1)`` control and readout noise input matrices.
        H: ``(1, 5)`` observation row.
        R: Scalar measurement noise variance of the observed PID output.
    '''

    def __init__(self, config):
        '''
        Build every constant matrix from a validated configuration.

        Args:
            config: A :class:`config.Config` instance.

        Raises:
            ModelError: When the parameters produce an unusable covariance.
        '''
        self.config = config
        self.m = config.m
        self.c = config.c
        self.k = config.k
        self.K_u = config.K_u
        self.Ts = config.Ts
        self.k_s = config.k_s
        self.K_i = config.K_i
        self.alpha = config.alpha
        self.beta = config.beta
        self.q_b = config.q_b
        self.q_a = config.q_a
        self.r = config.r
        self.R = config.R

        self.Phi, self.Gamma_u, self.Gamma_a = discretize_zero_order_hold(
            self.m, self.c, self.k, self.K_u, self.Ts)

        _, self.Q_m, self.W_m = mechanical_process_noise(
            continuous_mechanics(self.m, self.c, self.k, self.K_u)[0],
            self.q_b, self.Ts)

        self._check_gamma_relation()
        self._build_state_matrices()

    def _check_gamma_relation(self):
        '''
        Verify ``Gamma_u = (K_u / m) Gamma_a`` as required by section 12.1.
        '''
        scale = max(1e-30, float(np.max(np.abs(self.Gamma_a))))
        error = float(np.max(np.abs(self.Gamma_u - (self.K_u / self.m) * self.Gamma_a)))
        if error > 1e-10 * scale:
            raise ModelError('Gamma_u and (K_u/m) Gamma_a disagree by '
                             f'{error:.3e}, the discretisation is inconsistent')

    def _build_state_matrices(self):
        '''
        Assemble ``A``, ``G``, ``L``, ``H``, ``Q0`` and the closed loop pair.
        '''
        phi = self.Phi
        ga = self.Gamma_a
        gu = self.Gamma_u

        A = np.zeros((5, 5), dtype=np.float64)
        A[0:2, 0:2] = phi
        A[0:2, 2] = ga.ravel()
        A[2, 2] = 1.
        A[3, 0] = self.K_i * self.k_s
        A[3, 3] = 1.
        A[4, 0] = self.k_s
        self.A = A

        G = np.zeros((5, 1), dtype=np.float64)
        G[0:2, 0] = gu.ravel()
        self.G = G

        L = np.zeros((5, 1), dtype=np.float64)
        L[3, 0] = self.K_i
        L[4, 0] = 1.
        self.L = L

        H = np.zeros((1, 5), dtype=np.float64)
        H[0, 0] = self.alpha * self.k_s
        H[0, 3] = 1.
        H[0, 4] = self.beta
        self.H = H

        Q0 = np.zeros((5, 5), dtype=np.float64)
        Q0[0:2, 0:2] = self.Q_m
        Q0[2, 2] = self.q_a
        self.Q0 = check_covariance(Q0, 'Q0', self.config.tolerances['psd_eigen'])

        self.F = self.A + self.G @ self.H
        self.D = self.L + self.alpha * self.G
        self.Q_cl = symmetrize(self.Q0 + self.r * (self.D @ self.D.T))
        self.N_cl = self.alpha * self.r * self.D
        self.Q_in = symmetrize(self.Q0 + self.r * (self.L @ self.L.T))
        self.N_in = self.alpha * self.r * self.L

    def with_q_a(self, q_a: float) -> 'ClosedLoopModel':
        '''
        Return a copy of this model with another ``q_a``.

        Section 20: a ``q_a`` sweep only changes the third diagonal entry of
        ``Q0``; every other matrix stays the same.
        '''
        import copy

        clone = copy.deepcopy(self)
        clone.q_a = float(q_a)
        Q0 = clone.Q0.copy()
        Q0[2, 2] = clone.q_a
        clone.Q0 = Q0
        clone.Q_cl = symmetrize(clone.Q0 + clone.r * (clone.D @ clone.D.T))
        return clone

    # ------------------------------------------------------------------
    # observability / detectability, section 17
    # ------------------------------------------------------------------
    def observability_matrix(self, F: Optional[np.ndarray] = None,
                             H: Optional[np.ndarray] = None) -> np.ndarray:
        '''
        Stacked observability matrix ``[H; H F; H F^2; H F^3; H F^4]``.

        Args:
            F: Dynamics matrix, defaults to ``self.F``.
            H: Observation row, defaults to ``self.H``.  Both have to come from
                the same coordinate system: when ``F`` is the scaled
                ``F_z = T^-1 F T`` the observation row must be ``H_z = H T``.
        '''
        matrix = self.F if F is None else F
        observation = self.H if H is None else H
        rows = []
        row = observation.copy()
        for _ in range(5):
            rows.append(row)
            row = row @ matrix
        return np.vstack(rows)

    def _default_scales(self) -> np.ndarray:
        '''
        Physical state scales used for the rank decision.

        The prior standard deviations are the natural scales of the state; a
        zero variance is replaced by one so that the scaling stays invertible.
        '''
        if self.config.state_scales is not None:
            return np.asarray(self.config.state_scales, dtype=np.float64)
        variances = np.diag(self.config.P0).astype(np.float64)
        scales = np.sqrt(np.abs(variances))
        scales[scales <= 0] = 1.
        return scales

    def observability_report(self) -> ObservabilityReport:
        '''
        Compute the scaled rank, the PBH detectability and the acceleration
        estimability of the closed loop pair ``(F, H)``.

        Returns:
            ObservabilityReport: Rank, singular values, detectability and the
            component of the acceleration direction inside ``ker(O)``.
        '''
        scales = self._default_scales()
        inverse = 1. / scales
        T = np.diag(scales)
        Tinv = np.diag(inverse)

        F_z = Tinv @ self.F @ T
        H_z = self.H @ T
        O = self.observability_matrix(F_z, H_z)

        singular_values = np.linalg.svd(O, compute_uv=False)
        rtol = self.config.tolerances['rank_relative']
        rank = int(np.sum(singular_values > rtol * singular_values[0])) if singular_values[0] > 0 else 0
        condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float('inf')

        notes: List[str] = []
        if self.k_s == 0:
            notes.append('k_s = 0: the displacement does not reach the readout, '
                         'the external acceleration cannot be reconstructed.')
        if self.K_i == 0:
            notes.append('K_i = 0: the direction v_dc = [1, 0, k/m, -K_p K_s, K_s] '
                         'is a static unobservable mode (section 17.2).')
        if self.config.K_d == 0:
            notes.append('K_d = 0: h does not influence any other state, the fifth '
                         'state should be removed to obtain a four state model.')

        undetectable: List[complex] = []
        eigenvalues = np.linalg.eigvals(F_z)
        for eigenvalue in eigenvalues:
            if abs(eigenvalue) < 1. - 1e-9:
                continue
            pbh = np.vstack([eigenvalue * np.eye(5) - F_z, H_z])
            pbh_singular = np.linalg.svd(pbh, compute_uv=False)
            pbh_rank = int(np.sum(pbh_singular > rtol * pbh_singular[0]))
            if pbh_rank < 5:
                undetectable.append(complex(eigenvalue))

        # Kernel of O: does the unobservable subspace touch the acceleration?
        _, _, vh = np.linalg.svd(O)
        null_basis = vh[rank:].T if rank < 5 else np.zeros((5, 0))
        c_a = np.array([[0., 0., 1., 0., 0.]])
        if null_basis.size:
            projection = c_a @ null_basis
            accel_null = float(np.linalg.norm(projection))
        else:
            accel_null = 0.

        return ObservabilityReport(
            rank=rank,
            singular_values=singular_values,
            condition=condition,
            scales=scales,
            detectable=not undetectable,
            undetectable_modes=undetectable,
            accel_observable=accel_null < 1e-9,
            accel_null_component=accel_null,
            notes=notes,
        )

    def as_dict(self) -> Dict[str, object]:
        '''
        JSON friendly view of the constant matrices.
        '''
        return {
            'Phi': self.Phi.tolist(),
            'Gamma_u': self.Gamma_u.ravel().tolist(),
            'Gamma_a': self.Gamma_a.ravel().tolist(),
            'Q_m': self.Q_m.tolist(),
            'A': self.A.tolist(),
            'G': self.G.ravel().tolist(),
            'L': self.L.ravel().tolist(),
            'H': self.H.ravel().tolist(),
            'Q0': self.Q0.tolist(),
            'R': float(self.R),
            'F': self.F.tolist(),
        }

    def __repr__(self) -> str:
        return (f'ClosedLoopModel(m={self.m!r}, c={self.c!r}, k={self.k!r}, '
                f'K_u={self.K_u!r}, Ts={self.Ts!r})')
