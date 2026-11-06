'''
Discrete time varying Kalman estimator for the closed loop MEMS accelerometer.

The single recommended recursion path is (14-1), (14-2), (14-3), (14-6),
(14-7) and (14-9) of the task sheet.  The joint form (14-4)-(14-5) and the
posterior noise form (15-1)-(15-3) are kept as independent review formulas:
they are used by the test suite to prove that the implemented recursion really
solves the conditional prediction problem, they are not used by ``step``.

Only the current PID output ``u[k]`` enters :meth:`ClosedLoopKalman.step`.
History is carried by the internal estimate, the covariance and the two PID
memory states ``I`` and ``h = e[k-1]``.
'''
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .model import ClosedLoopModel, ModelError, symmetrize


class FilterError(RuntimeError):
    '''
    Raised when the recursion detects a numerical or model inconsistency.
    '''


@dataclass
class KalmanStep:
    '''
    Output of one :meth:`ClosedLoopKalman.step` call.

    Attributes:
        index: Sample index ``k`` this output belongs to.
        state: Posterior mean ``mu_k^+`` with ``[x, v, a, I, h]``.
        covariance: Posterior covariance ``P_k^+``, ``(5, 5)``.
        acceleration: Estimated external acceleration ``mu_a^+``.
        displacement: Estimated displacement ``mu_x^+``.
        velocity: Estimated velocity ``mu_v^+``.
        innovation: Innovation ``nu_k``.
        innovation_cov: Innovation variance ``S_k``.
        gain: Current gain ``K_k``.
        next_gain: Gain ``J_k`` used for the conditional prediction.
        posterior_variance: Diagonal of ``P_k^+``.
        prior_variance: Diagonal of the stored next prior ``P_{k+1}^-``.
        singular: True when the zero innovation variance path (18-1) was taken.
    '''

    index: int
    state: np.ndarray
    covariance: np.ndarray
    acceleration: float
    displacement: float
    velocity: float
    innovation: float
    innovation_cov: float
    gain: np.ndarray
    next_gain: np.ndarray
    posterior_variance: np.ndarray
    prior_variance: np.ndarray
    singular: bool = False

    def as_dict(self) -> Dict[str, object]:
        '''
        JSON friendly view of one step.
        '''
        return {
            'index': int(self.index),
            'state': [float(value) for value in self.state],
            'acceleration': float(self.acceleration),
            'displacement': float(self.displacement),
            'velocity': float(self.velocity),
            'innovation': float(self.innovation),
            'innovation_cov': float(self.innovation_cov),
            'gain': [float(value) for value in self.gain],
            'next_gain': [float(value) for value in self.next_gain],
            'posterior_variance': [float(value) for value in self.posterior_variance],
            'prior_variance': [float(value) for value in self.prior_variance],
            'singular': bool(self.singular),
        }


def innovation(model: ClosedLoopModel, mu: np.ndarray, P: np.ndarray, u_k: float) -> tuple:
    '''
    Innovation and its variance, equation (14-1).

    Args:
        model: Constant matrices of the closed loop model.
        mu: Prior mean ``mu_k``, shape ``(5,)``.
        P: Prior covariance ``P_k``, shape ``(5, 5)``.
        u_k: Current PID output.

    Returns:
        tuple: ``(nu, S, HPHt)``.
    '''
    h = model.H.ravel()
    prediction = float(h @ mu)
    nu = float(u_k) - prediction
    hpht = float(h @ P @ h)
    S = hpht + float(model.R)
    return nu, S, hpht


def reference_predict_joint(model: ClosedLoopModel, mu: np.ndarray, P: np.ndarray,
                            u_k: float) -> tuple:
    '''
    Review formula (14-4)-(14-5): joint Gaussian conditioning of ``X[k+1]`` on ``u[k]``.

    This form subtracts two comparably sized matrices and is therefore not used
    by the estimator itself, only to verify the implemented recursion.
    '''
    nu, S, _ = innovation(model, mu, P, u_k)
    if S == 0:
        raise FilterError('S = 0, the joint form is undefined')

    V = model.F @ P @ model.F.T + model.Q_cl
    C_cl = model.F @ P @ model.H.T + model.N_cl
    mu_next = model.F @ mu + (C_cl.ravel() * nu / S)
    P_next = V - (C_cl @ C_cl.T) / S
    return mu_next, symmetrize(P_next)


def reference_predict_posterior(model: ClosedLoopModel, mu: np.ndarray, P: np.ndarray,
                                u_k: float) -> tuple:
    '''
    Review formula (15-1)-(15-3): predict from the posterior plus the posterior
    statistics of ``n[k]``.

    Kept as an independent route to the same next prior, including the
    conditional noise mean ``eta`` and the cross covariance ``T``.
    '''
    nu, S, _ = innovation(model, mu, P, u_k)
    if S == 0:
        raise FilterError('S = 0, the posterior form is undefined')

    h = model.H.ravel()
    alpha = model.alpha
    r = float(model.r)

    K = (P @ h) / S
    mu_post = mu + K * nu
    IKH = np.eye(5) - np.outer(K, h)
    P_post = symmetrize(IKH @ P @ IKH.T + model.R * np.outer(K, K))

    eta = alpha * r * nu / S
    rho = r - (alpha * r) ** 2 / S
    T = -K * (alpha * r)

    l = model.L.ravel()
    A = model.A
    AT = A @ T
    mu_next = A @ mu_post + model.G.ravel() * u_k + l * eta
    P_next = (A @ P_post @ A.T + model.Q0 + rho * np.outer(l, l)
              + np.outer(AT, l) + np.outer(l, AT))
    return mu_next, symmetrize(P_next)


class ClosedLoopKalman:
    '''
    Five state discrete time varying Kalman estimator of section 16.

    The estimator stores the prior ``(mu_k, P_k)`` of the *next* sample.  A
    call to :meth:`step` consumes ``u[k]``, produces the posterior of sample
    ``k`` and leaves the prior of ``k+1`` in place.  The recursion therefore
    never needs the previous PID outputs.
    '''

    def __init__(self, model: ClosedLoopModel, x0_hat: np.ndarray, P0: np.ndarray,
                 tolerances: Optional[Dict[str, float]] = None):
        '''
        Create the estimator from constants and a prior.

        Args:
            model: Validated closed loop model.
            x0_hat: Prior mean of ``[x[0], v[0], a[0], I[0], e[-1]]``.
            P0: Prior covariance of the same state, shape ``(5, 5)``.
            tolerances: Numerical tolerances, defaults to the model ones.

        Raises:
            FilterError: When the prior is not a usable covariance.
        '''
        self.model = model
        self.tolerances = dict(tolerances or model.config.tolerances)

        mu = np.asarray(x0_hat, dtype=np.float64).ravel()
        P = np.asarray(P0, dtype=np.float64)
        if mu.shape != (5,):
            raise FilterError(f'x0_hat must have five entries, got {mu.shape}')
        if P.shape != (5, 5):
            raise FilterError(f'P0 must be 5x5, got {P.shape}')
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(P)):
            raise FilterError('the prior contains non finite entries')

        self.mu = mu.copy()
        self.P = self._validated_covariance(P, 'P0')
        self.index = 0

        self.mu_post = self.mu.copy()
        self.P_post = self.P.copy()

        self.history: Dict[str, List] = {
            'index': [],
            'state': [],
            'acceleration': [],
            'displacement': [],
            'velocity': [],
            'innovation': [],
            'innovation_cov': [],
            'gain': [],
            'next_gain': [],
            'posterior_variance': [],
            'prior_variance': [],
        }

    @classmethod
    def initialize(cls, config) -> 'ClosedLoopKalman':
        '''
        Build the estimator from a validated configuration, section 5.

        Args:
            config: A :class:`config.Config` instance.

        Returns:
            ClosedLoopKalman: Estimator positioned before the first update.
        '''
        model = ClosedLoopModel(config)
        return cls(model, config.X0_hat, config.P0, config.tolerances)

    def _validated_covariance(self, matrix: np.ndarray, name: str) -> np.ndarray:
        '''
        Symmetrise a covariance and reject significantly negative eigenvalues.
        '''
        symmetric = symmetrize(matrix)
        try:
            return _psd_checked(symmetric, name, self.tolerances['psd_eigen'])
        except ModelError as error:
            raise FilterError(str(error)) from error

    def step(self, u_k: float) -> KalmanStep:
        '''
        Consume one PID output and advance the estimator.

        The execution order follows section 16: read the stored prior, build
        the innovation, form both gains from the same prior, publish the
        posterior of ``k``, then store the conditional prior of ``k + 1``.

        Args:
            u_k: Current PID output ``u[k]``, the only online measurement.

        Returns:
            KalmanStep: Posterior estimate of sample ``k`` plus diagnostics.

        Raises:
            FilterError: On a non finite input, a broken covariance, or an
                observation that contradicts a degenerate innovation variance.
        '''
        if not np.isfinite(u_k):
            raise FilterError(f'u[{self.index}] is not finite: {u_k!r}')

        mu = self.mu
        P = self.P
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(P)):
            raise FilterError(f'non finite prior at index {self.index}')

        model = self.model
        h = model.H.ravel()
        A = model.A
        l = model.L.ravel()
        g = model.G.ravel()
        alpha = model.alpha
        r = float(model.r)

        nu, S, hpht = innovation(model, mu, P, u_k)

        scale = abs(hpht) + abs(float(model.R))
        singular = S <= self.tolerances['singular_relative'] * scale

        if singular:
            if scale == 0.:
                # Both HPHt and R vanish, so S carries no scale of its own.
                # Only a rounding level innovation is consistent with a
                # deterministic observation.
                rounding = self.tolerances['innovation_relative'] * max(1., abs(u_k))
                consistent = abs(nu) <= rounding
            else:
                consistent = nu * nu <= self.tolerances['innovation_relative'] * scale
            if not consistent:
                raise FilterError(
                    f'innovation variance is zero at index {self.index} but the '
                    f'innovation is {nu:.6e}; the observation contradicts the '
                    'deterministic model or the initialisation is misaligned')

            mu_post = mu.copy()
            P_post = P.copy()
            K = np.zeros(5)
            J = np.zeros(5)
            mu_next = A @ mu + g * u_k
            P_next = A @ P @ A.T + model.Q0 + r * np.outer(l, l)
        else:
            K = (P @ h) / S
            mu_post = mu + K * nu
            IKH = np.eye(5) - np.outer(K, h)
            P_post = IKH @ P @ IKH.T + model.R * np.outer(K, K)

            C = A @ (P @ h) + alpha * r * l
            J = C / S
            mu_next = A @ mu + g * u_k + J * nu

            B = A - np.outer(J, h)
            ell = l - alpha * J
            P_next = B @ P @ B.T + r * np.outer(ell, ell) + model.Q0

        P_post = self._validated_covariance(P_post, f'P+[{self.index}]')
        P_next = self._validated_covariance(P_next, f'P-[{self.index + 1}]')

        self.mu_post = mu_post
        self.P_post = P_post
        self.mu = mu_next
        self.P = P_next

        output = KalmanStep(
            index=self.index,
            state=mu_post,
            covariance=P_post,
            acceleration=float(mu_post[2]),
            displacement=float(mu_post[0]),
            velocity=float(mu_post[1]),
            innovation=float(nu),
            innovation_cov=float(S),
            gain=K,
            next_gain=J,
            posterior_variance=np.diag(P_post).copy(),
            prior_variance=np.diag(P_next).copy(),
            singular=bool(singular),
        )

        self._record(output)
        self.index += 1
        return output

    def _record(self, step_output: KalmanStep):
        '''
        Append one step to the diagnostic history.
        '''
        self.history['index'].append(step_output.index)
        self.history['state'].append(step_output.state.copy())
        self.history['acceleration'].append(step_output.acceleration)
        self.history['displacement'].append(step_output.displacement)
        self.history['velocity'].append(step_output.velocity)
        self.history['innovation'].append(step_output.innovation)
        self.history['innovation_cov'].append(step_output.innovation_cov)
        self.history['gain'].append(step_output.gain.copy())
        self.history['next_gain'].append(step_output.next_gain.copy())
        self.history['posterior_variance'].append(step_output.posterior_variance.copy())
        self.history['prior_variance'].append(step_output.prior_variance.copy())

    def run(self, u_sequence: Sequence[float]) -> Dict[str, np.ndarray]:
        '''
        Filter a whole PID output sequence.

        Args:
            u_sequence: PID outputs ``u[0 ... N-1]``.

        Returns:
            Dict[str, np.ndarray]: Stacked diagnostic history including
            ``acceleration``, ``state`` and the covariance diagonals.
        '''
        for value in u_sequence:
            self.step(float(value))

        result = {}
        for key, values in self.history.items():
            result[key] = np.asarray(values)
        return result


def _psd_checked(matrix: np.ndarray, name: str, tolerance: float) -> np.ndarray:
    '''
    Symmetrise and check positive semi definiteness without importing errors.
    '''
    symmetric = symmetrize(np.asarray(matrix, dtype=np.float64))
    eigen_min = float(np.min(np.linalg.eigvalsh(symmetric)))
    scale = max(1., float(np.max(np.abs(symmetric))))
    if eigen_min < -tolerance * scale:
        raise ModelError(f'{name} is not positive semi definite '
                         f'(min eigenvalue {eigen_min:.3e}, scale {scale:.3e})')
    return symmetric
