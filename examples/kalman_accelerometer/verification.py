'''
Algebraic verification of the implementation, section 19.1 of the V1.1 task
sheet plus section 19.1 of the V2.1 task sheet for the FOH path.

Every check in this module compares two independent routes to the same
quantity.  Nothing here is a performance claim: the routines only prove that
the discretisation, the covariances and the recursive formulas agree with
their definitions.
'''
import copy
from typing import Dict, Optional

import numpy as np
import scipy.integrate
import scipy.linalg

from . import kalman as kalman_module
from .config import with_prior_model
from .model import (ClosedLoopModel, continuous_mechanics, discretize_first_order_hold,
                    discretize_zero_order_hold, foh_matrices, mechanical_process_noise,
                    symmetrize)


def _integrate_matrix(function, Ts: float, shape: tuple) -> np.ndarray:
    '''
    Numerically integrate a matrix valued function on ``[0, Ts]``.
    '''
    def wrapper(tau):
        return np.asarray(function(tau), dtype=np.float64).ravel()

    values, _ = scipy.integrate.quad_vec(wrapper, 0., Ts, epsabs=1e-14, epsrel=1e-12)
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _state_scales(model: ClosedLoopModel) -> np.ndarray:
    '''
    Physical scales used to draw random legal priors for the algebraic checks.

    The values are test magnitudes only, they are not device numbers.  The
    Jerk scale is chosen independently of the acceleration scale, as section
    17.4 requires.
    '''
    if model.has_jerk:
        return np.array([1e-18, 1e-12, 1e-4, 1e2, 1e-4, 1e-6])
    return np.array([1e-18, 1e-12, 1e-4, 1e-4, 1e-6])


def check_discretization(m: float, c: float, k: float, K_u: float, Ts: float) -> Dict[str, object]:
    '''
    Compare the augmented exponential with an independent numerical integral.

    Section 19.1 of the V1.1 task sheet (ZOH path).

    Returns:
        Dict[str, object]: Maximum absolute deviations for ``Phi``, ``Gamma_u``,
        ``Gamma_a`` and the ``Gamma_u = (K_u/m) Gamma_a`` identity.
    '''
    A_m, B_u, B_a = continuous_mechanics(m, c, k, K_u)
    Phi, Gamma_u, Gamma_a = discretize_zero_order_hold(m, c, k, K_u, Ts)

    Phi_ref = scipy.linalg.expm(A_m * Ts)
    Gamma_u_ref = _integrate_matrix(lambda tau: scipy.linalg.expm(A_m * tau) @ B_u, Ts, (2, 1))
    Gamma_a_ref = _integrate_matrix(lambda tau: scipy.linalg.expm(A_m * tau) @ B_a, Ts, (2, 1))

    errors = {
        'Phi': float(np.max(np.abs(Phi - Phi_ref))),
        'Gamma_u': float(np.max(np.abs(Gamma_u - Gamma_u_ref))),
        'Gamma_a': float(np.max(np.abs(Gamma_a - Gamma_a_ref))),
    }
    scale = max(1e-30, float(np.max(np.abs(Gamma_a_ref))))
    errors['Gamma_relation'] = float(
        np.max(np.abs(Gamma_u - (K_u / m) * Gamma_a)) / scale)
    errors['Phi_relative'] = errors['Phi'] / max(1., float(np.max(np.abs(Phi_ref))))
    errors['Gamma_u_relative'] = errors['Gamma_u'] / max(1e-30, float(np.max(np.abs(Gamma_u_ref))))
    errors['Gamma_a_relative'] = errors['Gamma_a'] / scale
    errors['passed'] = (errors['Phi_relative'] < 1e-9
                        and errors['Gamma_u_relative'] < 1e-9
                        and errors['Gamma_a_relative'] < 1e-9)
    return errors


def check_foh_discretization(m: float, c: float, k: float, K_u: float,
                             Ts: float) -> Dict[str, object]:
    '''
    Compare the 5x5 augmented exponential with the independent integrals of
    section 11.2, and check the two algebraic identities for ``Gamma_j``.

    The kernel of ``Gamma_j`` is ``exp(A_m (Ts - tau)) B_a tau``: the Jerk is
    the derivative of the acceleration, so the displacement weights it by the
    remaining time ``Ts - tau``.  The second route uses
    ``Gamma_j = Ts Gamma_a - int tau exp(A_m tau) B_a dtau``, which follows from
    the same substitution and fails for any implementation that writes
    ``int exp(A_m tau) B_a tau dtau``.

    Returns:
        Dict[str, object]: Maximum absolute and relative deviations.
    '''
    A_m, B_u, B_a = continuous_mechanics(m, c, k, K_u)
    Phi, Gamma_u, Gamma_a, Gamma_j = discretize_first_order_hold(m, c, k, K_u, Ts)

    Phi_ref = scipy.linalg.expm(A_m * Ts)
    Gamma_u_ref = _integrate_matrix(lambda tau: scipy.linalg.expm(A_m * tau) @ B_u, Ts, (2, 1))
    Gamma_a_ref = _integrate_matrix(lambda tau: scipy.linalg.expm(A_m * tau) @ B_a, Ts, (2, 1))
    Gamma_j_ref = _integrate_matrix(
        lambda tau: scipy.linalg.expm(A_m * (Ts - tau)) @ B_a * tau, Ts, (2, 1))
    first_moment = _integrate_matrix(
        lambda tau: scipy.linalg.expm(A_m * tau) @ B_a * tau, Ts, (2, 1))
    Gamma_j_identity = Ts * Gamma_a_ref - first_moment

    errors = {
        'Phi': float(np.max(np.abs(Phi - Phi_ref))),
        'Gamma_u': float(np.max(np.abs(Gamma_u - Gamma_u_ref))),
        'Gamma_a': float(np.max(np.abs(Gamma_a - Gamma_a_ref))),
        'Gamma_j': float(np.max(np.abs(Gamma_j - Gamma_j_ref))),
        'Gamma_j_identity': float(np.max(np.abs(Gamma_j - Gamma_j_identity))),
    }
    scale_j = max(1e-30, float(np.max(np.abs(Gamma_j_ref))))
    scale_a = max(1e-30, float(np.max(np.abs(Gamma_a_ref))))
    errors['Gamma_relation'] = float(
        np.max(np.abs(Gamma_u - (K_u / m) * Gamma_a)) / max(scale_a, 1e-30))
    errors['Phi_relative'] = errors['Phi'] / max(1., float(np.max(np.abs(Phi_ref))))
    errors['Gamma_u_relative'] = errors['Gamma_u'] / max(1e-30, float(np.max(np.abs(Gamma_u_ref))))
    errors['Gamma_a_relative'] = errors['Gamma_a'] / scale_a
    errors['Gamma_j_relative'] = errors['Gamma_j'] / scale_j
    errors['Gamma_j_identity_relative'] = errors['Gamma_j_identity'] / scale_j
    errors['passed'] = (errors['Phi_relative'] < 1e-9
                        and errors['Gamma_u_relative'] < 1e-9
                        and errors['Gamma_a_relative'] < 1e-9
                        and errors['Gamma_j_relative'] < 1e-9
                        and errors['Gamma_j_identity_relative'] < 1e-9)
    return errors


def check_free_mass(q_b: float, Ts: float) -> Dict[str, object]:
    '''
    Reproduce the closed form free particle limit of equation (19-1).

    The mechanical matrix of a free mass is singular, so this check also
    proves that the discretisation never needs ``A_m`` to be invertible.
    '''
    A_m = np.array([[0., 1.], [0., 0.]])
    Phi, Q_m, _ = mechanical_process_noise(A_m, q_b, Ts)

    Phi_expected = np.array([[1., Ts], [0., 1.]])
    Q_expected = q_b * np.array([[Ts ** 3 / 3., Ts ** 2 / 2.],
                                 [Ts ** 2 / 2., Ts]])

    errors = {
        'Phi': float(np.max(np.abs(Phi - Phi_expected))),
        'Q_m': float(np.max(np.abs(Q_m - Q_expected))),
    }
    scale = max(1e-30, float(np.max(np.abs(Q_expected))))
    errors['Q_m_relative'] = errors['Q_m'] / scale
    errors['passed'] = errors['Phi'] < 1e-12 and errors['Q_m_relative'] < 1e-9
    return errors


def check_foh_free_mass(Ts: float) -> Dict[str, object]:
    '''
    Free particle limit of the FOH mechanical map.

    With ``A_m = [[0, 1], [0, 0]]`` and a constant Jerk the exact solution is
    ``x = x0 + v0 T + a0 T^2/2 + j0 T^3/6`` and ``v = v0 + a0 T + j0 T^2/2``,
    hence ``Gamma_a = [T^2/2, T]`` and ``Gamma_j = [T^3/6, T^2/2]``.  The
    ``T^3/6`` is what distinguishes the correct ``(Ts - tau)`` kernel from the
    incorrect ``int exp(A_m tau) B_a tau dtau``, which would give ``T^3/3``.
    '''
    A_m = np.array([[0., 1.], [0., 0.]])
    B_u = np.array([[0.], [0.]])
    B_a = np.array([[0.], [1.]])
    Phi, Gamma_u, Gamma_a, Gamma_j = foh_matrices(A_m, B_u, B_a, Ts)

    Phi_expected = np.array([[1., Ts], [0., 1.]])
    Gamma_a_expected = np.array([[0.5 * Ts ** 2], [Ts]])
    Gamma_j_expected = np.array([[Ts ** 3 / 6.], [0.5 * Ts ** 2]])

    errors = {
        'Phi': float(np.max(np.abs(Phi - Phi_expected))),
        'Gamma_u': float(np.max(np.abs(Gamma_u))),
        'Gamma_a': float(np.max(np.abs(Gamma_a - Gamma_a_expected))),
        'Gamma_j': float(np.max(np.abs(Gamma_j - Gamma_j_expected))),
    }
    scale = max(1e-30, float(np.max(np.abs(Gamma_j_expected))))
    errors['Gamma_j_relative'] = errors['Gamma_j'] / scale

    # Same statement as a trajectory check: one step of the discrete map with
    # a constant Jerk must reproduce the analytic polynomial.
    x0 = np.array([3e-7, -2e-3, 0.4])
    jerk = 12.5
    start = np.array([x0[0], x0[1]])
    propagated = Phi @ start + Gamma_a.ravel() * x0[2] + Gamma_j.ravel() * jerk
    analytic = np.array([x0[0] + x0[1] * Ts + x0[2] * Ts ** 2 / 2. + jerk * Ts ** 3 / 6.,
                         x0[1] + x0[2] * Ts + jerk * Ts ** 2 / 2.])
    errors['trajectory'] = float(np.max(np.abs(propagated - analytic)))
    errors['trajectory_relative'] = errors['trajectory'] / max(
        1e-30, float(np.max(np.abs(analytic))))

    errors['passed'] = (errors['Phi'] < 1e-12 and errors['Gamma_u'] < 1e-30
                        and errors['Gamma_a'] < 1e-15
                        and errors['Gamma_j_relative'] < 1e-12
                        and errors['trajectory_relative'] < 1e-12)
    return errors


def check_van_loan(model: ClosedLoopModel) -> Dict[str, object]:
    '''
    Compare the Van Loan covariance with a numerical integration of the
    continuous Lyapunov expression ``Q_m = int exp(A tau) W exp(A^T tau) dtau``.
    '''
    A_m, _, _ = continuous_mechanics(model.m, model.c, model.k, model.K_u)
    W_m = model.W_m
    Ts = model.Ts

    def integrand(tau):
        exponential = scipy.linalg.expm(A_m * tau)
        return exponential @ W_m @ exponential.T

    reference = _integrate_matrix(integrand, Ts, (2, 2))
    error = float(np.max(np.abs(model.Q_m - reference)))
    scale = max(1e-30, float(np.max(np.abs(reference))))
    eigen_min = float(np.min(np.linalg.eigvalsh(model.Q_m)))
    return {
        'max_abs_error': error,
        'relative_error': error / scale,
        'symmetry_error': float(np.max(np.abs(model.Q_m - model.Q_m.T))),
        'min_eigenvalue': eigen_min,
        'passed': error / scale < 1e-9 and eigen_min >= -1e-12 * scale,
    }


def _random_covariance(scale: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    '''
    Draw a random symmetric positive definite matrix with the given scales.
    '''
    rng = np.random.default_rng(seed)
    dimension = scale.size
    factor = rng.normal(size=(dimension, dimension))
    covariance = factor @ factor.T / dimension
    diagonal = np.diag(np.sqrt(scale))
    return symmetrize(diagonal @ covariance @ diagonal)


def check_prediction_equivalence(model: ClosedLoopModel, trials: int = 100,
                                 seed: int = 20261026) -> Dict[str, object]:
    '''
    Compare (14-7)-(14-9), (14-4)-(14-5) and (15-1)-(15-3) on random priors.

    All three expressions describe the same conditional prior ``X[k+1] | U[k]``.
    They differ only algebraically, so any difference beyond round off means
    one of them is implemented wrongly.  The same code covers the five and the
    six state model because the formulas are dimension independent.
    '''
    rng = np.random.default_rng(seed)
    scales = _state_scales(model)
    dim = model.dim

    mean_joint = 0.
    cov_joint = 0.
    mean_posterior = 0.
    cov_posterior = 0.
    state_scale = 0.
    covariance_scale = 0.

    for _ in range(trials):
        P = _random_covariance(scales, seed=int(rng.integers(1 << 30)))
        mu = rng.normal(size=dim) * np.sqrt(scales)
        u_k = float(rng.normal() * 1e-3)

        # Implemented recursion: one full step from a fresh estimator.
        estimator = kalman_module.ClosedLoopKalman(model, mu, P)
        estimator.step(u_k)
        mu_impl = estimator.mu
        P_impl = estimator.P

        mu_joint, P_joint = kalman_module.reference_predict_joint(model, mu, P, u_k)
        mu_post, P_post = kalman_module.reference_predict_posterior(model, mu, P, u_k)

        state_scale = max(state_scale, float(np.max(np.abs(mu_impl))))
        covariance_scale = max(covariance_scale, float(np.max(np.abs(P_impl))))
        mean_joint = max(mean_joint, float(np.max(np.abs(mu_impl - mu_joint))))
        cov_joint = max(cov_joint, float(np.max(np.abs(P_impl - P_joint))))
        mean_posterior = max(mean_posterior, float(np.max(np.abs(mu_impl - mu_post))))
        cov_posterior = max(cov_posterior, float(np.max(np.abs(P_impl - P_post))))

    mean_relative = max(mean_joint, mean_posterior) / max(state_scale, 1e-300)
    cov_relative = max(cov_joint, cov_posterior) / max(covariance_scale, 1e-300)
    return {
        'trials': trials,
        'dim': dim,
        'mean_vs_joint': mean_joint,
        'cov_vs_joint': cov_joint,
        'mean_vs_posterior': mean_posterior,
        'cov_vs_posterior': cov_posterior,
        'mean_relative': mean_relative,
        'cov_relative': cov_relative,
        'passed': mean_relative < 1e-9 and cov_relative < 1e-9,
    }


def check_monte_carlo(model: ClosedLoopModel, samples: int = 200000,
                      seed: int = 7) -> Dict[str, object]:
    '''
    Sample the joint Gaussian and verify that (14-9) is the conditional covariance.

    The same draw of ``n[k]`` drives the observation and the mechanical state,
    as required by section 19.1; drawing two independent noises would hide the
    correlation this estimator exists to handle.
    '''
    rng = np.random.default_rng(seed)
    dim = model.dim
    h = model.H.ravel()
    A = model.A
    G = model.G.ravel()
    L = model.L.ravel()

    scales = _state_scales(model)
    P = _random_covariance(scales, seed=seed + 1)
    mu = np.zeros(dim)

    root_P = np.linalg.cholesky(P + 1e-30 * np.eye(dim) * np.max(np.diag(P)))
    root_Q = np.linalg.cholesky(model.Q0 + 1e-30 * np.eye(dim) * np.max(np.diag(model.Q0))
                                + np.diag([1e-30] * dim))

    X = mu + rng.normal(size=(samples, dim)) @ root_P.T
    noise_n = rng.normal(scale=np.sqrt(model.r), size=samples)
    xi = rng.normal(size=(samples, dim)) @ root_Q.T

    u = X @ h + model.alpha * noise_n
    X_next = X @ A.T + np.outer(u, G) + np.outer(noise_n, L) + xi

    nu = u - h @ mu
    S = float(h @ P @ h + model.R)
    C_cl = model.F @ P @ h + model.N_cl.ravel()

    mu_next = model.F @ mu + np.outer(nu, C_cl) / S
    residual = X_next - mu_next

    covariance = np.cov(residual, rowvar=False)
    P_next_joint = model.F @ P @ model.F.T + model.Q_cl - np.outer(C_cl, C_cl) / S
    P_next_joint = symmetrize(P_next_joint)

    cross_max = 0.
    for column in range(dim):
        cross_max = max(cross_max, abs(float(np.cov(residual[:, column], nu)[0, 1])))

    mean_error = float(np.max(np.abs(np.mean(residual, axis=0))))
    covariance_error = float(np.max(np.abs(covariance - P_next_joint)))
    scale = max(1e-30, float(np.max(np.abs(P_next_joint))))
    cross_scale = max(1e-30, float(np.max(np.std(residual, axis=0)) * np.std(nu)))

    return {
        'samples': samples,
        'dim': dim,
        'mean_error': mean_error,
        'covariance_error': covariance_error,
        'covariance_relative_error': covariance_error / scale,
        'cross_covariance_max': cross_max,
        'cross_covariance_relative': cross_max / max(cross_scale, 1e-30),
        'innovation_std': float(np.std(nu)),
        'innovation_std_expected': float(np.sqrt(S)),
        'passed': (mean_error < 1e-2 * np.sqrt(scale)
                   and covariance_error < 2e-2 * scale
                   and cross_max < 2e-2 * cross_scale),
    }


def check_foh_noise_timing(model: ClosedLoopModel) -> Dict[str, object]:
    '''
    Section 14.2 / 19.1: where the Jerk increment enters, and where it does not.

    Two independent statements are checked:

    1. one step of the filter stores a next prior whose difference against the
       same step with ``q_j = 0`` is exactly ``dq_j e_j e_j^T`` -- the increment
       raises the Jerk variance only, never ``T^2 q_j`` on the acceleration of
       the same step;
    2. the deterministic one step propagation ``A[:, j]`` is
       ``[Gamma_j, T, 1, 0, 0]``, so the increment enters ``a`` with the factor
       ``T`` and ``x, v`` with ``Gamma_j`` on the *next* step only.
    '''
    if not model.has_jerk:
        raise ValueError('the FOH noise timing check needs the six state model')

    dim = model.dim
    i_a = model.index['a']
    i_j = model.index['j']
    scales = _state_scales(model)
    P0 = np.diag(scales)
    mu0 = np.zeros(dim)

    increment = 0.5 * float(model.q_j) + 1.
    base = model
    raised = model.with_q_j(model.q_j + increment)

    first = kalman_module.ClosedLoopKalman(base, mu0, P0)
    first.step(1e-7)
    second = kalman_module.ClosedLoopKalman(raised, mu0, P0)
    second.step(1e-7)

    difference = second.P - first.P
    expected = np.zeros((dim, dim))
    expected[i_j, i_j] = increment
    scale = max(1e-30, float(np.max(np.abs(expected))))
    injection_error = float(np.max(np.abs(difference - expected)))
    acceleration_leak = float(abs(difference[i_a, i_a]))
    acceleration_scale = max(1e-30, float(np.max(np.abs(difference))))

    direction = model.A[:, i_j]
    expected_direction = np.zeros(dim)
    expected_direction[0:2] = model.Gamma_j.ravel()
    expected_direction[i_a] = model.Ts
    expected_direction[i_j] = 1.
    direction_error = float(np.max(np.abs(direction - expected_direction)))
    direction_scale = max(1e-30, float(np.max(np.abs(expected_direction))))

    propagated = model.A @ expected @ model.A.T
    propagation = {
        'a_a': float(propagated[i_a, i_a]),
        'a_j': float(propagated[i_a, i_j]),
        'j_j': float(propagated[i_j, i_j]),
        'a_a_expected': float(model.Ts ** 2 * increment),
        'a_j_expected': float(model.Ts * increment),
        'j_j_expected': float(increment),
    }
    # Every entry is compared with its own predicted value: the acceleration
    # entry carries the tiny factor T^2, so a common denominator would either
    # hide an error there or report pure rounding on the Jerk entry.
    propagation_relative = 0.
    for name in ('a_a', 'a_j', 'j_j'):
        expected_value = propagation[f'{name}_expected']
        deviation = abs(propagation[name] - expected_value)
        propagation_relative = max(
            propagation_relative,
            deviation / max(abs(expected_value), 1e-300))

    return {
        'increment': increment,
        'injection_error': injection_error,
        'injection_relative': injection_error / scale,
        'acceleration_leak_relative': acceleration_leak / acceleration_scale,
        'direction_error': direction_error,
        'direction_relative': direction_error / direction_scale,
        'propagation': propagation,
        'propagation_relative': propagation_relative,
        'passed': (injection_error / scale < 1e-10
                   and acceleration_leak / acceleration_scale < 1e-10
                   and direction_error / direction_scale < 1e-12
                   and propagation_relative < 1e-10),
    }


def zoh_counterpart(config) -> 'object':
    '''
    Build the ZOH configuration that the FOH path degenerates to, section 15.2.

    The Jerk must be known to be zero and ``q_j = 0``; then the common states
    follow the ZOH model with ``q_a = 0``.  The five state prior is the common
    block of the six state prior.
    '''
    foh_prior = config.prior_covariance
    common = [0, 1, 2, 4, 5]
    common_prior = foh_prior[np.ix_(common, common)]
    common_mean = config.prior_mean[common]
    scales = None if config.state_scales is None else np.asarray(
        config.state_scales, dtype=np.float64)[common]

    return with_prior_model(
        config, 'ZOH',
        q_a=0.,
        X0_hat=np.asarray(common_mean, dtype=np.float64),
        P0=np.asarray(common_prior, dtype=np.float64),
        X0_FOH=None,
        P0_FOH=None,
        state_scales=scales,
    )


def check_foh_zoh_degeneracy(config, steps: int = 120,
                             seed: int = 20261026) -> Dict[str, object]:
    '''
    Section 15.2 / 19.1: with a known zero Jerk and ``q_j = 0`` the common
    states must equal the ZOH model with ``q_a = 0``.

    The degenerate configuration is built here, not taken from the file: the
    check is about the limit ``j0 = 0``, ``q_j = 0``, so the Jerk mean and the
    Jerk row of the prior are zeroed even when the file describes an unknown
    Jerk.  The two estimators are driven by the same PID output sequence; only
    the common states, the common covariance blocks and the innovation are
    compared.
    '''
    degenerate = copy.deepcopy(config)
    degenerate.q_j = 0.
    degenerate.jerk0_hat = 0.
    mean = np.array(config.prior_mean, dtype=np.float64, copy=True)
    prior = np.array(config.prior_covariance, dtype=np.float64, copy=True)
    i_j = degenerate.index['j']
    mean[i_j] = 0.
    prior[i_j, :] = 0.
    prior[:, i_j] = 0.
    degenerate.X0_FOH = mean
    degenerate.P0_FOH = symmetrize(prior)

    foh_model = ClosedLoopModel(degenerate)
    zoh_config = zoh_counterpart(degenerate)
    zoh_model = ClosedLoopModel(zoh_config)

    common = [0, 1, 2, 4, 5]
    rng = np.random.default_rng(seed)
    sequence = rng.normal(scale=1e-6, size=steps)

    foh = kalman_module.ClosedLoopKalman.initialize(degenerate)
    zoh = kalman_module.ClosedLoopKalman.initialize(zoh_config)
    if foh_model.dim != 6 or zoh_model.dim != 5:
        raise ValueError('the degeneracy check needs a six and a five state model')

    mean_error = 0.
    covariance_error = 0.
    innovation_error = 0.
    jerk_error = 0.
    scale_mean = 0.
    scale_covariance = 0.

    for value in sequence:
        output_foh = foh.step(float(value))
        output_zoh = zoh.step(float(value))
        foh_mean = output_foh.state[common]
        mean_error = max(mean_error, float(np.max(np.abs(foh_mean - output_zoh.state))))
        scale_mean = max(scale_mean, float(np.max(np.abs(output_zoh.state))))
        foh_cov = output_foh.covariance[np.ix_(common, common)]
        covariance_error = max(
            covariance_error, float(np.max(np.abs(foh_cov - output_zoh.covariance))))
        scale_covariance = max(scale_covariance,
                              float(np.max(np.abs(output_zoh.covariance))))
        innovation_error = max(
            innovation_error,
            abs(float(output_foh.innovation) - float(output_zoh.innovation)))
        jerk_error = max(jerk_error, abs(float(output_foh.jerk)))

    return {
        'steps': steps,
        'mean_error': mean_error,
        'mean_relative': mean_error / max(scale_mean, 1e-300),
        'covariance_error': covariance_error,
        'covariance_relative': covariance_error / max(scale_covariance, 1e-300),
        'innovation_error': innovation_error,
        'jerk_error': jerk_error,
        'passed': (mean_error / max(scale_mean, 1e-300) < 1e-9
                   and covariance_error / max(scale_covariance, 1e-300) < 1e-9
                   and innovation_error < 1e-9
                   and jerk_error < 1e-12),
    }


def check_foh_deterministic(config, steps: int = 60,
                            seed: int = 20261026) -> Dict[str, object]:
    '''
    Section 19.1: the deterministic limit of the FOH recursion.

    With ``r = 0``, no mechanical noise and ``q_j = 0`` the nominal trajectory
    of a constant Jerk interval is reproduced exactly, including the zero
    innovation variance path.  The trajectory is generated by the model itself,
    so this is an implementation consistency check, not a performance test.
    '''
    deterministic = copy.deepcopy(config)
    deterministic.sigma_n2 = 0.
    deterministic.r = 0.
    deterministic.R = 0.
    deterministic.mechanical_noise_psd = 0.
    deterministic.q_b = 0.
    deterministic.q_F = 0.
    deterministic.q_j = 0.

    model = ClosedLoopModel(deterministic)
    dim = model.dim
    i_j = model.index['j']

    rng = np.random.default_rng(seed)
    state = rng.normal(size=dim) * np.array([1e-8, 1e-5, 1e-1, 1e-1, 1e-4, 1e-6])
    # A constant Jerk keeps the interval model exact; the acceleration then
    # grows linearly, which is what the augmented map assumes.
    estimator = kalman_module.ClosedLoopKalman(
        model, state, np.zeros((dim, dim)), deterministic.tolerances)

    mean_error = 0.
    scale = 0.
    for _ in range(steps):
        u_k = float(model.H.ravel() @ state)
        output = estimator.step(u_k)
        mean_error = max(mean_error, float(np.max(np.abs(output.state - state))))
        scale = max(scale, float(np.max(np.abs(state))))
        state = model.A @ state + model.G.ravel() * u_k

    return {
        'steps': steps,
        'mean_error': mean_error,
        'mean_relative': mean_error / max(scale, 1e-300),
        'final_jerk': float(state[i_j]),
        'passed': mean_error / max(scale, 1e-300) < 1e-9,
    }


def check_foh_static_unobservable_direction(model: ClosedLoopModel) -> Dict[str, object]:
    '''
    Section 17.2: with ``K_i = 0`` the FOH model keeps a static unobservable
    direction ``[1, 0, k/m, 0, -K_p K_s, K_s]``.

    The check is the direct identity ``H v_dc = 0`` and ``F v_dc = v_dc``; it is
    reported for every FOH parameter set because it explains the shape of the
    observability result.
    '''
    if model.K_i != 0:
        return {'applicable': False, 'passed': True}

    dim = model.dim
    direction = np.zeros(dim)
    direction[0] = 1.
    direction[model.index['a']] = model.k / model.m
    direction[model.index['I']] = -model.config.K_p * model.k_s
    direction[model.index['h']] = model.k_s

    residual_h = float(np.max(np.abs(model.H.ravel() @ direction)))
    residual_f = float(np.max(np.abs((model.F - np.eye(dim)) @ direction)))
    return {
        'applicable': True,
        'direction': direction.tolist(),
        'H_residual': residual_h,
        'F_residual': residual_f,
        'passed': residual_h < 1e-12 and residual_f < 1e-12 * max(1., np.max(np.abs(direction))),
    }


def run_all(config) -> Dict[str, object]:
    '''
    Run every algebraic check on a validated configuration.

    Args:
        config: A :class:`config.Config` instance.

    Returns:
        Dict[str, object]: Named results plus an overall ``passed`` flag.
    '''
    model = ClosedLoopModel(config)
    foh = model.has_jerk

    results = {
        'prior_model': model.prior_model,
        'dim': model.dim,
        'van_loan': check_van_loan(model),
        'prediction_equivalence': check_prediction_equivalence(model),
        'monte_carlo': check_monte_carlo(model),
        'observability': model.observability_report().as_dict(),
    }
    if foh:
        results['discretization'] = check_foh_discretization(
            model.m, model.c, model.k, model.K_u, model.Ts)
        results['free_mass'] = check_foh_free_mass(model.Ts)
        results['noise_timing'] = check_foh_noise_timing(model)
        results['zoh_degeneracy'] = check_foh_zoh_degeneracy(config)
        results['deterministic'] = check_foh_deterministic(config)
        results['static_unobservable'] = check_foh_static_unobservable_direction(model)
    else:
        results['discretization'] = check_discretization(
            model.m, model.c, model.k, model.K_u, model.Ts)
        results['free_mass'] = check_free_mass(model.q_b, model.Ts)

    checked = [key for key in ('discretization', 'free_mass', 'van_loan',
                               'prediction_equivalence', 'monte_carlo',
                               'noise_timing', 'zoh_degeneracy',
                               'deterministic', 'static_unobservable')
               if key in results]
    results['passed'] = all(bool(results[key].get('passed', True)) for key in checked)
    return results
