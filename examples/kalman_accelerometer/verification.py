'''
Algebraic verification of the implementation, section 19.1 of the task sheet.

Every check in this module compares two independent routes to the same
quantity.  Nothing here is a performance claim: the routines only prove that
the discretisation, the covariances and the recursive formulas agree with
their definitions.
'''
from typing import Dict, Optional

import numpy as np
import scipy.integrate
import scipy.linalg

from . import kalman as kalman_module
from .model import (ClosedLoopModel, continuous_mechanics, discretize_zero_order_hold,
                    mechanical_process_noise, symmetrize)


def _integrate_matrix(function, Ts: float, shape: tuple) -> np.ndarray:
    '''
    Numerically integrate a matrix valued function on ``[0, Ts]``.
    '''
    def wrapper(tau):
        return np.asarray(function(tau), dtype=np.float64).ravel()

    values, _ = scipy.integrate.quad_vec(wrapper, 0., Ts, epsabs=1e-14, epsrel=1e-12)
    return np.asarray(values, dtype=np.float64).reshape(shape)


def check_discretization(m: float, c: float, k: float, K_u: float, Ts: float) -> Dict[str, object]:
    '''
    Compare the augmented exponential with an independent numerical integral.

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
    one of them is implemented wrongly.

    Returns:
        Dict[str, object]: Maximum absolute deviations of the mean and of the
        covariance between the implemented recursion, the joint form and the
        posterior form.
    '''
    rng = np.random.default_rng(seed)
    scales = np.array([1e-18, 1e-12, 1e-4, 1e-4, 1e-6])

    mean_joint = 0.
    cov_joint = 0.
    mean_posterior = 0.
    cov_posterior = 0.
    state_scale = 0.
    covariance_scale = 0.

    for _ in range(trials):
        P = _random_covariance(scales, seed=int(rng.integers(1 << 30)))
        mu = rng.normal(size=5) * np.sqrt(scales)
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
    h = model.H.ravel()
    A = model.A
    G = model.G.ravel()
    L = model.L.ravel()

    scales = np.array([1e-18, 1e-12, 1e-4, 1e-4, 1e-6])
    P = _random_covariance(scales, seed=seed + 1)
    mu = np.zeros(5)

    root_P = np.linalg.cholesky(P + 1e-30 * np.eye(5) * np.max(np.diag(P)))
    root_Q = np.linalg.cholesky(model.Q0 + 1e-30 * np.eye(5) * np.max(np.diag(model.Q0))
                                + np.diag([1e-30] * 5))

    X = mu + rng.normal(size=(samples, 5)) @ root_P.T
    noise_n = rng.normal(scale=np.sqrt(model.r), size=samples)
    xi = rng.normal(size=(samples, 5)) @ root_Q.T

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
    for column in range(5):
        cross_max = max(cross_max, abs(float(np.cov(residual[:, column], nu)[0, 1])))

    mean_error = float(np.max(np.abs(np.mean(residual, axis=0))))
    covariance_error = float(np.max(np.abs(covariance - P_next_joint)))
    scale = max(1e-30, float(np.max(np.abs(P_next_joint))))
    cross_scale = max(1e-30, float(np.max(np.std(residual, axis=0)) * np.std(nu)))

    return {
        'samples': samples,
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


def run_all(config) -> Dict[str, object]:
    '''
    Run every algebraic check on a validated configuration.

    Args:
        config: A :class:`config.Config` instance.

    Returns:
        Dict[str, object]: Named results plus an overall ``passed`` flag.
    '''
    model = ClosedLoopModel(config)

    results = {
        'discretization': check_discretization(model.m, model.c, model.k,
                                               model.K_u, model.Ts),
        'free_mass': check_free_mass(model.q_b, model.Ts),
        'van_loan': check_van_loan(model),
        'prediction_equivalence': check_prediction_equivalence(model),
        'monte_carlo': check_monte_carlo(model),
        'observability': model.observability_report().as_dict(),
    }
    results['passed'] = all(bool(results[key].get('passed', True))
                            for key in ('discretization', 'free_mass', 'van_loan',
                                        'prediction_equivalence', 'monte_carlo'))
    return results
