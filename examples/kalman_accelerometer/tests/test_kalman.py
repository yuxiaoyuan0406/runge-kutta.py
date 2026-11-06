'''
Recursion tests: equivalence of the three prediction forms, the Monte Carlo
conditional check, index boundaries, the q_a sweep and the singular path.
'''
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.kalman_accelerometer import config as config_module  # noqa: E402
from examples.kalman_accelerometer import kalman as kalman_module  # noqa: E402
from examples.kalman_accelerometer import model as model_module  # noqa: E402
from examples.kalman_accelerometer import verification  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'configs', 'algebraic-check.json')


def make_config(**overrides):
    '''
    Build the synthetic configuration, optionally overriding entries.
    '''
    with open(CONFIG_PATH, 'r', encoding='utf-8', errors='replace') as handle:
        raw = json.load(handle)
    raw.update(overrides)
    return config_module.build_config(raw)


class TestPredictionEquivalence(unittest.TestCase):
    '''
    Section 14.3 and 19.1: (14-4)-(14-5), (14-7)-(14-9) and (15-2)-(15-3)
    must produce the same next prior.
    '''

    def test_three_forms_agree(self):
        model = model_module.ClosedLoopModel(make_config())
        result = verification.check_prediction_equivalence(model, trials=60)
        self.assertTrue(result['passed'], result)
        self.assertLess(result['mean_relative'], 1e-10)
        self.assertLess(result['cov_relative'], 1e-10)


class TestMonteCarlo(unittest.TestCase):
    '''
    Section 19.1: sampling the joint Gaussian must reproduce (14-9).
    '''

    def test_conditional_moments(self):
        model = model_module.ClosedLoopModel(make_config())
        result = verification.check_monte_carlo(model, samples=60000)
        self.assertTrue(result['passed'], result)

    def test_innovation_std_matches_sqrt_s(self):
        model = model_module.ClosedLoopModel(make_config())
        result = verification.check_monte_carlo(model, samples=60000)
        ratio = result['innovation_std'] / result['innovation_std_expected']
        self.assertAlmostEqual(ratio, 1., delta=0.02)


class TestIndexBoundary(unittest.TestCase):
    '''
    Section 19.1: the first output is the posterior of sample 0, not a
    prediction.  A consistent non-singular prior reproduces the observation.
    '''

    def setUp(self):
        self.config = make_config()
        self.model = model_module.ClosedLoopModel(self.config)

    def test_first_output_matches_sample_zero(self):
        x0 = np.array([1e-8, -3e-6, 0.2, 1e-3, 2e-6])
        P0 = np.diag([1e-20, 1e-14, 1e-6, 1e-8, 1e-14])
        estimator = kalman_module.ClosedLoopKalman(self.model, x0, P0)

        u0 = float(self.model.H.ravel() @ x0)
        output = estimator.step(u0)

        self.assertEqual(output.index, 0)
        self.assertAlmostEqual(output.acceleration, x0[2], places=14)
        self.assertAlmostEqual(output.displacement, x0[0], places=20)
        self.assertAlmostEqual(output.innovation, 0., places=12)

    def test_second_step_moves_to_the_next_prior(self):
        x0 = np.zeros(5)
        P0 = np.diag([1e-20, 1e-14, 1e-6, 1e-8, 1e-14])
        estimator = kalman_module.ClosedLoopKalman(self.model, x0, P0)

        estimator.step(0.)
        stored_mean = estimator.mu.copy()
        stored_cov = estimator.P.copy()

        estimator.step(0.)
        self.assertEqual(estimator.index, 2)
        # The stored prior before the second step is the one produced by the
        # first step; both must be finite and symmetric.
        self.assertTrue(np.all(np.isfinite(stored_mean)))
        np.testing.assert_allclose(stored_cov, stored_cov.T, rtol=0., atol=1e-24)

    def test_run_length(self):
        x0 = np.zeros(5)
        P0 = np.diag([1e-20, 1e-14, 1e-6, 1e-8, 1e-14])
        estimator = kalman_module.ClosedLoopKalman(self.model, x0, P0)
        history = estimator.run([0., 1e-6, -2e-6, 3e-6])
        self.assertEqual(history['acceleration'].shape, (4,))
        self.assertEqual(history['state'].shape, (4, 5))
        np.testing.assert_array_equal(history['index'], np.arange(4))


class TestQaSensitivity(unittest.TestCase):
    '''
    Section 19.1: increasing q_a adds to the acceleration prior variance of the
    very step that produced it, and to nothing else.
    '''

    def test_increment_lands_on_the_acceleration_entry(self):
        config = make_config()
        base = model_module.ClosedLoopModel(config)
        scaled = base.with_q_a(config.q_a * 3.)

        x0 = np.zeros(5)
        P0 = np.diag([1e-20, 1e-14, 1e-6, 1e-8, 1e-14])

        first = kalman_module.ClosedLoopKalman(base, x0, P0)
        second = kalman_module.ClosedLoopKalman(scaled, x0, P0)
        first.step(1e-7)
        second.step(1e-7)

        difference = second.P - first.P
        expected = np.zeros((5, 5))
        expected[2, 2] = config.q_a * 2.

        np.testing.assert_allclose(difference, expected, rtol=1e-10, atol=1e-24)
        # The mechanical block must not be touched by q_a directly.
        self.assertTrue(np.allclose(difference[0:2, 0:2], 0., atol=1e-26))


class TestSingularObservation(unittest.TestCase):
    '''
    Section 18: a degenerate innovation variance with a consistent innovation
    is skipped, an inconsistent one is an error.
    '''

    def _degenerate_estimator(self):
        config = make_config(sigma_n2=0.)
        model = model_module.ClosedLoopModel(config)
        estimator = kalman_module.ClosedLoopKalman(
            model, np.zeros(5), np.zeros((5, 5)), config.tolerances)
        return estimator

    def test_consistent_zero_variance_is_skipped(self):
        estimator = self._degenerate_estimator()
        output = estimator.step(0.)
        self.assertTrue(output.singular)
        np.testing.assert_allclose(output.gain, np.zeros(5), rtol=0., atol=0.)
        np.testing.assert_allclose(output.state, np.zeros(5), rtol=0., atol=0.)
        expected = estimator.model.Q0 + estimator.model.r * np.outer(
            estimator.model.L.ravel(), estimator.model.L.ravel())
        np.testing.assert_allclose(estimator.P, expected, rtol=1e-12, atol=1e-24)

    def test_inconsistent_zero_variance_raises(self):
        estimator = self._degenerate_estimator()
        with self.assertRaises(kalman_module.FilterError):
            estimator.step(1.)


class TestNoiselessNominalModel(unittest.TestCase):
    '''
    Section 6.3: with no readout noise and no process noise, a consistent
    prior must be reproduced exactly by the recursion.
    '''

    def test_estimate_follows_the_nominal_trajectory(self):
        config = make_config(sigma_n2=0., q_a=0., mechanical_noise_psd=0.)
        model = model_module.ClosedLoopModel(config)

        x0 = np.array([1e-9, 2e-6, 0.3, 1e-4, -2e-7])
        P0 = np.zeros((5, 5))
        estimator = kalman_module.ClosedLoopKalman(model, x0, P0, config.tolerances)

        state = x0.copy()
        for _ in range(50):
            u_k = float(model.H.ravel() @ state)
            output = estimator.step(u_k)
            # The observation is consistent with the prior, so the posterior
            # equals the prior and the stored prediction is exact.
            np.testing.assert_allclose(output.state, state, rtol=0., atol=1e-18)
            state = model.A @ state + model.G.ravel() * u_k

        np.testing.assert_allclose(estimator.mu, state, rtol=0., atol=1e-18)
        self.assertTrue(np.allclose(estimator.P, 0., atol=1e-30))


class TestInputGuards(unittest.TestCase):
    '''
    Section 16 step 1: non finite values must stop the recursion.
    '''

    def setUp(self):
        config = make_config()
        model = model_module.ClosedLoopModel(config)
        self.estimator = kalman_module.ClosedLoopKalman(
            model, np.zeros(5), np.diag([1e-20, 1e-14, 1e-6, 1e-8, 1e-14]))

    def test_nan_input(self):
        with self.assertRaises(kalman_module.FilterError):
            self.estimator.step(float('nan'))

    def test_infinite_input(self):
        with self.assertRaises(kalman_module.FilterError):
            self.estimator.step(float('inf'))

    def test_bad_prior_shape(self):
        with self.assertRaises(kalman_module.FilterError):
            kalman_module.ClosedLoopKalman(self.estimator.model, np.zeros(4),
                                           np.eye(4))

    def test_non_psd_prior(self):
        with self.assertRaises(kalman_module.FilterError):
            kalman_module.ClosedLoopKalman(self.estimator.model, np.zeros(5),
                                           -np.eye(5))


class TestCovarianceHealth(unittest.TestCase):
    '''
    Section 4.3: every stored covariance stays symmetric and positive semi
    definite over a longer run.
    '''

    def test_run_keeps_covariances_healthy(self):
        config = make_config()
        model = model_module.ClosedLoopModel(config)
        estimator = kalman_module.ClosedLoopKalman(
            model, np.zeros(5), np.diag([1e-20, 1e-14, 1e-6, 1e-8, 1e-14]))

        rng = np.random.default_rng(11)
        sequence = rng.normal(scale=1e-6, size=400)
        for value in sequence:
            estimator.step(value)
            covariance = estimator.P
            np.testing.assert_allclose(covariance, covariance.T, rtol=0., atol=1e-24)
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(covariance))), -1e-18)

        history = estimator.history
        posterior = np.asarray(history['posterior_variance'])
        self.assertTrue(np.all(posterior >= -1e-18))


if __name__ == '__main__':
    unittest.main()
