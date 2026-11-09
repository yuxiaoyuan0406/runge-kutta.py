'''
FOH recursion tests, sections 13, 15, 16, 18 and 19.1 of the V2.1 task sheet.
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

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'configs')
FOH_PATH = os.path.join(CONFIG_DIR, 'algebraic-check-foh.json')
ZOH_PATH = os.path.join(CONFIG_DIR, 'algebraic-check.json')


def make_config(prior_model='FOH', **overrides):
    '''
    Build the synthetic configuration of the requested prior model.
    '''
    path = FOH_PATH if prior_model == 'FOH' else ZOH_PATH
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        raw = json.load(handle)
    raw.update(overrides)
    return config_module.build_config(raw)


class TestPredictionEquivalence(unittest.TestCase):
    '''
    Section 19.1: (14-4)-(14-5), (14-7)-(14-9) and (15-2)-(15-3) must produce
    the same next prior, now for the six state model.
    '''

    def test_three_forms_agree(self):
        model = model_module.ClosedLoopModel(make_config())
        result = verification.check_prediction_equivalence(model, trials=60)
        self.assertTrue(result['passed'], result)
        self.assertLess(result['mean_relative'], 1e-10)
        self.assertLess(result['cov_relative'], 1e-10)

    def test_matrix_covers_six_states(self):
        model = model_module.ClosedLoopModel(make_config())
        estimator = kalman_module.ClosedLoopKalman.initialize(model.config)
        estimator.step(1e-7)
        self.assertEqual(estimator.P.shape, (6, 6))
        self.assertEqual(estimator.mu.shape, (6,))


class TestMonteCarlo(unittest.TestCase):
    '''
    Section 19.1: sampling the joint Gaussian must reproduce (14-9).
    '''

    def test_conditional_moments(self):
        model = model_module.ClosedLoopModel(make_config())
        result = verification.check_monte_carlo(model, samples=80000)
        self.assertTrue(result['passed'], result)

    def test_innovation_std_matches_sqrt_s(self):
        model = model_module.ClosedLoopModel(make_config())
        result = verification.check_monte_carlo(model, samples=80000)
        ratio = result['innovation_std'] / result['innovation_std_expected']
        self.assertAlmostEqual(ratio, 1., delta=0.02)


class TestBoundaryTiming(unittest.TestCase):
    '''
    Section 14.2: the boundary increment raises only the Jerk variance of that
    step; the acceleration follows one step later with the factor T.
    '''

    def setUp(self):
        self.model = model_module.ClosedLoopModel(make_config())

    def test_increment_stays_on_the_jerk_diagonal(self):
        result = verification.check_foh_noise_timing(self.model)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['acceleration_leak_relative'], 0.)
        self.assertGreater(result['propagation']['a_a_expected'], 0.)
        self.assertAlmostEqual(result['propagation']['a_a'] /
                               result['propagation']['a_j'],
                               self.model.Ts, places=12)

    def test_q_j_does_not_touch_the_acceleration_of_the_same_step(self):
        scales = verification._state_scales(self.model)
        P0 = np.diag(scales)
        base = kalman_module.ClosedLoopKalman(self.model, np.zeros(6), P0)
        raised_model = self.model.with_q_j(self.model.q_j + 2.)
        raised = kalman_module.ClosedLoopKalman(raised_model, np.zeros(6), P0)
        base.step(0.)
        raised.step(0.)
        difference = raised.P - base.P
        self.assertAlmostEqual(difference[2, 2], 0., places=24)
        self.assertAlmostEqual(difference[2, 3], 0., places=24)
        self.assertAlmostEqual(difference[3, 3], 2., places=12)


class TestZohDegeneracy(unittest.TestCase):
    '''
    Section 15.2: a known zero Jerk with q_j = 0 degenerates to the ZOH model
    with q_a = 0 on the common states.
    '''

    def test_common_states_match_zoh(self):
        config = make_config(jerk_init_mode='known', jerk_var0=0.0, q_j=0.0)
        result = verification.check_foh_zoh_degeneracy(config)
        self.assertTrue(result['passed'], result)
        self.assertLess(result['mean_relative'], 1e-11)
        self.assertLess(result['covariance_relative'], 1e-11)

    def test_unknown_jerk_is_not_the_degenerate_case(self):
        config = make_config()  # jerk_init_mode = unknown, q_j > 0
        zoh = verification.zoh_counterpart(config)
        self.assertEqual(zoh.prior_mean.shape, (5,))
        self.assertEqual(zoh.q_a, 0.)


class TestDeterministicLimit(unittest.TestCase):
    '''
    Section 19.1: with r, mechanical noise and q_j zero the constant Jerk
    trajectory is reproduced exactly.
    '''

    def test_known_prior_is_reproduced(self):
        config = make_config()
        result = verification.check_foh_deterministic(config)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['mean_error'], 0.)


class TestJerkOutput(unittest.TestCase):
    '''
    Section 5 and 16: the posterior Jerk and its variance are appended to the
    output, with the same index k as the phase 1 fields.
    '''

    def test_step_output_has_jerk_fields(self):
        config = make_config()
        estimator = kalman_module.ClosedLoopKalman.initialize(config)
        output = estimator.step(1e-7)
        self.assertIsNotNone(output.jerk)
        self.assertIsNotNone(output.jerk_variance)
        self.assertGreater(output.jerk_variance, 0.)
        self.assertAlmostEqual(output.jerk, float(output.state[3]), places=15)
        self.assertAlmostEqual(output.jerk_variance,
                               float(output.covariance[3, 3]), places=20)
        self.assertIsNotNone(output.jerk_accel_correlation)
        self.assertLessEqual(abs(output.jerk_accel_correlation), 1. + 1e-12)

    def test_history_and_dict_layout(self):
        config = make_config()
        estimator = kalman_module.ClosedLoopKalman.initialize(config)
        history = estimator.run([0., 1e-7, -2e-7])
        self.assertEqual(history['jerk'].shape, (3,))
        self.assertEqual(history['jerk_variance'].shape, (3,))
        self.assertEqual(history['state'].shape, (3, 6))
        payload = estimator.step(0.).as_dict()
        for key in ('jerk', 'jerk_variance', 'jerk_accel_correlation'):
            self.assertIn(key, payload)

    def test_undefined_correlation_is_reported(self):
        config = make_config(sigma_n2=0., mechanical_noise_psd=0., q_j=0.,
                             jerk_init_mode='known', jerk_var0=0.)
        model = model_module.ClosedLoopModel(config)
        state = np.zeros(6)
        state[3] = 0.1
        estimator = kalman_module.ClosedLoopKalman(
            model, state, np.zeros((6, 6)), config.tolerances)
        output = estimator.step(float(model.H.ravel() @ state))
        self.assertIsNone(output.jerk_accel_correlation)
        self.assertIn('jerk_accel_correlation', output.as_dict())
        self.assertIsNone(output.as_dict()['jerk_accel_correlation'])
        history = estimator.run([0., 0.])
        self.assertTrue(np.all(np.isnan(history['jerk_accel_correlation'])))

    def test_zoh_has_no_jerk_fields(self):
        config = make_config('ZOH')
        estimator = kalman_module.ClosedLoopKalman.initialize(config)
        history = estimator.run([0., 1e-7])
        self.assertNotIn('jerk', history)
        output = estimator.step(0.)
        self.assertIsNone(output.jerk)
        self.assertNotIn('jerk', output.as_dict())


class TestInputGuards(unittest.TestCase):
    '''
    Section 16: wrong dimensions and non finite inputs stop the recursion.
    '''

    def setUp(self):
        self.config = make_config()
        self.model = model_module.ClosedLoopModel(self.config)
        self.P0 = np.diag(verification._state_scales(self.model))

    def test_five_state_prior_is_rejected(self):
        with self.assertRaises(kalman_module.FilterError):
            kalman_module.ClosedLoopKalman(self.model, np.zeros(5),
                                           np.eye(5))

    def test_non_finite_input(self):
        estimator = kalman_module.ClosedLoopKalman(self.model, np.zeros(6), self.P0)
        with self.assertRaises(kalman_module.FilterError):
            estimator.step(float('nan'))


class TestCovarianceHealth(unittest.TestCase):
    '''
    Section 16: every stored covariance stays symmetric and positive semi
    definite over a longer run.
    '''

    def test_run_keeps_covariances_healthy(self):
        model = model_module.ClosedLoopModel(make_config())
        estimator = kalman_module.ClosedLoopKalman.initialize(model.config)
        rng = np.random.default_rng(5)
        for value in rng.normal(scale=1e-6, size=400):
            estimator.step(float(value))
            covariance = estimator.P
            np.testing.assert_allclose(covariance, covariance.T, rtol=0., atol=1e-24)
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(covariance))),
                                    -1e-18)
        self.assertTrue(np.all(np.asarray(estimator.history['jerk_variance']) >= 0))


if __name__ == '__main__':
    unittest.main()
