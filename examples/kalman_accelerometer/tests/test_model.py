'''
Discretisation and matrix layout tests, sections 12, 13 and 19.1.
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


class TestDiscretization(unittest.TestCase):
    '''
    Section 12.1 and equation (19-1).
    '''

    def setUp(self):
        self.config = make_config()
        self.model = model_module.ClosedLoopModel(self.config)

    def test_augmented_exponential_matches_integration(self):
        result = verification.check_discretization(
            self.model.m, self.model.c, self.model.k, self.model.K_u, self.model.Ts)
        self.assertTrue(result['passed'], result)

    def test_gamma_relation(self):
        ratio = self.model.K_u / self.model.m
        np.testing.assert_allclose(self.model.Gamma_u.ravel(),
                                   ratio * self.model.Gamma_a.ravel(),
                                   rtol=1e-12, atol=1e-18)

    def test_free_mass_closed_form(self):
        result = verification.check_free_mass(self.model.q_b, self.model.Ts)
        self.assertTrue(result['passed'], result)

    def test_van_loan_matches_integration(self):
        result = verification.check_van_loan(self.model)
        self.assertTrue(result['passed'], result)

    def test_mechanical_covariance_is_symmetric(self):
        self.assertLess(float(np.max(np.abs(self.model.Q_m - self.model.Q_m.T))), 1e-24)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(self.model.Q_m))), -1e-24)


class TestStateMatrices(unittest.TestCase):
    '''
    Section 13.1: explicit layout of A, G, L, H and Q0.
    '''

    def setUp(self):
        self.config = make_config()
        self.model = model_module.ClosedLoopModel(self.config)

    def test_state_matrix_layout(self):
        model = self.model
        phi = model.Phi
        ga = model.Gamma_a.ravel()

        expected = np.zeros((5, 5))
        expected[0, 0], expected[0, 1] = phi[0, 0], phi[0, 1]
        expected[1, 0], expected[1, 1] = phi[1, 0], phi[1, 1]
        expected[0, 2], expected[1, 2] = ga[0], ga[1]
        expected[2, 2] = 1.
        expected[3, 0] = self.config.K_i * self.config.k_s
        expected[3, 3] = 1.
        expected[4, 0] = self.config.k_s

        np.testing.assert_allclose(model.A, expected, rtol=0., atol=0.)

    def test_control_vector_layout(self):
        model = self.model
        expected = np.zeros((5, 1))
        expected[0:2, 0] = model.Gamma_u.ravel()
        np.testing.assert_allclose(model.G, expected, rtol=0., atol=0.)

    def test_noise_vector_layout(self):
        model = self.model
        expected = np.array([[0.], [0.], [0.], [self.config.K_i], [1.]])
        np.testing.assert_allclose(model.L, expected, rtol=0., atol=0.)

    def test_observation_row(self):
        model = self.model
        expected = np.array([[self.config.alpha * self.config.k_s, 0., 0., 1., self.config.beta]])
        np.testing.assert_allclose(model.H, expected, rtol=0., atol=0.)

    def test_measurement_variance(self):
        self.assertAlmostEqual(self.model.R, self.config.alpha ** 2 * self.config.r)

    def test_process_covariance_layout(self):
        model = self.model
        expected = np.zeros((5, 5))
        expected[0:2, 0:2] = model.Q_m
        expected[2, 2] = self.config.q_a
        np.testing.assert_allclose(model.Q0, expected, rtol=0., atol=0.)

    def test_closed_loop_definitions(self):
        model = self.model
        np.testing.assert_allclose(model.F, model.A + model.G @ model.H, rtol=0., atol=0.)
        np.testing.assert_allclose(model.D, model.L + model.alpha * model.G, rtol=0., atol=0.)
        np.testing.assert_allclose(model.Q_cl, model.Q0 + model.r * model.D @ model.D.T,
                                   rtol=1e-12, atol=0.)
        np.testing.assert_allclose(model.N_cl, model.alpha * model.r * model.D,
                                   rtol=1e-12, atol=0.)

    def test_q_a_override_touches_only_one_entry(self):
        model = self.model
        other = model.with_q_a(self.config.q_a * 4.)
        difference = other.Q0 - model.Q0
        expected = np.zeros((5, 5))
        expected[2, 2] = self.config.q_a * 3.
        np.testing.assert_allclose(difference, expected, rtol=1e-12, atol=0.)
        np.testing.assert_allclose(other.F, model.F, rtol=0., atol=0.)


class TestObservability(unittest.TestCase):
    '''
    Section 17: the check must act on the closed loop pair and report findings.
    '''

    def test_report_shape(self):
        model = model_module.ClosedLoopModel(make_config())
        report = model.observability_report()
        self.assertEqual(report.singular_values.size, 5)
        self.assertGreaterEqual(report.rank, 0)
        self.assertLessEqual(report.rank, 5)
        payload = report.as_dict()
        for key in ('rank', 'condition', 'detectable', 'accel_observable', 'notes'):
            self.assertIn(key, payload)

    def test_zero_readout_gain_is_reported(self):
        model = model_module.ClosedLoopModel(make_config(k_cv=0.))
        report = model.observability_report()
        self.assertTrue(any('k_s = 0' in note for note in report.notes))
        self.assertLess(report.rank, 5)

    def test_zero_derivative_gain_is_reported(self):
        model = model_module.ClosedLoopModel(make_config(K_d=0.))
        report = model.observability_report()
        self.assertTrue(any('K_d = 0' in note for note in report.notes))

    def test_zero_integral_gain_is_reported(self):
        model = model_module.ClosedLoopModel(make_config(K_i=0.))
        report = model.observability_report()
        self.assertTrue(any('K_i = 0' in note for note in report.notes))

    def test_observability_matrix_has_five_blocks(self):
        model = model_module.ClosedLoopModel(make_config())
        matrix = model.observability_matrix()
        self.assertEqual(matrix.shape, (5, 5))
        np.testing.assert_allclose(matrix[0], model.H.ravel(), rtol=0., atol=0.)

    def test_scaling_is_a_similarity_transform(self):
        '''
        Scaling the state must not change the rank of the observability matrix,
        and ``observability_matrix`` must accept the matching observation row.
        '''
        model = model_module.ClosedLoopModel(make_config())
        scales = model._default_scales()
        T = np.diag(scales)
        O = model.observability_matrix()
        O_scaled = model.observability_matrix(np.diag(1. / scales) @ model.F @ T,
                                              model.H @ T)
        self.assertEqual(np.linalg.matrix_rank(O), np.linalg.matrix_rank(O_scaled))
        self.assertEqual(O.shape, O_scaled.shape)


class TestDeviceParameters(unittest.TestCase):
    '''
    Regression on the real parameter set of parameters/xiong-2018.json.

    The three noise entries are placeholders here: this test only exercises the
    constant matrices and the observability analysis, which do not depend on
    the noise level.
    '''

    def _device_model(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs', 'xiong-2018.json')
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            raw = json.load(handle)
        raw['sigma_n2'] = 1e-10
        raw['mechanical_noise_psd'] = 1e-18
        raw['psd_convention'] = 'two_sided_psd'
        config = config_module.build_config(raw)
        return config, model_module.ClosedLoopModel(config)

    def test_derived_device_gains(self):
        config, model = self._device_model()
        self.assertAlmostEqual(config.k_s, 4.7825831126802e7, delta=1.)
        self.assertAlmostEqual(config.K_u, -1.31521035598706e-4, delta=1e-15)
        self.assertAlmostEqual(config.alpha, 37.)
        self.assertAlmostEqual(config.beta, -28.)

    def test_closed_loop_is_observable_and_detectable(self):
        _, model = self._device_model()
        report = model.observability_report()
        self.assertEqual(report.rank, 5)
        self.assertTrue(report.detectable)
        self.assertTrue(report.accel_observable)
        self.assertLess(report.condition, 1e6)

    def test_closed_loop_is_not_unstable(self):
        _, model = self._device_model()
        radius = float(np.max(np.abs(np.linalg.eigvals(model.F))))
        self.assertLessEqual(radius, 1. + 1e-9)


if __name__ == '__main__':
    unittest.main()
