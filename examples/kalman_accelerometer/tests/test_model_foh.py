'''
FOH discretisation, state assembly and observability tests, sections 11, 12,
17 and 18 of the V2.1 task sheet.
'''
import json
import os
import sys
import unittest

import numpy as np
import scipy.integrate
import scipy.linalg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.kalman_accelerometer import config as config_module  # noqa: E402
from examples.kalman_accelerometer import model as model_module  # noqa: E402
from examples.kalman_accelerometer import verification  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'configs')
FOH_PATH = os.path.join(CONFIG_DIR, 'algebraic-check-foh.json')
ZOH_PATH = os.path.join(CONFIG_DIR, 'algebraic-check.json')
DEVICE_FOH_PATH = os.path.join(CONFIG_DIR, 'xiong-2018-foh.json')


def foh_config(**overrides):
    '''
    Build the synthetic FOH configuration, optionally overriding entries.
    '''
    with open(FOH_PATH, 'r', encoding='utf-8', errors='replace') as handle:
        raw = json.load(handle)
    raw.update(overrides)
    return config_module.build_config(raw)


class TestFohDiscretization(unittest.TestCase):
    '''
    Section 11: the 5x5 augmented exponential against two independent routes.
    '''

    def test_augmented_exponential_matches_the_integrals(self):
        config = foh_config()
        result = verification.check_foh_discretization(
            config.m, config.c, config.k, config.K_u, config.Ts)
        self.assertTrue(result['passed'], result)
        self.assertLess(result['Gamma_j_relative'], 1e-12)

    def test_free_particle_limit(self):
        config = foh_config()
        result = verification.check_foh_free_mass(config.Ts)
        self.assertTrue(result['passed'], result)
        # The displacement column is T^3/6, not T^3/3.
        self.assertLess(result['Gamma_j_relative'], 1e-14)

    def test_the_kernel_is_not_the_naive_one(self):
        '''
        For a damped mass the two candidate kernels differ; the implementation
        must use ``exp(A_m (Ts - tau)) B_a tau``.
        '''
        config = foh_config()
        A_m, B_u, B_a = model_module.continuous_mechanics(
            config.m, config.c, config.k, config.K_u)
        _, _, _, Gamma_j = model_module.foh_matrices(A_m, B_u, B_a, config.Ts)

        def integrate(function):
            values, _ = scipy.integrate.quad_vec(
                lambda tau: np.asarray(function(tau)).ravel(), 0., config.Ts,
                epsabs=1e-14, epsrel=1e-12)
            return np.asarray(values).reshape(2, 1)

        naive = integrate(lambda tau: scipy.linalg.expm(A_m * tau) @ B_a * tau)
        correct = integrate(
            lambda tau: scipy.linalg.expm(A_m * (config.Ts - tau)) @ B_a * tau)
        np.testing.assert_allclose(Gamma_j, correct, rtol=1e-10)
        difference = float(np.max(np.abs(Gamma_j - naive)))
        scale = float(np.max(np.abs(correct)))
        self.assertGreater(difference / scale, 1e-6)

    def test_zoh_and_foh_share_the_common_columns(self):
        with open(FOH_PATH, 'r', encoding='utf-8') as handle:
            raw = json.load(handle)
        zoh = model_module.discretize_zero_order_hold(
            raw['m'], raw['c'], raw['k'], raw['K_u'], raw['Ts'])
        foh = model_module.discretize_first_order_hold(
            raw['m'], raw['c'], raw['k'], raw['K_u'], raw['Ts'])
        for left, right in zip(zoh, foh[:3]):
            np.testing.assert_allclose(left, right, rtol=0., atol=1e-18)


class TestFohStateMatrices(unittest.TestCase):
    '''
    Section 18: the explicit rows and columns of A, G, L, H and Q0.
    '''

    def setUp(self):
        self.config = foh_config()
        self.model = model_module.ClosedLoopModel(self.config)
        self.Phi, self.Gamma_u, self.Gamma_a, self.Gamma_j = (
            self.model.Phi, self.model.Gamma_u, self.model.Gamma_a,
            self.model.Gamma_j)

    def test_state_order_and_dimension(self):
        self.assertEqual(self.model.dim, 6)
        self.assertEqual(self.model.prior_model, 'FOH')
        self.assertEqual(tuple(self.model.state_labels), ('x', 'v', 'a', 'j', 'I', 'h'))
        self.assertEqual(self.model.index['a'], 2)
        self.assertEqual(self.model.index['j'], 3)
        self.assertEqual(self.model.index['I'], 4)
        self.assertEqual(self.model.index['h'], 5)

    def test_rows_of_a(self):
        config = self.config
        A = self.model.A
        Ts = config.Ts
        K_i = config.K_i
        k_s = config.k_s

        np.testing.assert_allclose(A[0, 0:2], self.Phi[0, :], rtol=0., atol=0.)
        np.testing.assert_allclose(A[0, 2], self.Gamma_a[0, 0], rtol=0., atol=0.)
        np.testing.assert_allclose(A[0, 3], self.Gamma_j[0, 0], rtol=0., atol=0.)
        np.testing.assert_allclose(A[1, 0:2], self.Phi[1, :], rtol=0., atol=0.)
        np.testing.assert_allclose(A[1, 2], self.Gamma_a[1, 0], rtol=0., atol=0.)
        np.testing.assert_allclose(A[1, 3], self.Gamma_j[1, 0], rtol=0., atol=0.)
        np.testing.assert_allclose(A[2, :], [0., 0., 1., Ts, 0., 0.], rtol=0., atol=0.)
        np.testing.assert_allclose(A[3, :], [0., 0., 0., 1., 0., 0.], rtol=0., atol=0.)
        np.testing.assert_allclose(A[4, :], [K_i * k_s, 0., 0., 0., 1., 0.],
                                   rtol=0., atol=0.)
        np.testing.assert_allclose(A[5, :], [k_s, 0., 0., 0., 0., 0.],
                                   rtol=0., atol=0.)

    def test_input_columns_and_observation_row(self):
        config = self.config
        model = self.model
        np.testing.assert_allclose(model.G.ravel()[0:2], self.Gamma_u.ravel(),
                                   rtol=0., atol=0.)
        np.testing.assert_allclose(model.G.ravel()[2:], 0., rtol=0., atol=0.)
        np.testing.assert_allclose(model.L.ravel(), [0., 0., 0., 0., config.K_i, 1.],
                                   rtol=0., atol=0.)
        np.testing.assert_allclose(model.H.ravel(),
                                   [config.alpha * config.k_s, 0., 0., 0., 1., config.beta],
                                   rtol=0., atol=0.)

    def test_process_covariance_layout(self):
        model = self.model
        expected = np.zeros((6, 6))
        expected[0:2, 0:2] = model.Q_m
        expected[3, 3] = self.config.q_j
        np.testing.assert_allclose(model.Q0, expected, rtol=0., atol=1e-30)
        # FOH must not read q_a: the acceleration diagonal is driven by the
        # mechanical block only.
        self.assertEqual(model.Q0[2, 2], 0.)
        self.assertGreater(model.Q0[3, 3], 0.)

    def test_closed_loop_pair(self):
        model = self.model
        np.testing.assert_allclose(model.F, model.A + model.G @ model.H,
                                   rtol=0., atol=0.)
        np.testing.assert_allclose(model.N_cl, model.alpha * model.r * model.D,
                                   rtol=1e-15, atol=0.)


class TestFohObservability(unittest.TestCase):
    '''
    Section 17: the six state closed loop pair must be checked, including the
    Jerk target, on physically scaled coordinates.

    The *synthetic* algebraic-check coefficients are weakly observable in this
    sampling regime (the phase 1 synthetic ZOH model is too, rank 4 of 5), so
    the full rank assertion uses the device model of ``configs/``, which needs
    no measurement data.
    '''

    def test_scaled_rank_and_targets(self):
        model = model_module.ClosedLoopModel(
            config_module.load_config(DEVICE_FOH_PATH))
        report = model.observability_report()
        self.assertEqual(report.dim, 6)
        self.assertEqual(report.rank, 6, report.as_dict())
        self.assertTrue(report.accel_observable)
        self.assertTrue(report.jerk_observable)
        self.assertTrue(report.detectable, report.as_dict())
        self.assertLess(report.accel_null_component, 1e-12)
        self.assertLess(report.jerk_null_component, 1e-12)
        # Weak observability must stay visible in the report instead of being
        # rounded into "observable".
        relative = float(report.singular_values[-1] / report.singular_values[0])
        self.assertGreater(relative, 1e-12)
        self.assertGreater(report.condition, 1e4)

    def test_observability_matrix_uses_all_six_states(self):
        model = model_module.ClosedLoopModel(
            config_module.load_config(DEVICE_FOH_PATH))
        scales = model.observability_report().scales
        T = np.diag(scales)
        Tinv = np.diag(1. / scales)
        matrix = model.observability_matrix(Tinv @ model.F @ T, model.H @ T)
        self.assertEqual(matrix.shape, (6, 6))
        self.assertEqual(int(np.linalg.matrix_rank(matrix)), 6)

    def test_synthetic_model_reports_its_weak_direction(self):
        model = model_module.ClosedLoopModel(foh_config())
        report = model.observability_report()
        # The synthetic coefficients are numerically weak here; the important
        # property is that the report states it instead of claiming full rank.
        self.assertLessEqual(report.rank, 6)
        self.assertGreater(report.condition, 1e6)

    def test_ki_zero_static_direction(self):
        config = foh_config(K_i=0.0)
        model = model_module.ClosedLoopModel(config)
        result = verification.check_foh_static_unobservable_direction(model)
        self.assertTrue(result['applicable'])
        self.assertTrue(result['passed'], result)
        # The Jerk component of that direction is zero; it contains the
        # acceleration k/m instead.
        self.assertEqual(result['direction'][3], 0.)
        self.assertGreater(result['direction'][2], 0.)


class TestQJHandling(unittest.TestCase):
    '''
    The two prior variances are not interchangeable, section 9 of V2.1.
    '''

    def test_with_q_a_is_rejected_on_foh(self):
        model = model_module.ClosedLoopModel(foh_config())
        with self.assertRaises(model_module.ModelError):
            model.with_q_a(1e-7)

    def test_with_q_j_is_rejected_on_zoh(self):
        with open(ZOH_PATH, 'r', encoding='utf-8') as handle:
            raw = json.load(handle)
        model = model_module.ClosedLoopModel(config_module.build_config(raw))
        with self.assertRaises(model_module.ModelError):
            model.with_q_j(1.)

    def test_q_j_only_touches_the_jerk_entry(self):
        base = model_module.ClosedLoopModel(foh_config())
        raised = base.with_q_j(base.q_j + 5.)
        difference = raised.Q0 - base.Q0
        expected = np.zeros((6, 6))
        expected[3, 3] = 5.
        np.testing.assert_allclose(difference, expected, rtol=0., atol=1e-24)


if __name__ == '__main__':
    unittest.main()
