'''
FOH configuration tests: the phase 2 keys of section 6 of the V2.1 task sheet.
'''
import copy
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

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'configs')
FOH_PATH = os.path.join(CONFIG_DIR, 'algebraic-check-foh.json')
ZOH_PATH = os.path.join(CONFIG_DIR, 'algebraic-check.json')


def foh_raw() -> dict:
    '''
    Load the synthetic FOH configuration as a mutable dictionary.
    '''
    with open(FOH_PATH, 'r', encoding='utf-8', errors='replace') as handle:
        return json.load(handle)


class TestFohPrior(unittest.TestCase):
    '''
    Section 15.1: the six state prior is assembled, never invented.
    '''

    def test_foh_keys_without_prior_model_fall_back_to_zoh(self):
        raw = foh_raw()
        raw.pop('prior_model')
        # Without ``prior_model`` the file describes the ZOH path, which then
        # misses q_a: the fallback is explicit and never silently FOH.
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('q_a', str(context.exception))

    def test_embedded_prior_uses_the_selection_matrix(self):
        raw = foh_raw()
        config = config_module.build_config(raw)
        selection = np.zeros((6, 5))
        for source, target in enumerate((0, 1, 2, 4, 5)):
            selection[target, source] = 1.
        base = np.asarray(raw['P0'], dtype=np.float64)
        expected = selection @ base @ selection.T
        expected[3, 3] = raw['jerk_var0']
        np.testing.assert_allclose(config.prior_covariance, expected,
                                   rtol=0., atol=1e-30)
        self.assertEqual(config.dim, 6)
        self.assertEqual(config.prior_mean.shape, (6,))

    def test_mean_inserts_the_jerk_prior(self):
        raw = foh_raw()
        raw['X0_hat'] = [1., 2., 3., 4., 5.]
        raw['jerk0_hat'] = 6.
        config = config_module.build_config(raw)
        np.testing.assert_allclose(config.prior_mean, [1., 2., 3., 6., 4., 5.])

    def test_six_entry_mean_needs_no_separate_jerk_prior(self):
        raw = foh_raw()
        raw['X0_hat'] = [1., 2., 3., 6., 4., 5.]
        raw.pop('jerk0_hat')
        config = config_module.build_config(raw)
        np.testing.assert_allclose(config.prior_mean, [1., 2., 3., 6., 4., 5.])

    def test_k_minus_1_shifts_the_integral_state(self):
        raw = foh_raw()
        raw['init_mode'] = 'k_minus_1'
        raw['X0_hat'] = [0., 0., 0., 0., 2., 3.]
        raw['P0'] = [[0.] * 5 for _ in range(5)]
        raw['P0'][4][4] = 1.
        config = config_module.build_config(raw)
        expected_i0 = 2. + raw['K_i'] * 3.
        self.assertAlmostEqual(config.prior_mean[4], expected_i0)
        self.assertAlmostEqual(config.prior_covariance[4, 4], raw['K_i'] ** 2)
        self.assertAlmostEqual(config.prior_covariance[4, 5], raw['K_i'])
        self.assertAlmostEqual(config.prior_covariance[5, 5], 1.)


class TestFohRejected(unittest.TestCase):
    '''
    Incomplete or contradictory FOH configurations must raise.
    '''

    def test_missing_q_j(self):
        raw = foh_raw()
        raw.pop('q_j')
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('q_j', str(context.exception))

    def test_missing_jerk_init_mode(self):
        raw = foh_raw()
        raw.pop('jerk_init_mode')
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('jerk_init_mode', str(context.exception))

    def test_unknown_prior_model(self):
        raw = foh_raw()
        raw['prior_model'] = 'FOH2'
        with self.assertRaises(config_module.ConfigError):
            config_module.build_config(raw)

    def test_known_jerk_with_nonzero_variance(self):
        raw = foh_raw()
        raw['jerk_init_mode'] = 'known'
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('known', str(context.exception))

    def test_unknown_jerk_with_zero_variance(self):
        raw = foh_raw()
        raw['jerk_var0'] = 0.
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('unknown', str(context.exception))

    def test_negative_q_j(self):
        raw = foh_raw()
        raw['q_j'] = -1.
        with self.assertRaises(config_module.ConfigError):
            config_module.build_config(raw)

    def test_wrong_prior_shape(self):
        raw = foh_raw()
        raw['P0'] = [[1., 0.], [0., 1.]]
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('shape', str(context.exception))

    def test_missing_covariance(self):
        raw = foh_raw()
        raw.pop('P0')
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('P0_FOH', str(context.exception))


class TestFohReportedQuantities(unittest.TestCase):
    '''
    Section 6: q_a belongs to ZOH, q_j to FOH, and the report is prior aware.
    '''

    def test_q_a_is_not_required_on_the_foh_path(self):
        raw = foh_raw()
        self.assertNotIn('q_a', raw)
        config = config_module.build_config(raw)
        self.assertEqual(config.q_j, raw['q_j'])
        self.assertEqual(config.q_a, 0.)

    def test_summary_reports_the_foh_keys(self):
        config = config_module.build_config(foh_raw())
        summary = config.summary()
        for key in ('prior_model', 'q_j', 'jerk_init_mode', 'state_labels',
                    'foh_output_fields', 'zoh_reference'):
            self.assertIn(key, summary)
        self.assertEqual(summary['prior_model'], 'FOH')
        self.assertNotIn('q_a', summary)

    def test_missing_report_for_foh(self):
        report = config_module.missing_parameter_report(FOH_PATH)
        self.assertTrue(report['ok'], report)
        self.assertEqual(report['prior_model'], 'FOH')

    def test_zoh_path_is_untouched(self):
        with open(ZOH_PATH, 'r', encoding='utf-8') as handle:
            raw = json.load(handle)
        config = config_module.build_config(raw)
        self.assertEqual(config.prior_model, 'ZOH')
        self.assertEqual(config.dim, 5)
        self.assertIsNone(config.P0_FOH)

    def test_foh_ignores_q_a_numerically(self):
        raw = foh_raw()
        first = config_module.build_config(copy.deepcopy(raw))
        raw['q_a'] = 12345.
        second = config_module.build_config(raw)
        self.assertEqual(first.q_j, second.q_j)
        np.testing.assert_array_equal(first.prior_covariance, second.prior_covariance)


if __name__ == '__main__':
    unittest.main()
