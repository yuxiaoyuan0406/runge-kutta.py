'''
Configuration validation tests, section 9 and section 16 of the task sheet.
'''
import copy
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.kalman_accelerometer import config as config_module  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'configs', 'algebraic-check.json')
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'configs', 'parameter-template.json')


def raw_config() -> dict:
    '''
    Load the synthetic configuration used by the algebraic checks.
    '''
    with open(CONFIG_PATH, 'r', encoding='utf-8', errors='replace') as handle:
        return json.load(handle)


class TestParameterCompleteness(unittest.TestCase):
    '''
    The template must report every missing entry instead of inventing values.
    '''

    def test_template_reports_missing_parameters(self):
        report = config_module.missing_parameter_report(TEMPLATE_PATH)
        self.assertFalse(report['ok'])
        for key in ('m', 'c', 'k', 'K_u', 'K_p', 'sigma_n2', 'q_a', 'X0_hat', 'P0'):
            self.assertIn(key, report['missing'])

    def test_template_cannot_be_built(self):
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.load_config(TEMPLATE_PATH)
        message = str(context.exception)
        self.assertIn('missing required parameter', message)

    def test_all_problems_are_reported_at_once(self):
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.load_config(TEMPLATE_PATH)
        self.assertGreaterEqual(len(context.exception.problems), 10)


class TestDerivedQuantities(unittest.TestCase):
    '''
    Derived gains and noise intensities must follow sections 2 and 12.
    '''

    def test_derived_gains(self):
        config = config_module.build_config(raw_config())
        self.assertAlmostEqual(config.k_s, config.k_cv * config.k_adc)
        self.assertAlmostEqual(config.alpha, config.K_p + config.K_d)
        self.assertAlmostEqual(config.beta, -config.K_d)
        self.assertAlmostEqual(config.r, config.sigma_n2)
        self.assertAlmostEqual(config.R, config.alpha ** 2 * config.sigma_n2)

    def test_force_psd_to_acceleration_psd(self):
        config = config_module.build_config(raw_config())
        self.assertAlmostEqual(config.q_b, config.q_F / config.m ** 2)

    def test_one_sided_asd_is_squared_and_halved(self):
        raw = raw_config()
        raw['psd_convention'] = 'one_sided_asd'
        raw['mechanical_noise_psd'] = 3e-9
        config = config_module.build_config(raw)
        self.assertAlmostEqual(config.q_F, (3e-9 ** 2) / 2)
        self.assertAlmostEqual(config.q_b, (3e-9 ** 2) / 2 / raw['m'] ** 2)

    def test_one_sided_psd_is_halved(self):
        raw = raw_config()
        raw['psd_convention'] = 'one_sided_psd'
        raw['mechanical_noise_psd'] = 4e-18
        config = config_module.build_config(raw)
        self.assertAlmostEqual(config.q_F, 2e-18)

    def test_acceleration_psd_unit(self):
        raw = raw_config()
        raw['mechanical_noise_unit'] = 'accel'
        raw['mechanical_noise_psd'] = 1e-6
        config = config_module.build_config(raw)
        self.assertAlmostEqual(config.q_b, 1e-6)
        self.assertAlmostEqual(config.q_F, 1e-6 * raw['m'] ** 2)


class TestRejectedConfigurations(unittest.TestCase):
    '''
    Inconsistent configurations must raise instead of being patched.
    '''

    def test_fs_and_ts_disagreement(self):
        raw = raw_config()
        raw['fs'] = 64000.
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('disagree', str(context.exception))

    def test_negative_sigma_n2(self):
        raw = raw_config()
        raw['sigma_n2'] = -1.
        with self.assertRaises(config_module.ConfigError):
            config_module.build_config(raw)

    def test_non_positive_mass(self):
        raw = raw_config()
        raw['m'] = 0.
        with self.assertRaises(config_module.ConfigError):
            config_module.build_config(raw)

    def test_p0_not_positive_semidefinite(self):
        raw = raw_config()
        raw['P0'] = copy.deepcopy(raw['P0'])
        raw['P0'][0][0] = -1.
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('positive semi definite', str(context.exception))

    def test_p0_wrong_shape(self):
        raw = raw_config()
        raw['P0'] = [[1., 0.], [0., 1.]]
        with self.assertRaises(config_module.ConfigError) as context:
            config_module.build_config(raw)
        self.assertIn('shape', str(context.exception))

    def test_unknown_psd_convention(self):
        raw = raw_config()
        raw['psd_convention'] = 'magic'
        with self.assertRaises(config_module.ConfigError):
            config_module.build_config(raw)


class TestInitialIndex(unittest.TestCase):
    '''
    Section 16: a prior given at I[-1] must be shifted to I[0].
    '''

    def test_k_minus_1_transform(self):
        raw = raw_config()
        raw['init_mode'] = 'k_minus_1'
        raw['X0_hat'] = [0., 0., 0., 2., 3.]
        raw['P0'] = [[0.] * 5 for _ in range(5)]
        raw['P0'][4][4] = 1.

        config = config_module.build_config(raw)
        expected_i0 = 2. + raw['K_i'] * 3.
        self.assertAlmostEqual(config.X0_hat[3], expected_i0)
        # I[0] = I[-1] + K_i e[-1] shares e[-1] with h[0], so the cross
        # covariance must appear in the transformed prior.
        self.assertAlmostEqual(config.P0[3, 3], raw['K_i'] ** 2)
        self.assertAlmostEqual(config.P0[3, 4], raw['K_i'])
        self.assertAlmostEqual(config.P0[4, 4], 1.)

    def test_k0_keeps_the_prior(self):
        raw = raw_config()
        config = config_module.build_config(raw)
        self.assertTrue((config.X0_hat == 0.).all())


if __name__ == '__main__':
    unittest.main()
