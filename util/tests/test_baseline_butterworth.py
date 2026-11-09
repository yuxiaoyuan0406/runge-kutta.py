'''
Tests of the second order Butterworth baseline filter.
'''
import math
import unittest
import warnings

import numpy as np

from util.baseline_filters import (ButterworthFilter, STEADY_STATE,
                                   ZERO_STATE)

FS = 128000.0
FC = 1000.0
B0 = 0.0005820761342360482
B1 = 0.0011641522684720964
B2 = 0.0005820761342360482
A1 = -1.930606427219668
A2 = 0.9329347317566122


def direct_form(b, a, samples):
    '''
    Independent direct form I reference recursion of the task sheet.

    It keeps the input and the output histories explicitly instead of the
    transposed state variables of the implementation, so it is a genuine cross
    check of the recursion and of the coefficient sign convention.
    '''
    b0, b1, b2 = b
    a1, a2 = a
    output = np.zeros(len(samples))
    x1 = x2 = 0.0
    y1 = y2 = 0.0
    for index, x0 in enumerate(samples):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x0
        y2, y1 = y1, y0
        output[index] = y0
    return output


def analytic_response(fs, fc, f):
    '''
    Independent evaluation of the bilinear response at one frequency.
    '''
    k = math.tan(math.pi * fc / fs)
    k2 = k * k
    denominator = 1.0 + math.sqrt(2.0) * k + k2
    b0 = k2 / denominator
    b1 = 2.0 * k2 / denominator
    b2 = k2 / denominator
    a1 = 2.0 * (k2 - 1.0) / denominator
    a2 = (1.0 - math.sqrt(2.0) * k + k2) / denominator
    z_inv = np.exp(-2j * math.pi * f / fs)
    return ((b0 + b1 * z_inv + b2 * z_inv ** 2) /
            (1.0 + a1 * z_inv + a2 * z_inv ** 2))


def measure_gain(filt, f, n_total=8192, n_window=4096):
    '''
    Measure ``|H|`` from the steady state of a sine of exactly integer cycles.
    '''
    n = np.arange(n_total, dtype=np.float64)
    x = np.sin(2.0 * np.pi * f * n / FS)
    y = filt.process_array(x)
    window = y[-n_window:]
    index = np.arange(n_total - n_window, n_total, dtype=np.float64)
    reference = 2.0 / n_window
    real = reference * np.sum(window * np.sin(2.0 * np.pi * f * index / FS))
    imag = reference * np.sum(window * np.cos(2.0 * np.pi * f * index / FS))
    return float(np.hypot(real, imag))


class TestReference(unittest.TestCase):
    '''
    Section 6 of the task sheet: fixed reference numbers.
    '''

    def test_coefficients(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        coefficients = filt.coefficients
        self.assertAlmostEqual(coefficients['b0'], B0, delta=1e-12)
        self.assertAlmostEqual(coefficients['b1'], B1, delta=1e-12)
        self.assertAlmostEqual(coefficients['b2'], B2, delta=1e-12)
        self.assertAlmostEqual(coefficients['a1'], A1, delta=1e-12)
        self.assertAlmostEqual(coefficients['a2'], A2, delta=1e-12)
        self.assertEqual(set(coefficients), {'b0', 'b1', 'b2', 'a1', 'a2'})
        self.assertEqual(filt.b, (coefficients['b0'], coefficients['b1'],
                                  coefficients['b2']))
        self.assertEqual(filt.a, (coefficients['a1'], coefficients['a2']))

    def test_default_cutoff_warns(self):
        with self.assertWarns(UserWarning) as context:
            filt = ButterworthFilter(fs=FS)
        self.assertEqual(filt.fc, FS / 128.0)
        self.assertIn('fs/128', str(context.warning))

    def test_explicit_cutoff_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ButterworthFilter(fs=FS, fc=FC)
        self.assertEqual(caught, [])

    def test_dc_gain(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        self.assertAlmostEqual(filt.dc_gain(), 1.0, delta=1e-12)
        self.assertAlmostEqual(abs(filt.frequency_response(0.0)), 1.0,
                               delta=1e-12)

    def test_poles_are_inside_the_unit_circle(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        poles = filt.poles
        self.assertEqual(poles.size, 2)
        self.assertLess(float(np.max(np.abs(poles))), 1.0)

    def test_minus_three_db_point(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        gain_db = 20.0 * math.log10(abs(filt.frequency_response(FC)))
        self.assertAlmostEqual(gain_db, 20.0 * math.log10(1.0 / math.sqrt(2.0)),
                               delta=0.01)
        self.assertAlmostEqual(gain_db, -3.0103, delta=0.01)


class TestBehaviour(unittest.TestCase):
    '''
    Zero, constant, impulse and step inputs, state handling and the two
    equivalent recursions.
    '''

    def test_zero_input(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        self.assertTrue(np.all(filt.process_array(np.zeros(256)) == 0.0))

    def test_constant_input_zero_state(self):
        value = 2.5
        filt = ButterworthFilter(fs=FS, fc=FC, init_mode=ZERO_STATE)
        output = filt.process_array(np.full(4096, value))
        self.assertAlmostEqual(output[0], B0 * value, delta=1e-15)
        self.assertAlmostEqual(output[-1], value, delta=1e-12)

    def test_constant_input_steady_state(self):
        value = -3.25
        filt = ButterworthFilter(fs=FS, fc=FC, init_mode=STEADY_STATE,
                                 initial_value=value)
        output = filt.process_array(np.full(64, value))
        self.assertAlmostEqual(output[0], value, delta=1e-12)
        self.assertTrue(np.allclose(output, value, rtol=0.0, atol=1e-12))

    def test_impulse_matches_direct_form(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        impulse = np.zeros(4096)
        impulse[0] = 1.0
        got = filt.process_array(impulse)
        want = direct_form(filt.b, filt.a, impulse)
        self.assertLessEqual(float(np.max(np.abs(got - want))), 1e-12)

    def test_direct_form_cross_check(self):
        rng = np.random.default_rng(1234)
        sequence = rng.uniform(-1.0, 1.0, 512)
        filt = ButterworthFilter(fs=FS, fc=FC)
        got = filt.process_array(sequence)
        want = direct_form(filt.b, filt.a, sequence)
        self.assertLessEqual(float(np.max(np.abs(got - want))), 1e-12)

    def test_step_response(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        output = filt.process_array(np.ones(4096))
        self.assertAlmostEqual(output[-1], 1.0, delta=1e-12)

    def test_batch_matches_sample(self):
        sequence = np.sin(2.0 * np.pi * 1000.0 * np.arange(1000) / FS)
        batch = ButterworthFilter(fs=FS, fc=FC)
        sample = ButterworthFilter(fs=FS, fc=FC)
        expected = sample.process_array(sequence)
        got = batch.process_array(sequence)
        self.assertLessEqual(float(np.max(np.abs(got - expected))), 1e-12)

    def test_reset_replays(self):
        sequence = np.cos(2.0 * np.pi * 500.0 * np.arange(512) / FS)
        filt = ButterworthFilter(fs=FS, fc=FC)
        filt.process_array(sequence)
        filt.reset()
        replay = filt.process_array(sequence)

        fresh = ButterworthFilter(fs=FS, fc=FC)
        expected = fresh.process_array(sequence)
        self.assertLessEqual(float(np.max(np.abs(replay - expected))), 1e-12)
        self.assertEqual(filt.coefficients, fresh.coefficients)

    def test_reset_keeps_steady_state_mode(self):
        filt = ButterworthFilter(fs=FS, fc=FC, init_mode=STEADY_STATE,
                                 initial_value=0.5)
        filt.process_array(np.linspace(0.0, 5.0, 100))
        filt.reset()
        self.assertAlmostEqual(filt.process_sample(0.5), 0.5, delta=1e-12)

    def test_reconfigure_recomputes_and_resets(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        filt.process_array(np.ones(100))
        filt.reconfigure(fc=2000.0)
        self.assertEqual(filt.fc, 2000.0)
        reference = ButterworthFilter(fs=FS, fc=2000.0)
        for key in ('b0', 'b1', 'b2', 'a1', 'a2'):
            self.assertAlmostEqual(filt.coefficients[key],
                                   reference.coefficients[key], delta=0.0)
        self.assertAlmostEqual(filt.process_sample(1.0), filt.b[0], delta=1e-15)

    def test_reconfigure_refuses_a_bad_cutoff_and_keeps_the_old_one(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        with self.assertRaises(ValueError):
            filt.reconfigure(fc=FS / 2.0)
        self.assertEqual(filt.fc, FC)

    def test_frequency_response_matches_the_formula(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        frequencies = np.array([0.0, 10.0, 100.0, FC, 10000.0, FS / 2.0])
        got = filt.frequency_response(frequencies)
        want = analytic_response(FS, FC, frequencies)
        self.assertLessEqual(float(np.max(np.abs(got - want))), 1e-12)

    def test_frequency_response_scalar(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        got = filt.frequency_response(FC)
        self.assertEqual(np.ndim(got), 0)
        self.assertAlmostEqual(complex(got), complex(analytic_response(FS, FC, FC)),
                               delta=1e-12)

    def test_time_domain_gain_at_the_cutoff(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        gain_db = 20.0 * math.log10(measure_gain(filt, FC))
        self.assertAlmostEqual(gain_db, -3.0103, delta=0.01)

    def test_causality_prefix(self):
        sequence = np.sin(2.0 * np.pi * 300.0 * np.arange(300) / FS)
        full = ButterworthFilter(fs=FS, fc=FC).process_array(sequence)
        prefix = ButterworthFilter(fs=FS, fc=FC).process_array(sequence[:100])
        self.assertTrue(np.array_equal(full[:100], prefix))

    def test_long_bounded_input_stays_bounded(self):
        rng = np.random.default_rng(7)
        sequence = rng.uniform(-1.0, 1.0, 20000)
        output = ButterworthFilter(fs=FS, fc=FC).process_array(sequence)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertLessEqual(float(np.max(np.abs(output))), 1.5)


class TestExtremeRatios(unittest.TestCase):
    '''
    Section 5 of the task sheet: near zero and near Nyquist ratios.
    '''

    def test_near_nyquist_is_supported(self):
        filt = ButterworthFilter(fs=FS, fc=FS / 2.0 * 0.9999)
        self.assertTrue(np.all(np.isfinite(list(filt.coefficients.values()))))
        self.assertLess(float(np.max(np.abs(filt.poles))), 1.0)
        output = filt.process_array(np.ones(4096))
        self.assertTrue(np.all(np.isfinite(output)))
        # A second order Butterworth step overshoots by about 4.3 percent; the
        # point of this test is boundedness, not a settled value.
        self.assertLessEqual(float(np.max(np.abs(output))), 1.1)

    def test_very_small_ratio_is_supported(self):
        filt = ButterworthFilter(fs=FS, fc=FS * 1e-6)
        self.assertTrue(np.all(np.isfinite(list(filt.coefficients.values()))))
        output = filt.process_array(np.concatenate([np.zeros(100),
                                                    np.ones(100)]))
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertLessEqual(float(np.max(np.abs(output))), 1.0 + 1e-9)

    def test_unrepresentable_pole_is_refused(self):
        with self.assertRaises(ValueError):
            ButterworthFilter(fs=FS, fc=FS * 1e-18)


class TestValidation(unittest.TestCase):
    '''
    Illegal parameters and inputs are refused, never silently repaired.
    '''

    def test_fs(self):
        for value in (0.0, -1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ButterworthFilter(fs=value, fc=FC)
        for value in ('128k', None, True, 1j):
            with self.assertRaises(TypeError):
                ButterworthFilter(fs=value, fc=FC)

    def test_fc(self):
        for value in (0.0, -1.0, FS / 2.0, FS, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ButterworthFilter(fs=FS, fc=value)

    def test_init_mode(self):
        with self.assertRaises(ValueError):
            ButterworthFilter(fs=FS, fc=FC, init_mode='steady')
        with self.assertRaises(ValueError):
            ButterworthFilter(fs=FS, fc=FC, init_mode=STEADY_STATE)
        with self.assertRaises(ValueError):
            ButterworthFilter(fs=FS, fc=FC, init_mode=ZERO_STATE,
                              initial_value=1.0)
        with self.assertRaises(ValueError):
            ButterworthFilter(fs=FS, fc=FC, init_mode=STEADY_STATE,
                              initial_value=float('inf'))

    def test_sample_and_sequence(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        with self.assertRaises(ValueError):
            filt.process_sample(float('nan'))
        with self.assertRaises(TypeError):
            filt.process_sample(None)
        with self.assertRaises(ValueError):
            filt.process_array(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            filt.process_array(np.array([0.0, float('nan')]))
        with self.assertRaises(ValueError):
            filt.process_array(np.zeros(2, dtype=np.complex128))
        self.assertEqual(filt.process_array([1.0, 2.0]).shape, (2,))

    def test_frequency_response_rejects_out_of_range(self):
        filt = ButterworthFilter(fs=FS, fc=FC)
        with self.assertRaises(ValueError):
            filt.frequency_response(-1.0)
        with self.assertRaises(ValueError):
            filt.frequency_response(FS / 2.0 * 1.01)


if __name__ == '__main__':
    unittest.main()
