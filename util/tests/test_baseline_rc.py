'''
Tests of the first order RC baseline filter.
'''
import math
import unittest
import warnings

import numpy as np

from util.baseline_filters import RCFilter, STEADY_STATE, ZERO_STATE

FS = 128000.0
FC = 1000.0
ALPHA = 0.9520979267837046
BETA = 0.04790207321629536


def analytic_response(fc, fs, f):
    '''
    Independent evaluation of ``beta / (1 - alpha exp(-j w))``.
    '''
    alpha = math.exp(-2.0 * math.pi * fc / fs)
    beta = 1.0 - alpha
    omega = 2.0 * math.pi * f / fs
    return beta / (1.0 - alpha * np.exp(-1j * omega))


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
        filt = RCFilter(fs=FS, fc=FC)
        self.assertAlmostEqual(filt.alpha, ALPHA, delta=1e-12)
        self.assertAlmostEqual(filt.beta, BETA, delta=1e-12)
        self.assertAlmostEqual(filt.alpha + filt.beta, 1.0, delta=1e-15)
        self.assertEqual(set(filt.coefficients), {'alpha', 'beta'})

    def test_default_cutoff_warns(self):
        with self.assertWarns(UserWarning) as context:
            filt = RCFilter(fs=FS)
        self.assertEqual(filt.fc, FS / 128.0)
        self.assertIn('fs/128', str(context.warning))

    def test_explicit_cutoff_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            RCFilter(fs=FS, fc=FC)
        self.assertEqual(caught, [])


class TestBehaviour(unittest.TestCase):
    '''
    Zero, constant, impulse and step inputs, state handling and frequency
    response.
    '''

    def test_zero_input(self):
        filt = RCFilter(fs=FS, fc=FC)
        output = filt.process_array(np.zeros(256))
        self.assertTrue(np.all(output == 0.0))

    def test_constant_input_zero_state(self):
        value = 2.5
        filt = RCFilter(fs=FS, fc=FC, init_mode=ZERO_STATE)
        output = filt.process_array(np.full(4096, value))
        self.assertAlmostEqual(output[0], BETA * value, delta=1e-15)
        self.assertAlmostEqual(output[-1], value, delta=1e-12)
        self.assertTrue(np.all(np.diff(output) >= 0.0))

    def test_constant_input_steady_state(self):
        value = -3.25
        filt = RCFilter(fs=FS, fc=FC, init_mode=STEADY_STATE,
                        initial_value=value)
        output = filt.process_array(np.full(64, value))
        self.assertAlmostEqual(output[0], value, delta=1e-12)
        self.assertTrue(np.allclose(output, value, rtol=0.0, atol=1e-12))

    def test_impulse_response(self):
        filt = RCFilter(fs=FS, fc=FC)
        impulse = np.zeros(4096)
        impulse[0] = 1.0
        output = filt.process_array(impulse)
        expected = BETA * ALPHA ** np.arange(4096)
        self.assertLessEqual(float(np.max(np.abs(output - expected))), 1e-15)

    def test_step_response(self):
        filt = RCFilter(fs=FS, fc=FC)
        output = filt.process_array(np.ones(4096))
        self.assertAlmostEqual(output[-1], 1.0, delta=1e-12)

    def test_batch_matches_sample(self):
        sequence = np.sin(2.0 * np.pi * 1000.0 * np.arange(1000) / FS)
        batch = RCFilter(fs=FS, fc=FC)
        sample = RCFilter(fs=FS, fc=FC)
        expected = sample.process_array(sequence)
        got = batch.process_array(sequence)
        self.assertLessEqual(float(np.max(np.abs(got - expected))), 1e-12)

    def test_process_array_is_a_sample_loop(self):
        sequence = np.linspace(-1.0, 1.0, 257)
        filt = RCFilter(fs=FS, fc=FC)
        output = filt.process_array(sequence)
        reference = RCFilter(fs=FS, fc=FC)
        for index, value in enumerate(sequence):
            self.assertAlmostEqual(output[index],
                                   reference.process_sample(value),
                                   delta=1e-15)

    def test_reset_replays(self):
        sequence = np.cos(2.0 * np.pi * 500.0 * np.arange(512) / FS)
        filt = RCFilter(fs=FS, fc=FC)
        filt.process_array(sequence)
        filt.reset()
        replay = filt.process_array(sequence)

        fresh = RCFilter(fs=FS, fc=FC)
        expected = fresh.process_array(sequence)
        self.assertLessEqual(float(np.max(np.abs(replay - expected))), 1e-12)
        self.assertEqual(filt.coefficients, fresh.coefficients)

    def test_reset_keeps_steady_state_mode(self):
        filt = RCFilter(fs=FS, fc=FC, init_mode=STEADY_STATE, initial_value=0.5)
        filt.process_array(np.linspace(0.0, 5.0, 100))
        filt.reset()
        self.assertAlmostEqual(filt.process_sample(0.5), 0.5, delta=1e-12)

    def test_reconfigure_recomputes_and_resets(self):
        filt = RCFilter(fs=FS, fc=FC)
        filt.process_array(np.ones(100))
        filt.reconfigure(fc=2000.0)
        self.assertEqual(filt.fc, 2000.0)
        self.assertAlmostEqual(filt.alpha,
                               math.exp(-2.0 * math.pi * 2000.0 / FS),
                               delta=1e-15)
        self.assertAlmostEqual(filt.process_sample(1.0), filt.beta, delta=1e-15)

    def test_reconfigure_keeps_steady_state(self):
        filt = RCFilter(fs=FS, fc=FC, init_mode=STEADY_STATE, initial_value=0.5)
        filt.reconfigure(fs=64000.0)
        self.assertEqual(filt.fs, 64000.0)
        self.assertEqual(filt.fc, FC)
        self.assertAlmostEqual(filt.process_sample(0.5), 0.5, delta=1e-12)

    def test_reconfigure_refuses_a_bad_cutoff_and_keeps_the_old_one(self):
        filt = RCFilter(fs=FS, fc=FC)
        with self.assertRaises(ValueError):
            filt.reconfigure(fc=FS)
        self.assertEqual(filt.fc, FC)
        self.assertEqual(filt.fs, FS)

    def test_frequency_response_matches_the_formula(self):
        filt = RCFilter(fs=FS, fc=FC)
        frequencies = np.array([0.0, 10.0, 100.0, FC, 10000.0, FS / 2.0])
        got = filt.frequency_response(frequencies)
        want = analytic_response(FC, FS, frequencies)
        self.assertLessEqual(float(np.max(np.abs(got - want))), 1e-12)

    def test_frequency_response_scalar(self):
        filt = RCFilter(fs=FS, fc=FC)
        got = filt.frequency_response(FC)
        self.assertEqual(np.ndim(got), 0)
        self.assertAlmostEqual(complex(got), complex(analytic_response(FC, FS, FC)),
                               delta=1e-15)

    def test_frequency_response_dc_gain(self):
        filt = RCFilter(fs=FS, fc=FC)
        self.assertAlmostEqual(abs(filt.frequency_response(0.0)), 1.0,
                               delta=1e-12)
        self.assertAlmostEqual(filt.dc_gain(), 1.0, delta=1e-15)

    def test_time_domain_gain(self):
        filt = RCFilter(fs=FS, fc=FC)
        got = measure_gain(filt, FC)
        want = abs(analytic_response(FC, FS, FC))
        self.assertAlmostEqual(got / want, 1.0, delta=1e-9)

    def test_causality_prefix(self):
        sequence = np.sin(2.0 * np.pi * 300.0 * np.arange(300) / FS)
        full = RCFilter(fs=FS, fc=FC).process_array(sequence)
        prefix = RCFilter(fs=FS, fc=FC).process_array(sequence[:100])
        self.assertTrue(np.array_equal(full[:100], prefix))


class TestValidation(unittest.TestCase):
    '''
    Illegal parameters and inputs are refused, never silently repaired.
    '''

    def test_fs(self):
        for value in (0.0, -1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                RCFilter(fs=value, fc=FC)
        for value in ('128k', None, True, 1j):
            with self.assertRaises(TypeError):
                RCFilter(fs=value, fc=FC)

    def test_fc(self):
        for value in (0.0, -1.0, FS / 2.0, FS, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                RCFilter(fs=FS, fc=value)

    def test_init_mode(self):
        with self.assertRaises(ValueError):
            RCFilter(fs=FS, fc=FC, init_mode='steady')
        with self.assertRaises(ValueError):
            RCFilter(fs=FS, fc=FC, init_mode=STEADY_STATE)
        with self.assertRaises(ValueError):
            RCFilter(fs=FS, fc=FC, init_mode=ZERO_STATE, initial_value=1.0)
        with self.assertRaises(ValueError):
            RCFilter(fs=FS, fc=FC, init_mode=STEADY_STATE,
                     initial_value=float('nan'))
        with self.assertRaises(ValueError):
            RCFilter(fs=FS, fc=FC).reset(init_mode='steady')

    def test_sample(self):
        filt = RCFilter(fs=FS, fc=FC)
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                filt.process_sample(value)
        for value in ('1.0', None, 1j):
            with self.assertRaises(TypeError):
                filt.process_sample(value)

    def test_sequence(self):
        filt = RCFilter(fs=FS, fc=FC)
        with self.assertRaises(ValueError):
            filt.process_array(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            filt.process_array(np.zeros(1, dtype=np.complex128))
        with self.assertRaises(ValueError):
            filt.process_array(np.array([0.0, float('nan')]))
        with self.assertRaises(ValueError):
            filt.process_array(1.0)
        self.assertEqual(filt.process_array([]).size, 0)
        self.assertEqual(filt.process_array([1.0, 2.0]).shape, (2,))

    def test_frequency_response_rejects_out_of_range(self):
        filt = RCFilter(fs=FS, fc=FC)
        with self.assertRaises(ValueError):
            filt.frequency_response(-1.0)
        with self.assertRaises(ValueError):
            filt.frequency_response(FS / 2.0 * 1.01)
        with self.assertRaises(ValueError):
            filt.frequency_response(float('nan'))


if __name__ == '__main__':
    unittest.main()
