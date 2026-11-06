'''
Data adapter tests: layout, alignment, units and the reported inconsistencies.
'''
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.kalman_accelerometer import config as config_module  # noqa: E402
from examples.kalman_accelerometer import data_adapter  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'configs', 'algebraic-check.json')
FS = 128000.


def raw_config() -> dict:
    '''
    Load the synthetic configuration used by the algebraic checks.
    '''
    with open(CONFIG_PATH, 'r', encoding='utf-8', errors='replace') as handle:
        return json.load(handle)


class DataDirectory(unittest.TestCase):
    '''
    Create a small ``npy_dir`` layout that mimics the simulation output.
    '''

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='kalman-data-')
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.samples = 200
        self.time = np.arange(self.samples) / FS
        self.u = np.linspace(-1e-3, 1e-3, self.samples)

        os.makedirs(os.path.join(self.directory, 'pid'))
        os.makedirs(os.path.join(self.directory, 'mass_block'))
        np.save(os.path.join(self.directory, 'pid', 'time.npy'), self.time)
        np.save(os.path.join(self.directory, 'pid', 'pid.npy'), self.u)
        np.save(os.path.join(self.directory, 'mass_block', 'position.npy'),
                np.zeros(self.samples))
        np.save(os.path.join(self.directory, 'mass_block', 'velocity.npy'),
                np.zeros(self.samples))

    def build(self, **overrides):
        raw = raw_config()
        raw['data_path'] = self.directory
        raw.update(overrides)
        return config_module.build_config(raw)


class TestLoading(DataDirectory):
    '''
    Happy path and unit conversion.
    '''

    def test_loads_columns(self):
        data = data_adapter.load_data(self.build())
        self.assertEqual(len(data), self.samples)
        self.assertAlmostEqual(data.dt, 1. / FS)
        np.testing.assert_allclose(data.u, self.u)
        self.assertIsNotNone(data.x_true)
        self.assertIsNone(data.a_true)

    def test_unit_conversion(self):
        config = self.build(units={'t': 1., 'u': 1000.})
        data = data_adapter.load_data(config)
        np.testing.assert_allclose(data.u, self.u * 1000.)

    def test_summary(self):
        data = data_adapter.load_data(self.build())
        summary = data.summary()
        self.assertEqual(summary['samples'], self.samples)
        self.assertAlmostEqual(summary['fs'], FS)


class TestAlignment(DataDirectory):
    '''
    Section 5: offset, truncation and time shift belong to the adapter.
    '''

    def test_offset_and_length(self):
        config = self.build(sample_alignment={'offset': 10, 'length': 50})
        data = data_adapter.load_data(config)
        self.assertEqual(len(data), 50)
        self.assertAlmostEqual(data.t[0], self.time[10])
        np.testing.assert_allclose(data.u, self.u[10:60])

    def test_time_offset(self):
        config = self.build(sample_alignment={'offset': 0, 'length': None,
                                             'time_offset': 0.5})
        data = data_adapter.load_data(config)
        self.assertAlmostEqual(data.t[0], self.time[0] + 0.5)

    def test_offset_beyond_the_data(self):
        config = self.build(sample_alignment={'offset': 10000})
        with self.assertRaises(data_adapter.DataError):
            data_adapter.load_data(config)


class TestRejectedData(DataDirectory):
    '''
    Inconsistencies must be reported, never repaired silently.
    '''

    def test_missing_column(self):
        os.remove(os.path.join(self.directory, 'pid', 'pid.npy'))
        with self.assertRaises(data_adapter.DataError):
            data_adapter.load_data(self.build())

    def test_sampling_period_mismatch(self):
        wrong_time = np.arange(self.samples) / (FS / 2)
        np.save(os.path.join(self.directory, 'pid', 'time.npy'), wrong_time)
        with self.assertRaises(data_adapter.DataError) as context:
            data_adapter.load_data(self.build())
        self.assertIn('disagrees', str(context.exception))

    def test_non_monotonic_time(self):
        broken = self.time.copy()
        broken[50] = broken[40]
        np.save(os.path.join(self.directory, 'pid', 'time.npy'), broken)
        with self.assertRaises(data_adapter.DataError):
            data_adapter.load_data(self.build())

    def test_missing_optional_column_is_skipped(self):
        '''x_true and v_true are optional: a run may only provide a_true.'''
        os.remove(os.path.join(self.directory, 'mass_block', 'position.npy'))
        os.remove(os.path.join(self.directory, 'mass_block', 'velocity.npy'))
        data = data_adapter.load_data(self.build())
        self.assertIsNone(data.x_true)
        self.assertIsNone(data.v_true)
        self.assertTrue(any('skipped' in message for message in data.warnings))

    def test_non_finite_value(self):
        broken = self.u.copy()
        broken[7] = np.nan
        np.save(os.path.join(self.directory, 'pid', 'pid.npy'), broken)
        with self.assertRaises(data_adapter.DataError):
            data_adapter.load_data(self.build())

    def test_length_mismatch(self):
        np.save(os.path.join(self.directory, 'pid', 'time.npy'), self.time[:-1])
        with self.assertRaises(data_adapter.DataError):
            data_adapter.load_data(self.build())


class TestWindowing(DataDirectory):
    '''
    The evaluation windows of section 6.1 select samples by time.
    '''

    def test_time_window(self):
        data = data_adapter.load_data(self.build())
        window = data.window(0.0005, 0.001)
        self.assertGreater(len(window), 0)
        self.assertGreaterEqual(window.t[0], 0.0005)
        self.assertLessEqual(window.t[-1], 0.001)

    def test_empty_window(self):
        data = data_adapter.load_data(self.build())
        with self.assertRaises(data_adapter.DataError):
            data.window(10., 20.)


if __name__ == '__main__':
    unittest.main()
