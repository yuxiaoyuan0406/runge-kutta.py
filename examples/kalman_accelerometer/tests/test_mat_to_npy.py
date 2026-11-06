'''
Tests of the MAT to NPY converter.

Only the parts that work without MATLAB are exercised: plain MAT files with
numeric arrays and with MATLAB's ``Data``/``Time`` signal convention.
'''
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import scipy.io

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.kalman_accelerometer import mat_to_npy  # noqa: E402


class PlainMat(unittest.TestCase):
    '''
    Convert a plain MAT file that mimics a logged MATLAB signal.
    '''

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='mat2npy-test-')
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.samples = 32
        self.time = np.arange(self.samples) * 1e-3
        self.data = np.linspace(-1., 1., self.samples)
        self.path = os.path.join(self.directory, 'sample.mat')
        scipy.io.savemat(self.path, {
            'mems_out': {'Data': self.data.reshape(-1, 1),
                         'Time': self.time.reshape(-1, 1)},
            'a_true': {'Data': np.full((self.samples, 1), 9.81),
                       'Time': self.time.reshape(-1, 1)},
            'gain': 2.5,
        })

    def test_repo_layout(self):
        outdir = os.path.join(self.directory, 'out')
        manifest = mat_to_npy.convert(self.path, outdir, layout='repo')

        self.assertEqual(manifest['signals']['mems_out']['samples'], self.samples)
        self.assertAlmostEqual(manifest['signals']['mems_out']['dt'], 1e-3)

        u = np.load(os.path.join(outdir, 'pid', 'pid.npy'))
        t = np.load(os.path.join(outdir, 'pid', 'time.npy'))
        np.testing.assert_allclose(u, self.data, rtol=1e-12)
        np.testing.assert_allclose(t, self.time, rtol=1e-12)

        acceleration = np.load(os.path.join(outdir, 'mass_block', 'acceleration.npy'))
        np.testing.assert_allclose(acceleration, 9.81, rtol=1e-12)

    def test_flat_layout_and_manifest(self):
        outdir = os.path.join(self.directory, 'flat')
        manifest = mat_to_npy.convert(self.path, outdir, layout='flat')

        self.assertEqual(sorted(manifest['signals']), ['a_true', 'gain', 'mems_out'])
        self.assertEqual(manifest['repo_layout'], {})
        with open(os.path.join(outdir, 'manifest.json'), 'r', encoding='utf-8') as handle:
            stored = json.load(handle)
        self.assertEqual(sorted(stored['signals']), ['a_true', 'gain', 'mems_out'])
        self.assertTrue(os.path.isfile(os.path.join(outdir, 'raw', 'gain.npy')))

    def test_arrays_are_one_dimensional(self):
        outdir = os.path.join(self.directory, 'oned')
        mat_to_npy.convert(self.path, outdir, layout='repo')
        u = np.load(os.path.join(outdir, 'pid', 'pid.npy'))
        t = np.load(os.path.join(outdir, 'pid', 'time.npy'))
        self.assertEqual(u.shape, (self.samples,))
        self.assertEqual(t.shape, (self.samples,))

    def test_exclude(self):
        outdir = os.path.join(self.directory, 'excluded')
        manifest = mat_to_npy.convert(self.path, outdir, layout='flat',
                                      exclude=['gain'])
        self.assertNotIn('gain', manifest['signals'])

    def test_missing_input_signal(self):
        outdir = os.path.join(self.directory, 'empty')
        path = os.path.join(self.directory, 'empty.mat')
        scipy.io.savemat(path, {'nothing': np.zeros((0, 1))})
        with self.assertRaises(mat_to_npy.ConversionError):
            mat_to_npy.convert(path, outdir)


class MatlabDiscovery(unittest.TestCase):
    '''
    The lookup must not crash on a machine without MATLAB.
    '''

    def test_nonexistent_executable_is_ignored(self):
        result = mat_to_npy.find_matlab('/nonexistent/matlab')
        self.assertTrue(result is None or os.path.isfile(result))

    def test_environment_override(self):
        os.environ['MATLAB_EXE'] = '/nonexistent/matlab'
        try:
            result = mat_to_npy.find_matlab()
            self.assertTrue(result is None or os.path.isfile(result))
        finally:
            os.environ.pop('MATLAB_EXE', None)


if __name__ == '__main__':
    unittest.main()
