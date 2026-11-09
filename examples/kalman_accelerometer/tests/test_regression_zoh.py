'''
Compatibility regression of the phase 1 ZOH path, section 5 of the V2.1 task
sheet.

The fixture ``tests/fixtures/zoh-reference.npz`` was captured with the phase 1
implementation, before any FOH code existed.  The upgraded program must
reproduce it inside the tolerance declared in the manifest.  The test never
re-captures the record: overwriting it would turn the regression into a
tautology.
'''
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.kalman_accelerometer import regression  # noqa: E402


class TestReferenceRecord(unittest.TestCase):
    '''
    The stored record must be complete and must describe the declared cases.
    '''

    def test_record_is_present(self):
        arrays, manifest = regression.load()
        self.assertEqual(len(manifest['cases']), len(regression.CASES))
        for case in regression.CASES:
            self.assertIn(case['name'], [item['name'] for item in manifest['cases']])
            for key in regression.ARRAYS:
                self.assertIn(f'{case["name"]}/{key}', arrays)
        self.assertEqual(manifest['tolerance']['array_relative'], 1e-12)

    def test_manifest_matches_the_file(self):
        arrays, manifest = regression.load()
        self.assertEqual(manifest['file_sha256'],
                         regression.sha256(os.path.join(
                             regression.REFERENCE_DIR, manifest['file'])))


class TestZohRegression(unittest.TestCase):
    '''
    Section 5: the ZOH estimate, covariance, innovation, gain and the report
    metrics must match the phase 1 record.
    '''

    def test_zoh_path_reproduces_the_reference(self):
        arrays, manifest = regression.load()
        report = regression.compare(arrays, manifest)
        self.assertTrue(report['passed'], report.get('problems'))
        self.assertLessEqual(report['max_array_relative'], 1e-12)
        self.assertEqual(len(report['cases']), len(regression.CASES))

    def test_metrics_are_unchanged(self):
        arrays, manifest = regression.load()
        report = regression.compare(arrays, manifest)
        for case in report['cases']:
            self.assertTrue(case['metrics_passed'], case)


if __name__ == '__main__':
    unittest.main()
