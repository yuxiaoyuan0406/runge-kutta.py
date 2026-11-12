'''
Tests for `util.asd`.

The MATLAB-parity tests use the reference data provided in `asd_test_data/`:

    asd_test_data_128kHz_10_300.mat   input series `chopout4` (1x327681, unit g)
    matlab_ref_128kHz_10_300.npz      MATLAB R2025a output: freq / psd / nf_db

The reference was produced by running `bitstreamPSDfn_analogcl(x, 10, 300)`
(the original `.m` in the same folder) minus its plotting, i.e. the same
`pwelch(x, WinBH7(131072), 65536, 131072, 128000)` call and
`nf = 10*log10(mean(Psd(Freq>=10 & Freq<=300)))`.

Where the data folder lives is resolved in `_find_data_dir()`: the
`ASD_TEST_DATA` environment variable, then `util/tests/asd_test_data/`
(the committed location), then `<repo>/../asd_test_data/`. If it cannot be
found the parity tests are skipped with an explicit message instead of
silently passing.
'''
import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot)
import matplotlib.pyplot as plt  # noqa: E402

from util.asd import asd_fn, plot_asd, win_bh7  # noqa: E402
from util.signal import Signal  # noqa: E402

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is present in the project env
    h5py = None

CASE = "asd_test_data_128kHz_10_300"
REF = "matlab_ref_128kHz_10_300"
FS = 128000.0
NFFT = 131072
FL, FH = 10.0, 300.0


def _find_data_dir():
    '''Locate the folder holding the .mat input and the MATLAB .npz reference.'''
    here = Path(__file__).resolve()
    candidates = []
    env = os.environ.get("ASD_TEST_DATA")
    if env:
        candidates.append(Path(env))
    candidates.append(here.parent / "asd_test_data")
    candidates.append(here.parents[1] / "data" / "asd_test_data")
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / "asd_test_data")
    for candidate in candidates:
        if (candidate / f"{CASE}.mat").is_file() and (candidate / f"{REF}.npz").is_file():
            return candidate
    return None


DATA_DIR = _find_data_dir()
SKIP_REASON = (
    "asd_test_data/ not found (set ASD_TEST_DATA, or drop the folder into "
    "util/tests/data/)"
)


def bh7_analytic(n):
    '''DFT-even 7-term Blackman-Harris window, written out from its definition.'''
    a = [0.61640321314050, 0.98537119272586, 0.49603771622007, 0.14992232793243,
         0.02458719103474, 0.0017660465148687, 0.000031581188567106]
    k = np.arange(1, 7)[:, None]
    return a[0] + np.sum(
        [((-1) ** i) * a[i] * np.cos(2 * np.pi * i * (np.arange(n) + 0.5) / n)
         for i in range(1, 7)], axis=0)


class TestWindow(unittest.TestCase):
    '''7-term Blackman-Harris window (including the odd-length fix).'''

    def test_even_length_matches_analytic(self):
        for n in (8, 16, 1024):
            with self.subTest(n=n):
                self.assertLess(np.max(np.abs(win_bh7(n) - bh7_analytic(n))), 1e-12)

    def test_odd_length_has_full_length(self):
        # Regression: the MATLAB transliteration returned n-2 samples for odd n,
        # which made periodogram() reject the window.
        for n in (5, 9, 1001, 10001):
            with self.subTest(n=n):
                w = win_bh7(n)
                self.assertEqual(len(w), n)
                self.assertLess(np.max(np.abs(w - bh7_analytic(n))), 1e-12)

    def test_symmetric_and_coherent_gain(self):
        w = win_bh7(4096)
        self.assertEqual(np.max(np.abs(w - w[::-1])), 0.0)
        self.assertAlmostEqual(w.sum() / w.size, 0.61640321314050, places=12)

    def test_rejects_non_positive_length(self):
        for n in (0, -3):
            with self.subTest(n=n), self.assertRaises(ValueError):
                win_bh7(n)


@unittest.skipUnless(DATA_DIR is not None, SKIP_REASON)
@unittest.skipIf(h5py is None, "h5py is required to read the provided .mat input")
class TestMatlabParity(unittest.TestCase):
    '''End-to-end comparison against the MATLAB reference on the provided data.'''

    @classmethod
    def setUpClass(cls):
        cls.ref = np.load(DATA_DIR / f"{REF}.npz")
        with h5py.File(DATA_DIR / f"{CASE}.mat", "r") as f:
            cls.x = np.array(f["chopout4"]).ravel()
        cls.freq, cls.asd, cls.nf = asd_fn(cls.x, FL, FH)

    def test_reference_metadata(self):
        self.assertEqual(int(self.ref["n"]), self.x.size)
        self.assertEqual(int(self.ref["nfft"]), NFFT)
        self.assertEqual(float(self.ref["fs"]), FS)
        self.assertEqual(float(self.ref["fl"]), FL)
        self.assertEqual(float(self.ref["fh"]), FH)

    def test_case_exercises_welch_branch_with_odd_length(self):
        # 327681 samples: odd, and longer than nfft -> Welch branch, even window.
        self.assertEqual(self.x.size, 327681)
        self.assertNotEqual(self.x.size % 2, 0)
        self.assertGreaterEqual(self.x.size, NFFT)

    def test_frequency_axis_matches_matlab(self):
        self.assertTrue(np.allclose(self.freq, self.ref["freq"], rtol=0, atol=1e-9))
        self.assertEqual(self.freq.size, self.ref["freq"].size)
        self.assertAlmostEqual(float(self.freq[1] - self.freq[0]), FS / NFFT, places=10)

    def test_psd_matches_matlab(self):
        psd_python = self.asd ** 2
        psd_matlab = self.ref["psd"]
        np.testing.assert_allclose(psd_python, psd_matlab, rtol=1e-9, atol=0)

    def test_noise_floor_matches_matlab(self):
        self.assertAlmostEqual(self.nf, float(self.ref["nf_db"]), delta=1e-6)
        # and it is the power mean over the requested band
        mask = (self.freq >= FL) & (self.freq <= FH)
        expected = 10 * np.log10(np.mean(self.asd[mask] ** 2))
        self.assertAlmostEqual(self.nf, expected, places=10)


class TestBandValidation(unittest.TestCase):
    '''Invalid / empty noise-floor bands raise instead of returning NaN.'''

    @classmethod
    def setUpClass(cls):
        cls.x = np.random.default_rng(0).normal(size=131072)

    def test_reversed_band(self):
        with self.assertRaises(ValueError):
            asd_fn(self.x, 200, 20)

    def test_negative_lower_edge(self):
        with self.assertRaises(ValueError):
            asd_fn(self.x, -1, 100)

    def test_non_finite_edges(self):
        for fl, fh in ((np.nan, 200), (20, np.inf), (20, np.nan)):
            with self.subTest(fl=fl, fh=fh), self.assertRaises(ValueError):
                asd_fn(self.x, fl, fh)

    def test_band_without_bins(self):
        with self.assertRaises(ValueError):
            asd_fn(self.x, 1e6, 1e7)

    def test_short_record_band_without_bins(self):
        # N=2 -> df = fs/2, so nothing lands in 20..200 Hz
        with self.assertRaises(ValueError):
            asd_fn(np.array([1.0, -1.0]), 20, 200)


class TestSamplingAndSegments(unittest.TestCase):
    '''fs / nfft / noverlap plumbing.'''

    @classmethod
    def setUpClass(cls):
        cls.x = np.random.default_rng(1).normal(size=262144)

    def test_default_sampling_rate_is_128k(self):
        freq, _, _ = asd_fn(self.x, 20, 200)
        self.assertEqual(freq[-1], FS / 2)

    def test_signal_input_matches_array(self):
        t = np.arange(self.x.size) / 256000.0
        sig = Signal(self.x, t=t)
        f_sig, asd_sig, nf_sig = asd_fn(sig, 20, 2000)
        f_ref, asd_ref, nf_ref = asd_fn(self.x, 20, 2000, fs=256000.0)
        self.assertAlmostEqual(nf_sig, nf_ref, places=12)
        np.testing.assert_array_equal(asd_sig, asd_ref)
        self.assertEqual(f_sig[-1], 128000.0)

    def test_explicit_fs_wins_over_signal_dt(self):
        t = np.arange(self.x.size) / 256000.0
        sig = Signal(self.x, t=t)
        freq, _, _ = asd_fn(sig, 20, 2000, fs=128000.0)
        self.assertEqual(freq[-1], 64000.0)

    def test_nfft_controls_resolution(self):
        freq, _, _ = asd_fn(self.x, 20, 200, nfft=4096)
        self.assertAlmostEqual(float(freq[1] - freq[0]), FS / 4096, places=10)
        self.assertEqual(freq.size, 4096 // 2 + 1)

    def test_noverlap_changes_segment_count(self):
        _, asd_half, _ = asd_fn(self.x, 20, 200, nfft=4096)
        _, asd_none, _ = asd_fn(self.x, 20, 200, nfft=4096, noverlap=0)
        self.assertFalse(np.array_equal(asd_half, asd_none))

    def test_nfft_larger_than_record_falls_back_to_one_segment(self):
        short = self.x[:50000]
        freq, _, _ = asd_fn(short, 20, 200, nfft=NFFT)
        self.assertAlmostEqual(float(freq[1] - freq[0]), FS / 50000, places=10)
        self.assertEqual(freq.size, 50000 // 2 + 1)

    def test_invalid_fs(self):
        for fs in (0.0, -1.0, np.nan):
            with self.subTest(fs=fs), self.assertRaises(ValueError):
                asd_fn(self.x, fs=fs)

    def test_invalid_nfft_and_noverlap(self):
        for kwargs in ({"nfft": 1}, {"nfft": 1024, "noverlap": 1024},
                       {"nfft": 1024, "noverlap": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                asd_fn(self.x, **kwargs)


class TestUnitsAndShapes(unittest.TestCase):
    '''ASD convention, input shape handling and degenerate spectra.'''

    @classmethod
    def setUpClass(cls):
        cls.x = np.random.default_rng(2).normal(size=131072)

    def test_noise_floor_is_power_mean_of_asd(self):
        freq, asd, nf = asd_fn(self.x, 20, 200)
        mask = (freq >= 20) & (freq <= 200)
        power_mean = 20 * np.log10(np.sqrt(np.mean(asd[mask] ** 2)))
        amplitude_mean = 20 * np.log10(np.mean(asd[mask]))
        self.assertAlmostEqual(nf, power_mean, places=10)
        # the two averaging conventions must not be silently interchangeable
        self.assertGreater(abs(nf - amplitude_mean), 0.05)

    def test_parseval(self):
        freq, asd, _ = asd_fn(self.x, 20, 200)
        df = freq[1] - freq[0]
        self.assertAlmostEqual(float(np.sum(asd ** 2) * df), 1.0, delta=0.02)

    def test_column_and_row_vectors_are_accepted(self):
        _, asd_ref, nf_ref = asd_fn(self.x, 20, 200)
        for shaped in (self.x.reshape(-1, 1), self.x.reshape(1, -1)):
            with self.subTest(shape=shaped.shape):
                _, asd, nf = asd_fn(shaped, 20, 200)
                np.testing.assert_array_equal(asd, asd_ref)
                self.assertEqual(nf, nf_ref)

    def test_multi_channel_input_is_rejected(self):
        for shaped in (self.x.reshape(2, -1), np.ones((3, 4, 5))):
            with self.subTest(shape=shaped.shape), self.assertRaises(ValueError):
                asd_fn(shaped)

    def test_too_few_samples(self):
        for n in (0, 1):
            with self.subTest(n=n), self.assertRaises(ValueError):
                asd_fn(np.zeros(n))

    def test_zero_spectrum_yields_minus_inf_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, asd, nf = asd_fn(np.zeros(131072), 20, 200)
        self.assertTrue(np.all(asd == 0.0))
        self.assertEqual(nf, float("-inf"))
        self.assertEqual([str(w.message) for w in caught], [])


class TestPlotAsd(unittest.TestCase):
    '''plot_asd returns composable axes and covers the whole band.'''

    @classmethod
    def setUpClass(cls):
        cls.x = np.random.default_rng(3).normal(size=131072)
        cls.freq, cls.asd, cls.nf = asd_fn(cls.x, 20, 200)

    def tearDown(self):
        plt.close("all")

    def test_returns_figure_and_axes_covering_nyquist(self):
        fig, ax = plot_asd(self.freq, self.asd, self.nf, band=(20, 200), label="a")
        self.assertIs(ax.figure, fig)
        left, right = ax.get_xlim()
        self.assertAlmostEqual(right, FS / 2, places=6)
        self.assertLess(left, 2.0)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertIn("ASD", ax.get_ylabel())
        self.assertIsNotNone(ax.get_legend())

    def test_overlay_two_runs(self):
        fig, ax = plot_asd(self.freq, self.asd, self.nf, label="a")
        plot_asd(self.freq, self.asd * 2, self.nf + 6.0, ax=ax, label="b", band=(20, 200))
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertEqual(len(labels), 4)  # 2 curves + 2 noise-floor lines
        self.assertIn("a", labels)
        self.assertIn("b", labels)

    def test_zero_spectrum_is_plottable(self):
        _, asd, nf = asd_fn(np.zeros(131072), 20, 200)
        fig, ax = plot_asd(self.freq, asd, nf)
        self.assertTrue(all(np.isfinite(ax.get_ylim())))

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            plot_asd(self.freq, self.asd[:-1])


if __name__ == "__main__":
    unittest.main()
