'''
Bode plot of the two baseline low pass filters used by the comparison.

The curves are the *digital* responses actually implemented by
``util.baseline_filters`` (``frequency_response`` / coefficients), not the
analog prototypes, so the RC curve shows the small deviation of its -3 dB
point from ``fc`` and the Butterworth -3 dB point sits exactly on ``fc``
because of the prewarped bilinear transform.

The requested range 1 Hz - 100 kHz is wider than the Nyquist frequency of the
128 kHz sampling (64 kHz).  The response of a discrete time filter is periodic
in ``f``, so the part above ``fs/2`` is drawn dashed and labelled: it is the
aliased image of the 28-64 kHz band, not new attenuation.  The Butterworth
notch at 64 kHz is its zero at ``z = -1``.

Usage (project root)::

    python examples/kalman_accelerometer/bode_baselines.py --fc 1000 \\
        --fmin 1 --fmax 100000 -o figures/filter-bode-1khz.png
'''
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import util
from util.baseline_filters import ButterworthFilter, RCFilter

COLOR_RC = 'tab:blue'
COLOR_BW = 'tab:orange'
SENSOR_BAND = (0., 500.)


def argue_parser() -> argparse.ArgumentParser:
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Bode plot of the RC and Butterworth baseline filters.')
    parser.add_argument('--fs', type=float, default=128000.0,
                        help='Sampling frequency in Hz, default 128000.')
    parser.add_argument('--fc', type=float, default=1000.0,
                        help='Cutoff of both filters in Hz, default 1000.')
    parser.add_argument('-o', '--out', type=str, default='figures/filter-bode.png',
                        help='Output PNG path.')
    parser.add_argument('--fmin', type=float, default=1.0,
                        help='Lowest plotted frequency in Hz, default 1.')
    parser.add_argument('--fmax', type=float, default=100000.0,
                        help='Highest plotted frequency in Hz, default 100000.')
    parser.add_argument('--dpi', type=int, default=200, help='Figure resolution.')
    return parser


def digital_response(filt, frequencies, fs):
    '''
    ``H(e^{j 2 pi f / fs})`` for any frequency, including above Nyquist.

    ``BaselineFilter.frequency_response`` refuses ``f > fs/2`` because above
    Nyquist the value is an alias.  For the plot the periodic extension is
    wanted, so it is evaluated from the published coefficients and checked
    against the class method on the physical band.
    '''
    z_inverse = np.exp(-1j * 2. * np.pi * np.asarray(frequencies) / fs)
    if isinstance(filt, RCFilter):
        coefficients = filt.coefficients
        return coefficients['beta'] / (1. - coefficients['alpha'] * z_inverse)
    coefficients = filt.coefficients
    numerator = (coefficients['b0'] + coefficients['b1'] * z_inverse
                 + coefficients['b2'] * z_inverse ** 2)
    denominator = (1. + coefficients['a1'] * z_inverse
                   + coefficients['a2'] * z_inverse ** 2)
    return numerator / denominator


def check_response(filt, frequencies, fs, label):
    '''
    Compare the coefficient evaluation with the class method below Nyquist.
    '''
    physical = frequencies[frequencies <= fs / 2.]
    reference = np.atleast_1d(filt.frequency_response(physical))
    computed = digital_response(filt, physical, fs)
    error = float(np.max(np.abs(reference - computed)))
    print(f'{label}: coefficient evaluation vs frequency_response, '
          f'max |diff| = {error:.2e}')
    if error > 1e-12:
        raise SystemExit(f'{label}: response mismatch, refusing to plot')


def curves(frequencies, response):
    '''
    Magnitude in dB and unwrapped phase in degrees.
    '''
    magnitude = 20. * np.log10(np.maximum(np.abs(response), 1e-30))
    phase = np.degrees(np.unwrap(np.angle(response)))
    return magnitude, phase


def annotate(axis, text, frequency, value, color, offset=(6, 8)):
    '''
    Place one short annotation next to a marked point.
    '''
    axis.annotate(text, xy=(frequency, value), xytext=offset,
                  textcoords='offset points', fontsize=7.5, color=color,
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))


def main(argv=None) -> int:
    '''
    Draw the two Bode panels and write the figure.

    Only the physical band `f <= fs/2` is plotted: above Nyquist the response
    of a discrete time filter is the aliased image of 28-64 kHz, not extra
    attenuation, so that region is shaded and labelled instead of drawn.
    '''
    args = argue_parser().parse_args(argv)
    nyquist = args.fs / 2.
    if args.fmax <= nyquist:
        raise SystemExit(f'fmax must exceed Nyquist ({nyquist:g} Hz) for this '
                         'plot to show the aliased region')
    frequencies = np.logspace(np.log10(args.fmin), np.log10(args.fmax), 4000)
    physical = frequencies <= nyquist

    rc = RCFilter(fs=args.fs, fc=args.fc)
    bw = ButterworthFilter(fs=args.fs, fc=args.fc)
    check_response(rc, physical, args.fs, 'RC')
    check_response(bw, physical, args.fs, 'Butterworth')

    rc_mag, rc_phase = curves(frequencies, digital_response(rc, frequencies, args.fs))
    bw_mag, bw_phase = curves(frequencies, digital_response(bw, frequencies, args.fs))

    figure, (ax_mag, ax_phase) = util.default_freq_plot_fig(
        title=f'Baseline low pass filters, $f_c$ = {args.fc:g} Hz')
    ax_mag.set_ylabel('Magnitude [dB]')
    ax_phase.set_ylabel('Phase [deg]')

    ax_mag.plot(frequencies[physical], rc_mag[physical], color=COLOR_RC,
                linestyle='--', label=f'RC, $f_c$ = {rc.fc:g} Hz')
    ax_mag.plot(frequencies[physical], bw_mag[physical], color=COLOR_BW,
                linestyle='-', label=f'Butterworth (2nd order), $f_c$ = {bw.fc:g} Hz')
    ax_phase.plot(frequencies[physical], rc_phase[physical], color=COLOR_RC,
                  linestyle='--', label='RC')
    ax_phase.plot(frequencies[physical], bw_phase[physical], color=COLOR_BW,
                  linestyle='-', label='Butterworth')

    for axis in (ax_mag, ax_phase):
        axis.axvspan(args.fmin, SENSOR_BAND[1], color='gray', alpha=0.08,
                     label='0-500 Hz sensor band')
        axis.axvspan(nyquist, args.fmax, color='0.55', alpha=0.15,
                     label=f'above Nyquist $f_s/2$ = {nyquist / 1e3:g} kHz (aliased)')
        axis.axvline(args.fc, color='black', linestyle=':', linewidth=0.9)
        axis.axvline(nyquist, color='0.35', linestyle='-.', linewidth=0.9)
        axis.set_xlim(args.fmin, args.fmax)
    ax_mag.axhline(-3., color='black', linestyle=':', linewidth=0.9)
    ax_mag.set_ylim(-80., 5.)
    ax_phase.set_ylim(-190., 10.)
    ax_phase.set_yticks(np.arange(-180., 1., 45.))

    ax_mag.text(nyquist * 1.12, -70., 'aliased\nregion', fontsize=7.5,
                color='0.35', ha='left', va='bottom')

    probes = np.array([125., 500., args.fc])
    for filt, color, name in ((rc, COLOR_RC, 'RC'), (bw, COLOR_BW, 'BW')):
        response = digital_response(filt, probes, args.fs)
        magnitude, phase = curves(probes, response)
        ax_mag.plot(probes, magnitude, 'o', color=color, markersize=4)
        ax_phase.plot(probes, phase, 'o', color=color, markersize=4)
        annotate(ax_mag, f'{name} {magnitude[1]:.2f} dB @500 Hz', probes[1],
                 magnitude[1], color, offset=(6, 8 if name == 'RC' else -30))
        annotate(ax_phase, f'{name} {phase[1]:.1f} deg @500 Hz', probes[1],
                 phase[1], color, offset=(6, 8 if name == 'RC' else -16))
    ax_mag.text(args.fc * 1.05, -74., '-3 dB at $f_c$', fontsize=7.5,
                color='black', ha='left', va='bottom')

    ax_mag.set_title('')
    ax_mag.legend(loc='lower left', fontsize=7)
    ax_phase.legend(loc='lower left', fontsize=7)
    ax_mag.text(0.99, 0.97, f'$f_s$ = {args.fs:g} Hz, digital response',
                transform=ax_mag.transAxes, ha='right', va='top', fontsize=7.5,
                color='0.35')

    directory = os.path.dirname(args.out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    plt.close(figure)

    print(f'filters: RC fc={rc.fc:g} Hz, Butterworth fc={bw.fc:g} Hz, '
          f'fs={args.fs:g} Hz, Nyquist={nyquist:g} Hz')
    for frequency in (1., 125., 500., args.fc, 2000., 10000., 60000.):
        rc_m, rc_p = curves(np.array([frequency]),
                            digital_response(rc, np.array([frequency]), args.fs))
        bw_m, bw_p = curves(np.array([frequency]),
                            digital_response(bw, np.array([frequency]), args.fs))
        print(f'  {frequency:8.0f} Hz  RC {rc_m[0]:8.2f} dB {rc_p[0]:8.1f} deg   '
              f'BW {bw_m[0]:8.2f} dB {bw_p[0]:8.1f} deg')
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
