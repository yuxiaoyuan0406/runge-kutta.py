'''
Figures that show what the closed loop Kalman filter buys, drawn with the
repository plotting tools of :mod:`util`.

The script runs the estimator on one measurement sequence and writes four
figures:

``01-acceleration-overview.png``
    the whole record: true acceleration, Kalman estimate, PID baseline.
``02-window-zoom.png``
    the static window and the settled window, zoomed in, where the difference
    between the three curves is visible.
``03-error-comparison.png``
    the two error signals over the settled window, with their RMS.
``04-error-asd.png``
    amplitude spectral density of the two error signals.

Usage::

    python examples/kalman_accelerometer/plot_report.py \\
        --config examples/kalman_accelerometer/configs/xiong-2018.json \\
        --data data/20261026-unit_step_1 -o figures/unit_step_1
'''
import argparse
import os
import sys

import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.append('.')
import util
from examples.kalman_accelerometer import config as config_module
from examples.kalman_accelerometer import data_adapter
from examples.kalman_accelerometer.evaluation import error_metrics, pid_baseline
from examples.kalman_accelerometer.kalman import ClosedLoopKalman
from examples.kalman_accelerometer.model import ClosedLoopModel

COLOR_TRUTH = 'black'
COLOR_KALMAN = 'tab:blue'
COLOR_BASELINE = 'tab:red'


def argue_parser() -> argparse.ArgumentParser:
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Plot the Kalman estimate against the PID baseline.')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file in JSON format.')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory or override of data_path.')
    parser.add_argument('-o', '--out', type=str, required=True,
                        help='Directory for the figures.')
    parser.add_argument('--panels', type=str, default='',
                        help='Comma separated window names used as the zoom panels, '
                             'defaults to the first two windows of the configuration.')
    parser.add_argument('--error-window', type=str, default='',
                        help='Window name used for the error and ASD figures; defaults '
                             'to steady, or the last window.  A short window keeps a '
                             'long record readable.')
    parser.add_argument('--dpi', type=int, default=200, help='Figure resolution.')
    return parser


def build_signals(t, truth, estimate, baseline):
    '''
    Wrap the three curves into ``util.Signal`` objects.
    '''
    return (
        util.Signal(truth, t=t, color=COLOR_TRUTH, linestyle='--',
                    label='True acceleration'),
        util.Signal(estimate, t=t, color=COLOR_KALMAN,
                    label='Kalman estimate'),
        util.Signal(baseline, t=t, color=COLOR_BASELINE, linestyle=':',
                    label='PID baseline $-K_u u / m$'),
    )


def save(fig, directory, name, dpi):
    '''
    Save one figure and report its path.
    '''
    path = os.path.join(directory, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')
    return path


def figure_overview(t, signals, directory, dpi):
    '''
    Whole record: truth, Kalman estimate and PID baseline.
    '''
    fig, ax = util.default_time_plot_fig(
        title='Closed loop Kalman estimate of the external acceleration',
        ylabel='Acceleration [m/s$^2$]')
    for signal in signals:
        signal.plot_time_domain(ax=ax)
    ax.set_xlim(t[0], t[-1] * 1.02)
    return save(fig, directory, '01-acceleration-overview.png', dpi)


def figure_zoom(t, signals, panels, directory, dpi):
    '''
    Zoom on the configured evaluation windows.

    Args:
        t: Sample times.
        signals: The three ``util.Signal`` curves.
        panels: List of ``(title, [start, end])`` pairs, one panel each.
        directory: Output directory.
        dpi: Figure resolution.
    '''
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.6 * len(panels)),
                             sharex=False)
    for ax, (title, window) in zip(np.atleast_1d(axes), panels):
        start, end = window
        mask = (t >= start) & (t <= end)
        for signal in signals:
            ax.plot(signal.t[mask], np.real(signal.x[mask]),
                    color=signal.color, linestyle=signal.linestyle,
                    label=signal.label)
        ax.set_xlim(start, end)
        ax.set_title(f'{title} window  [{start * 1e3:.2f}, {end * 1e3:.2f}] ms',
                     fontsize=11)
        ax.set_ylabel('a [m/s$^2$]')
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel('Time [s]')
    fig.suptitle('Estimate against the PID baseline, window by window',
                 fontsize=util.FIG_TITLE_FONT_SIZE)
    fig.tight_layout()
    return save(fig, directory, '02-window-zoom.png', dpi)


def figure_error(kalman_error, baseline_error, window, metrics, directory, dpi):
    '''
    Error signals over the settled window.

    ``util.Signal`` re-bases time to zero on construction, so the full error
    signals are plotted and the axes are limited to the window instead of
    slicing them.
    '''
    start, end = window
    fig, ax = util.default_time_plot_fig(
        title='Estimation error in the settled window',
        ylabel='Estimate - truth [m/s$^2$]')
    kalman_error.plot_time_domain(ax=ax)
    baseline_error.plot_time_domain(ax=ax)
    ax.axhline(0., color='black', linewidth=0.8)
    ax.set_xlim(start, end)
    # Limiting x is not enough: the axis range would still be set by the step
    # transient of the full record, so limit y from the samples in the window.
    window_mask = (kalman_error.t >= start) & (kalman_error.t <= end)
    inside = np.concatenate([np.real(kalman_error.x[window_mask]),
                             np.real(baseline_error.x[window_mask])])
    limit = float(np.max(np.abs(inside))) * 1.15
    ax.set_ylim(-limit, limit)
    text = ('RMS  Kalman {:.3e}\n      baseline {:.3e}\n      ratio {:.1f}x'.format(
        metrics['kalman']['rmse'], metrics['baseline']['rmse'],
        metrics['baseline']['rmse'] / metrics['kalman']['rmse']))
    ax.text(0.02, 0.03, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    return save(fig, directory, '03-error-comparison.png', dpi)


def welch_asd(x, dt, nperseg=128, noverlap=64):
    '''
    One sided amplitude spectral density with Welch averaging.

    ``util.compute_asd_welch`` is deliberately not used here: it calls
    ``np.trapezoid`` for its self check, which only exists in numpy 2.

    Args:
        x: Signal samples.
        dt: Sampling period in seconds.
        nperseg: Segment length.
        noverlap: Overlap between segments.

    Returns:
        tuple: ``(f, asd)`` in Hz and unit/sqrt(Hz).
    '''
    x = np.asarray(x, dtype=np.float64).ravel()
    nperseg = int(min(nperseg, x.size))
    noverlap = int(min(noverlap, nperseg // 2))
    frequencies, psd = scipy.signal.welch(
        x, fs=1. / dt, window='hann', nperseg=nperseg, noverlap=noverlap,
        detrend='constant', scaling='density', return_onesided=True)
    return frequencies, np.sqrt(psd)


def figure_asd(kalman_error, baseline_error, dt, directory, dpi):
    '''
    Amplitude spectral density of the two error signals.

    Args:
        kalman_error: Error samples of the estimator on the evaluation window.
        baseline_error: Error samples of the PID baseline on the same window.
        dt: Sampling period in seconds.
    '''
    f, asd_kalman = welch_asd(kalman_error, dt)
    _, asd_base = welch_asd(baseline_error, dt)

    fig, ax = util.default_noise_plot_fig(
        title='Amplitude spectral density of the estimation error')
    ax.loglog(f[1:], asd_kalman[1:], color=COLOR_KALMAN,
              label=f'Kalman (RMS {np.std(kalman_error):.2e})')
    ax.loglog(f[1:], asd_base[1:], color=COLOR_BASELINE, linestyle=':',
              label=f'PID baseline (RMS {np.std(baseline_error):.2e})')
    ax.set_ylabel('ASD [(m/s$^2$)/$\\sqrt{Hz}$]')
    ax.legend(loc='upper right', fontsize=9)
    ax.text(0.02, 0.03, f'$\\Delta f$ = {f[1] - f[0]:.0f} Hz, '
            f'nperseg = {min(128, kalman_error.size)}',
            transform=ax.transAxes, fontsize=8)
    return save(fig, directory, '04-error-asd.png', dpi)


def main(argv=None) -> int:
    '''
    Run the estimator and write the figures.
    '''
    args = argue_parser().parse_args(argv)

    config = config_module.load_config(args.config)
    config.data_path = args.data
    model = ClosedLoopModel(config)
    data = data_adapter.load_data(config)
    if data.a_true is None:
        raise SystemExit('the data set has no true acceleration, nothing to compare')

    estimator = ClosedLoopKalman(model, config.X0_hat, config.P0, config.tolerances)
    history = estimator.run(data.u)

    t = data.t
    truth = np.asarray(data.a_true, dtype=np.float64)
    estimate = np.asarray(history['acceleration'], dtype=np.float64)
    baseline = pid_baseline(data.u, config.K_u, config.m, config.baseline_gain)

    os.makedirs(args.out, exist_ok=True)
    signals = build_signals(t, truth, estimate, baseline)

    windows = config.test_windows or {}
    if not windows:
        windows = {'all': [float(t[0]), float(t[-1])]}

    names = [item.strip() for item in args.panels.split(',') if item.strip()]
    if not names:
        names = list(windows)[:2]
    panels = [(name, windows.get(name, [float(t[0]), float(t[-1])]))
              for name in names]

    steady_name = args.error_window or ('steady' if 'steady' in windows
                                        else list(windows)[-1])
    if steady_name not in windows:
        raise SystemExit(f'window "{steady_name}" is not in the configuration')
    steady_window = windows[steady_name]

    figure_overview(t, signals, args.out, args.dpi)
    figure_zoom(t, signals, panels, args.out, args.dpi)

    start, end = _window_bounds(t, steady_window)
    kalman_error = signals[1] - signals[0]
    baseline_error = signals[2] - signals[0]
    kalman_error.label = 'Kalman error'
    baseline_error.label = 'Baseline error'
    kalman_error.color = COLOR_KALMAN
    baseline_error.color = COLOR_BASELINE
    kalman_error.linestyle = '-'
    baseline_error.linestyle = ':'

    metrics = {
        'kalman': error_metrics(estimate, truth, start, end),
        'baseline': error_metrics(baseline, truth, start, end),
    }
    figure_error(kalman_error, baseline_error, (float(t[start]), float(t[end - 1])),
                 metrics, args.out, args.dpi)
    figure_asd(kalman_error.x[start:end], baseline_error.x[start:end],
               data.dt, args.out, args.dpi)

    print(f'steady window [{steady_window[0]}, {steady_window[1]}] s, '
          f'{end - start} samples')
    print('  Kalman   RMSE {:.4e} m/s^2'.format(metrics['kalman']['rmse']))
    print('  baseline RMSE {:.4e} m/s^2'.format(metrics['baseline']['rmse']))
    return 0


def _window_bounds(t, window):
    '''
    Convert a ``[start, end]`` window in seconds into index bounds.
    '''
    start, end = float(window[0]), float(window[1])
    mask = np.nonzero((t >= start) & (t <= end))[0]
    if mask.size == 0:
        raise SystemExit(f'window [{start}, {end}] s contains no samples')
    return int(mask[0]), int(mask[-1]) + 1


if __name__ == '__main__':
    sys.exit(main())
