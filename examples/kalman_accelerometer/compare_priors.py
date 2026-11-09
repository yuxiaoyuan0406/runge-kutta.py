'''
Comparison of the two acceleration priors on one measurement sequence.

The script runs the ZOH and the FOH estimator on the *same* samples with the
same units and the same evaluation windows, adds the phase 1 static force
balance baseline and writes

``comparison.json``
    every number of the comparison, including the innovation statistics and
    the Jerk diagnostics;
``comparison.md``
    a short table view;
``01-comparison-overview.png``
    true acceleration, both estimates and the baseline over the whole record;
``02-comparison-zoom.png``
    one panel per configured window;
``03-comparison-error.png``
    the three error signals on the settled window;
``04-comparison-jerk.png``
    the FOH Jerk estimate with its posterior band;
``05-comparison-asd.png``
    amplitude spectral density of the three error signals.

Usage::

    python examples/kalman_accelerometer/compare_priors.py \\
        --zoh-config examples/kalman_accelerometer/configs/xiong-2018.json \\
        --foh-config examples/kalman_accelerometer/configs/xiong-2018-foh.json \\
        --data data/20261026-unit_step_1 -o data/compare-unit-step
'''
import argparse
import os
import sys

import numpy as np

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from examples.kalman_accelerometer import config as config_module
from examples.kalman_accelerometer import data_adapter, evaluation
from examples.kalman_accelerometer.kalman import ClosedLoopKalman

COLOR_TRUTH = 'black'
COLOR_ZOH = 'tab:blue'
COLOR_FOH = 'tab:green'
COLOR_BASELINE = 'tab:red'


def argue_parser() -> argparse.ArgumentParser:
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Compare the ZOH and FOH acceleration prior on one data set.')
    parser.add_argument('--zoh-config', type=str, required=True,
                        help='Phase 1 ZOH configuration.')
    parser.add_argument('--foh-config', type=str, required=True,
                        help='Phase 2 FOH configuration of the same device and data.')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory, overrides data_path of both files.')
    parser.add_argument('-o', '--out', type=str, required=True,
                        help='Output directory.')
    parser.add_argument('--dpi', type=int, default=200, help='Figure resolution.')
    return parser


def load_pair(args) -> tuple:
    '''
    Load both configurations and enforce that they describe the same record.
    '''
    zoh = config_module.load_config(args.zoh_config)
    foh = config_module.load_config(args.foh_config)
    zoh.data_path = args.data
    foh.data_path = args.data

    problems = []
    if zoh.prior_model != 'ZOH':
        problems.append(f'{args.zoh_config} is not a ZOH configuration')
    if foh.prior_model != 'FOH':
        problems.append(f'{args.foh_config} is not an FOH configuration')
    for attribute in ('fs', 'Ts', 'm', 'c', 'K_u', 'k_cv', 'k_adc',
                      'K_p', 'K_i', 'K_d', 'sigma_n2', 'mechanical_noise_psd'):
        if not np.isclose(getattr(zoh, attribute), getattr(foh, attribute),
                          rtol=0., atol=0.):
            problems.append(f'{attribute} differs between the two configurations')
    if zoh.test_windows != foh.test_windows:
        problems.append('test_windows differ between the two configurations')
    if problems:
        raise SystemExit('; '.join(problems))
    return zoh, foh


def run_estimator(config) -> tuple:
    '''
    Load the data, filter it and evaluate the estimate.
    '''
    data = data_adapter.load_data(config)
    for message in data.warnings:
        print(f'warning: {message}')
    estimator = ClosedLoopKalman.initialize(config)
    history = estimator.run(data.u)
    model = estimator.model
    metrics = evaluation.evaluate(data.t, history, data, config, model)
    return data, history, model, metrics


def window_row(name: str, window, zoh: dict, foh: dict, baseline: dict,
               data, foh_metrics: dict) -> dict:
    '''
    Collect the ZOH, FOH and baseline numbers of one evaluation window.
    '''
    start, end = evaluation.window_slice(data.t, window)
    truth = data.a_true
    row = {
        'window': str(name),
        'bounds': [float(window[0]), float(window[1])],
        'samples': int(end - start),
        'zoh': evaluation.error_metrics(zoh['acceleration'], truth, start, end),
        'foh': evaluation.error_metrics(foh['acceleration'], truth, start, end),
        'baseline': evaluation.error_metrics(baseline, truth, start, end),
    }
    if row['zoh']['rmse'] > 0:
        row['foh_over_zoh_rmse'] = float(row['foh']['rmse'] / row['zoh']['rmse'])
        row['foh_gain_db'] = float(20. * np.log10(row['zoh']['rmse'] / row['foh']['rmse']))
    row['zoh_innovation'] = evaluation.nis_statistics(
        zoh['innovation'][start:end], zoh['innovation_cov'][start:end])
    row['foh_innovation'] = evaluation.nis_statistics(
        foh['innovation'][start:end], foh['innovation_cov'][start:end])
    window_metrics = (foh_metrics.get('windows') or {}).get(str(name)) or {}
    if window_metrics.get('jerk') is not None:
        row['foh_jerk'] = window_metrics['jerk']
    return row


def save_figure(figure, path: str):
    '''
    Save and close a figure.
    '''
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(figure)


def plot_overview(t, truth, zoh, foh, baseline, path):
    '''
    Whole record: truth, both estimates and the baseline.
    '''
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(t, truth, color=COLOR_TRUTH, linestyle='-', linewidth=1.4,
              label='True acceleration')
    axis.plot(t, zoh, color=COLOR_ZOH, linestyle='--', linewidth=1.0,
              label='ZOH estimate')
    axis.plot(t, foh, color=COLOR_FOH, linestyle='-', linewidth=1.0,
              label='FOH estimate')
    axis.plot(t, baseline, color=COLOR_BASELINE, linestyle=':', linewidth=1.0,
              label='PID baseline')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Acceleration [m/s$^2$]')
    axis.set_title('Acceleration estimate: ZOH against FOH prior')
    axis.grid(True)
    axis.legend(loc='upper left')
    figure.tight_layout()
    save_figure(figure, path)


def plot_zoom(t, truth, zoh, foh, baseline, windows, path):
    '''
    One panel per evaluation window.
    '''
    names = list(windows)
    figure, axes = plt.subplots(len(names), 1, figsize=(9, 3.2 * len(names)),
                                squeeze=False)
    for axis, name in zip(axes.ravel(), names):
        start, end = windows[name]
        mask = (t >= start) & (t <= end)
        axis.plot(t[mask], truth[mask], color=COLOR_TRUTH, linestyle='-',
                  linewidth=1.4, label='Truth')
        axis.plot(t[mask], zoh[mask], color=COLOR_ZOH, linestyle='--',
                  label='ZOH')
        axis.plot(t[mask], foh[mask], color=COLOR_FOH, linestyle='-',
                  label='FOH')
        axis.plot(t[mask], baseline[mask], color=COLOR_BASELINE, linestyle=':',
                  label='PID baseline')
        axis.set_xlim(start, end)
        axis.set_title(f'{name} window [{start * 1e3:.2f}, {end * 1e3:.2f}] ms',
                       fontsize=10)
        axis.set_ylabel('a [m/s$^2$]')
        axis.grid(True)
        axis.legend(loc='upper right', fontsize=8)
    axes.ravel()[-1].set_xlabel('Time [s]')
    figure.tight_layout()
    save_figure(figure, path)


def plot_error(t, truth, zoh, foh, baseline, window, rows, path):
    '''
    Error signals of the three methods on the settled window.
    '''
    start, end = window
    mask = (t >= start) & (t <= end)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(t[mask], (zoh - truth)[mask], color=COLOR_ZOH, linestyle='--',
              label='ZOH error')
    axis.plot(t[mask], (foh - truth)[mask], color=COLOR_FOH, linestyle='-',
              label='FOH error')
    axis.plot(t[mask], (baseline - truth)[mask], color=COLOR_BASELINE,
              linestyle=':', label='Baseline error')
    axis.axhline(0., color='black', linewidth=0.8)
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Estimate - truth [m/s$^2$]')
    axis.set_title('Estimation error in the settled window')
    axis.grid(True)
    text = ('RMS  FOH {:.3e}\n      ZOH {:.3e}\n      baseline {:.3e}'.format(
        rows['foh']['rmse'], rows['zoh']['rmse'], rows['baseline']['rmse']))
    axis.text(0.02, 0.03, text, transform=axis.transAxes, fontsize=9,
              verticalalignment='bottom',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    axis.legend(loc='upper right', fontsize=9)
    figure.tight_layout()
    save_figure(figure, path)


def plot_jerk(t, history, window, path):
    '''
    FOH Jerk estimate with the two sigma posterior band.
    '''
    jerk = np.asarray(history['jerk'], dtype=np.float64)
    deviation = np.sqrt(np.maximum(
        np.asarray(history['jerk_variance'], dtype=np.float64), 0.))
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(t, jerk, color=COLOR_FOH, label='$\\hat j$')
    axis.fill_between(t, jerk - 2. * deviation, jerk + 2. * deviation,
                      color=COLOR_FOH, alpha=0.2, label='$\\pm 2\\sigma$')
    if window:
        axis.axvspan(float(window[0]), float(window[1]), color='gray', alpha=0.1,
                     label='settled window')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Jerk [m/s$^3$]')
    axis.set_title('FOH Jerk estimate and its posterior band')
    axis.grid(True)
    axis.legend(loc='upper right', fontsize=9)
    figure.tight_layout()
    save_figure(figure, path)


def plot_asd(t, truth, zoh, foh, baseline, window, dt, path):
    '''
    Amplitude spectral density of the three error signals on the window.
    '''
    start, end = evaluation.window_slice(t, window)
    spectra = {
        'FOH error': foh[start:end] - truth[start:end],
        'ZOH error': zoh[start:end] - truth[start:end],
        'Baseline error': baseline[start:end] - truth[start:end],
    }
    figure, axis = plt.subplots(figsize=(9, 5))
    colors = {'FOH error': COLOR_FOH, 'ZOH error': COLOR_ZOH,
              'Baseline error': COLOR_BASELINE}
    linestyles = {'FOH error': '-', 'ZOH error': '--', 'Baseline error': ':'}
    for name, signal in spectra.items():
        estimate = evaluation.spectrum(signal, dt)
        axis.loglog(estimate['f'][1:], estimate['asd'][1:], color=colors[name],
                    linestyle=linestyles[name],
                    label=f"{name} (RMS {np.std(signal):.2e})")
    axis.set_xlabel('Frequency [Hz]')
    axis.set_ylabel('ASD [(m/s$^2$)/$\\sqrt{Hz}$]')
    axis.set_title('Amplitude spectral density of the estimation error')
    axis.grid(True, which='both')
    axis.legend(loc='upper right', fontsize=9)
    figure.tight_layout()
    save_figure(figure, path)


def main(argv=None) -> int:
    '''
    Run the comparison and write the report files.
    '''
    args = argue_parser().parse_args(argv)
    zoh_config, foh_config = load_pair(args)

    data_zoh, zoh, zoh_model, zoh_metrics = run_estimator(zoh_config)
    data_foh, foh, foh_model, foh_metrics = run_estimator(foh_config)

    if len(data_zoh) != len(data_foh) or not np.array_equal(data_zoh.u, data_foh.u):
        raise SystemExit('the two configurations did not load the same samples')
    if data_zoh.a_true is None:
        raise SystemExit('the data set has no true acceleration, nothing to compare')

    t = data_zoh.t
    truth = np.asarray(data_zoh.a_true, dtype=np.float64)
    zoh_acceleration = np.asarray(zoh['acceleration'], dtype=np.float64)
    foh_acceleration = np.asarray(foh['acceleration'], dtype=np.float64)
    baseline = evaluation.pid_baseline(data_zoh.u, zoh_config.K_u, zoh_config.m,
                                       zoh_config.baseline_gain)

    windows = zoh_config.test_windows or {'all': [float(t[0]), float(t[-1])]}
    rows = [window_row(name, window, zoh, foh, baseline, data_zoh, foh_metrics)
            for name, window in windows.items()]

    settled_name = 'steady' if 'steady' in windows else list(windows)[-1]
    settled = next(row for row in rows if row['window'] == settled_name)

    report = {
        'data': data_zoh.summary(),
        'zoh_config': zoh_config.summary(),
        'foh_config': foh_config.summary(),
        'zoh_model': zoh_model.as_dict(),
        'foh_model': foh_model.as_dict(),
        'zoh_observability': zoh_model.observability_report().as_dict(),
        'foh_observability': foh_model.observability_report().as_dict(),
        'windows': rows,
        'settled_window': settled_name,
        'whole_record': {
            'zoh': evaluation.error_metrics(zoh_acceleration, truth),
            'foh': evaluation.error_metrics(foh_acceleration, truth),
            'baseline': evaluation.error_metrics(baseline, truth),
        },
        'zoh_innovation': zoh_metrics['innovation'],
        'foh_innovation': foh_metrics['innovation'],
        'foh_jerk': foh_metrics.get('jerk'),
        'foh_jerk_settled': settled.get('foh_jerk'),
        'baseline_gain': float(zoh_config.baseline_gain
                               if zoh_config.baseline_gain is not None
                               else -zoh_config.K_u / zoh_config.m),
    }

    os.makedirs(args.out, exist_ok=True)
    evaluation.save_json(os.path.join(args.out, 'comparison.json'), report)

    plot_overview(t, truth, zoh_acceleration, foh_acceleration, baseline,
                  os.path.join(args.out, '01-comparison-overview.png'))
    plot_zoom(t, truth, zoh_acceleration, foh_acceleration, baseline, windows,
              os.path.join(args.out, '02-comparison-zoom.png'))
    plot_error(t, truth, zoh_acceleration, foh_acceleration, baseline,
               windows[settled_name], settled,
               os.path.join(args.out, '03-comparison-error.png'))
    if foh_metrics.get('jerk') is not None:
        plot_jerk(t, foh, windows[settled_name],
                  os.path.join(args.out, '04-comparison-jerk.png'))
    plot_asd(t, truth, zoh_acceleration, foh_acceleration, baseline,
             windows[settled_name], data_zoh.dt,
             os.path.join(args.out, '05-comparison-asd.png'))

    lines = [f'# ZOH against FOH prior on {os.path.basename(args.data)}', '']
    lines.append(f"* samples: {len(data_zoh)}")
    lines.append(f"* q_a (ZOH): {zoh_config.q_a:.3e} (m/s^2)^2")
    lines.append(f"* q_j (FOH): {foh_config.q_j:.3e} (m/s^3)^2")
    lines.append(f"* settled window: {settled_name} {windows[settled_name]}")
    lines.extend(['', '| window | ZOH RMSE | FOH RMSE | baseline RMSE | '
                      'FOH/ZOH | gain dB |', '|---|---|---|---|---|---|'])
    for row in rows:
        lines.append('| {} | {:.4e} | {:.4e} | {:.4e} | {:.3f} | {:+.2f} |'.format(
            row['window'], row['zoh']['rmse'], row['foh']['rmse'],
            row['baseline']['rmse'], row.get('foh_over_zoh_rmse', float('nan')),
            row.get('foh_gain_db', float('nan'))))
    with open(os.path.join(args.out, 'comparison.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
