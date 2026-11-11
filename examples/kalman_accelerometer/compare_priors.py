'''
Comparison of the two acceleration priors on one measurement sequence.

The script runs the ZOH and the FOH estimator on the *same* samples with the
same units and the same evaluation windows, and adds three classical
references on exactly those samples:

* the phase 1 static force balance estimate ``a = -K_u u / m``;
* ``util.baseline_filters.RCFilter`` on that estimate;
* ``util.baseline_filters.ButterworthFilter`` on that estimate.

The two filters only accept an acceleration in m/s^2, so the calibration
``-K_u/m`` (and the code to control unit scale of the stored PID column) is
applied once, before filtering; that conversion is recorded in the output.

Outputs:

``comparison.json``
    every number of the comparison, including the filter coefficients, the
    innovation statistics and the Jerk diagnostics;
``comparison.md``
    a short table view;
``01-comparison-overview.png``
    true acceleration, both estimates and the two filtered baselines;
``02-comparison-zoom.png``
    one panel per configured window;
``03-comparison-error.png``
    the error signals on the settled window;
``04-comparison-jerk.png``
    the FOH Jerk estimate with its posterior band;
``05-comparison-asd.png``
    amplitude spectral density of the error signals.

Usage::

    python examples/kalman_accelerometer/compare_priors.py \\
        --zoh-config examples/kalman_accelerometer/configs/xiong-2018.json \\
        --foh-config examples/kalman_accelerometer/configs/xiong-2018-foh.json \\
        --data data/20261026-unit_step_1 --rc-fc 1000 --bw-fc 1000 \\
        -o data/compare-unit-step
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

from examples.kalman_accelerometer import config as config_module
from examples.kalman_accelerometer import data_adapter, evaluation
from examples.kalman_accelerometer.kalman import ClosedLoopKalman
from util.baseline_filters import ButterworthFilter, RCFilter

STYLE = {
    'truth': {'color': 'black', 'linestyle': '-', 'linewidth': 1.4},
    'zoh': {'color': 'tab:blue', 'linestyle': '--', 'linewidth': 1.0},
    'foh': {'color': 'tab:green', 'linestyle': '-', 'linewidth': 1.0},
    'rc': {'color': 'tab:orange', 'linestyle': '-.', 'linewidth': 1.0},
    'bw': {'color': 'tab:purple', 'linestyle': (0, (1, 1)), 'linewidth': 1.2},
    'pid': {'color': 'tab:red', 'linestyle': ':', 'linewidth': 1.0},
}
'''Line styles, distinguishable in greyscale as well as in colour.'''


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
    parser.add_argument('--tone-hz', type=float, default=0.0,
                        help='Excitation frequency used for the tone/residual split, 0 disables it.')
    parser.add_argument('--rc-fc', type=float, default=1000.0,
                        help='RC low pass cutoff in Hz, default 1000.')
    parser.add_argument('--bw-fc', type=float, default=1000.0,
                        help='Butterworth low pass cutoff in Hz, default 1000.')
    parser.add_argument('--windows', type=str, default='',
                        help='Override test_windows, e.g. "steady=0.01,0.99;detail=0.04,0.048".')
    parser.add_argument('--dpi', type=int, default=200, help='Figure resolution.')
    return parser


def parse_windows(text: str) -> dict:
    '''
    Parse ``name=start,end`` entries separated by semicolons.
    '''
    windows = {}
    for item in text.split(';'):
        item = item.strip()
        if not item:
            continue
        name, bounds = item.split('=', 1)
        start, end = bounds.split(',', 1)
        windows[name.strip()] = [float(start), float(end)]
    return windows


def load_pair(args) -> tuple:
    '''
    Load both configurations and enforce that they describe the same record.
    '''
    zoh = config_module.load_config(args.zoh_config)
    foh = config_module.load_config(args.foh_config)
    zoh.data_path = args.data
    foh.data_path = args.data
    if getattr(args, 'windows', ''):
        windows = parse_windows(args.windows)
        zoh.test_windows = dict(windows)
        foh.test_windows = dict(windows)

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


def pid_acceleration(config, u_signal) -> tuple:
    '''
    Build the acceleration sequence the baseline filters expect.

    The stored PID column is a fixed point code.  ``data_adapter`` applies
    ``units['u']`` to it, so the signal handed to the Kalman estimator is in
    control units (volts for the device files) and its model carries ``K_u``,
    ``K_s`` and ``R`` in those units.  The filters do not model anything: they
    need an acceleration in m/s^2, therefore the static calibration is applied
    here, once, before filtering::

        a_raw[k] = (-K_u / m) * u[k]

    With ``units['u']`` in V/code and ``-K_u/m`` in (m/s^2)/V the product is
    the (m/s^2)/code scale.  Filtering commutes with this constant scale, so
    applying it before or after the filter gives the same waveform; applying
    it once is what keeps the numbers interpretable.

    Args:
        config: Validated configuration (uses ``units``, ``K_u``, ``m`` and
            ``baseline_gain``).
        u_signal: PID output sequence in control units, as returned by
            :func:`data_adapter.load_data`.

    Returns:
        tuple: ``(a_raw, info)`` where ``info`` records the two scale factors.
    '''
    control_units_per_code = float((config.units or {}).get('u', 1.0))
    acceleration_per_control_unit = (float(config.baseline_gain)
                                     if config.baseline_gain is not None
                                     else -config.K_u / config.m)
    a_raw = acceleration_per_control_unit * np.asarray(u_signal, dtype=np.float64)
    info = {
        'control_units_per_code': control_units_per_code,
        'acceleration_per_control_unit': acceleration_per_control_unit,
        'acceleration_per_code': control_units_per_code * acceleration_per_control_unit,
        'note': ('The Kalman estimator consumes u[k] in control units (the code '
                 'scale of units.u is applied by data_adapter); the filters need '
                 'a_raw in m/s^2, so -K_u/m is applied once before filtering.'),
    }
    return a_raw, info


def _trapezoid(values, grid) -> float:
    '''
    Trapezoidal integral without the numpy version dependent ``trapz`` name.
    '''
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))


def filter_baseline(config, a_raw, rc_fc, bw_fc) -> tuple:
    '''
    Run both baseline filters on the converted acceleration.

    Args:
        config: Validated configuration, used for ``fs``.
        a_raw: Acceleration in m/s^2, the common input of both filters.
        rc_fc: RC cutoff in Hz.
        bw_fc: Butterworth cutoff in Hz.

    Returns:
        tuple: ``(a_rc, a_bw, metadata)``.  The metadata holds the coefficients,
        the response at 125 Hz and 500 Hz, and the equivalent noise bandwidth.
    '''
    rc = RCFilter(fs=config.fs, fc=rc_fc)
    bw = ButterworthFilter(fs=config.fs, fc=bw_fc)
    a_rc = rc.process_array(a_raw)
    a_bw = bw.process_array(a_raw)

    frequencies = np.linspace(0., config.fs / 2., 20001)
    grid = np.array([0., 125., 500.])

    def describe(filt, label):
        response = np.atleast_1d(filt.frequency_response(frequencies))
        gain = np.abs(response) ** 2
        enbw = _trapezoid(gain, frequencies) / max(float(gain[0]), 1e-300)
        probes = np.atleast_1d(filt.frequency_response(grid))
        return {
            'label': label,
            'fc': float(filt.fc),
            'coefficients': {key: float(value)
                             for key, value in filt.coefficients.items()},
            'dc_gain': float(filt.dc_gain()),
            'equivalent_noise_bandwidth_hz': float(enbw),
            'white_noise_rms_factor': float(np.sqrt(2. * enbw / config.fs)),
            'gain_db_125hz': float(20. * np.log10(abs(probes[1]))),
            'phase_deg_125hz': float(np.degrees(np.angle(probes[1]))),
            'gain_db_500hz': float(20. * np.log10(abs(probes[2]))),
            'phase_deg_500hz': float(np.degrees(np.angle(probes[2]))),
        }

    metadata = {
        'rc': describe(rc, 'RC'),
        'bw': describe(bw, 'Butterworth'),
    }
    return a_rc, a_bw, metadata


def window_row(name: str, window, zoh: dict, foh: dict, baseline: dict,
               rc: np.ndarray, bw: np.ndarray, data, foh_metrics: dict,
               tone_hz: float = 0.) -> dict:
    '''
    Collect the estimator and baseline numbers of one evaluation window.

    When ``tone_hz`` is positive the error of every method is also split into a
    coherent component at that frequency and the residual, which explains why a
    plain measurement can beat a filter on a strong in-band tone.
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
        'rc': evaluation.error_metrics(rc, truth, start, end),
        'bw': evaluation.error_metrics(bw, truth, start, end),
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
    if tone_hz and tone_hz > 0:
        time = np.asarray(data.t, dtype=np.float64)[start:end]
        truth_window = np.asarray(truth, dtype=np.float64)[start:end]
        row['tone'] = {}
        for method, estimate in (('zoh', np.asarray(zoh['acceleration'], dtype=np.float64)[start:end]),
                                 ('foh', np.asarray(foh['acceleration'], dtype=np.float64)[start:end]),
                                 ('baseline', np.asarray(baseline, dtype=np.float64)[start:end]),
                                 ('rc', np.asarray(rc, dtype=np.float64)[start:end]),
                                 ('bw', np.asarray(bw, dtype=np.float64)[start:end])):
            row['tone'][method] = evaluation.tone_decomposition(
                estimate - truth_window, time, tone_hz)
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


def plot_overview(t, truth, zoh, foh, rc, bw, path):
    '''
    Whole record: truth, both estimates and the two filtered baselines.
    '''
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(t, truth, label='True acceleration', **STYLE['truth'])
    axis.plot(t, zoh, label='ZOH estimate', **STYLE['zoh'])
    axis.plot(t, foh, label='FOH estimate', **STYLE['foh'])
    axis.plot(t, rc, label='RC baseline', **STYLE['rc'])
    axis.plot(t, bw, label='Butterworth baseline', **STYLE['bw'])
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Acceleration [m/s$^2$]')
    axis.set_title('Acceleration estimate: ZOH, FOH and the filtered baselines')
    axis.grid(True)
    axis.legend(loc='upper left')
    figure.tight_layout()
    save_figure(figure, path)


def plot_zoom(t, truth, zoh, foh, rc, bw, windows, path):
    '''
    One panel per evaluation window.
    '''
    names = list(windows)
    figure, axes = plt.subplots(len(names), 1, figsize=(9, 3.2 * len(names)),
                                squeeze=False)
    for axis, name in zip(axes.ravel(), names):
        start, end = windows[name]
        mask = (t >= start) & (t <= end)
        axis.plot(t[mask], truth[mask], label='Truth', **STYLE['truth'])
        axis.plot(t[mask], zoh[mask], label='ZOH', **STYLE['zoh'])
        axis.plot(t[mask], foh[mask], label='FOH', **STYLE['foh'])
        axis.plot(t[mask], rc[mask], label='RC', **STYLE['rc'])
        axis.plot(t[mask], bw[mask], label='Butterworth', **STYLE['bw'])
        axis.set_xlim(start, end)
        axis.set_title(f'{name} window [{start * 1e3:.2f}, {end * 1e3:.2f}] ms',
                       fontsize=10)
        axis.set_ylabel('a [m/s$^2$]')
        axis.grid(True)
        axis.legend(loc='upper right', fontsize=8)
    axes.ravel()[-1].set_xlabel('Time [s]')
    figure.tight_layout()
    save_figure(figure, path)


def plot_error(t, truth, zoh, foh, baseline, rc, bw, window, rows, path):
    '''
    Error signals of every method on the settled window.
    '''
    start, end = window
    mask = (t >= start) & (t <= end)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(t[mask], (foh - truth)[mask], label='FOH error', **STYLE['foh'])
    axis.plot(t[mask], (zoh - truth)[mask], label='ZOH error', **STYLE['zoh'])
    axis.plot(t[mask], (rc - truth)[mask], label='RC error', **STYLE['rc'])
    axis.plot(t[mask], (bw - truth)[mask], label='Butterworth error', **STYLE['bw'])
    axis.plot(t[mask], (baseline - truth)[mask], label='PID raw error',
              **STYLE['pid'])
    axis.axhline(0., color='black', linewidth=0.8)
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Estimate - truth [m/s$^2$]')
    axis.set_title('Estimation error in the settled window')
    axis.grid(True)
    text = ('RMS  FOH {:>10.3e}\n      ZOH {:>10.3e}\n      RC  {:>10.3e}\n'
            '      BW  {:>10.3e}\n      PID {:>10.3e}').format(
        rows['foh']['rmse'], rows['zoh']['rmse'], rows['rc']['rmse'],
        rows['bw']['rmse'], rows['baseline']['rmse'])
    axis.text(0.02, 0.03, text, transform=axis.transAxes, fontsize=8,
              verticalalignment='bottom', family='monospace',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    axis.legend(loc='upper right', fontsize=8)
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
    axis.plot(t, jerk, label='$\\hat j$', **STYLE['foh'])
    axis.fill_between(t, jerk - 2. * deviation, jerk + 2. * deviation,
                      color=STYLE['foh']['color'], alpha=0.2, label='$\\pm 2\\sigma$')
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


def plot_asd(t, truth, zoh, foh, baseline, rc, bw, window, dt, path):
    '''
    Amplitude spectral density of every error signal on the window.
    '''
    start, end = evaluation.window_slice(t, window)
    spectra = {
        'FOH error': (foh[start:end] - truth[start:end], 'foh'),
        'ZOH error': (zoh[start:end] - truth[start:end], 'zoh'),
        'RC error': (rc[start:end] - truth[start:end], 'rc'),
        'Butterworth error': (bw[start:end] - truth[start:end], 'bw'),
        'PID raw error': (baseline[start:end] - truth[start:end], 'pid'),
    }
    figure, axis = plt.subplots(figsize=(9, 5))
    for name, (signal, key) in spectra.items():
        estimate = evaluation.spectrum(signal, dt)
        axis.loglog(estimate['f'][1:], estimate['asd'][1:],
                    label=f"{name} (RMS {np.std(signal):.2e})", **STYLE[key])
    axis.set_xlabel('Frequency [Hz]')
    axis.set_ylabel('ASD [(m/s$^2$)/$\\sqrt{Hz}$]')
    axis.set_title('Amplitude spectral density of the estimation error')
    axis.grid(True, which='both')
    axis.legend(loc='upper right', fontsize=8)
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
    a_raw, conversion = pid_acceleration(zoh_config, data_zoh.u)
    a_rc, a_bw, filters = filter_baseline(zoh_config, a_raw, args.rc_fc, args.bw_fc)

    windows = zoh_config.test_windows or {'all': [float(t[0]), float(t[-1])]}
    rows = [window_row(name, window, zoh, foh, a_raw, a_rc, a_bw, data_zoh,
                       foh_metrics, args.tone_hz)
            for name, window in windows.items()]

    settled_name = 'steady' if 'steady' in windows else list(windows)[-1]
    settled = next(row for row in rows if row['window'] == settled_name)

    # Independent check that the conversion really lands in m/s^2: on a record
    # whose settled window has a non zero mean, the raw baseline must match it.
    truth_level = float(np.mean(truth[evaluation.window_slice(
        t, windows[settled_name])[0]:evaluation.window_slice(t, windows[settled_name])[1]]))
    if abs(truth_level) > 0.1:
        start, end = evaluation.window_slice(t, windows[settled_name])
        conversion['settled_level_check'] = {
            'truth_mean': truth_level,
            'raw_baseline_mean': float(np.mean(a_raw[start:end])),
            'ratio': float(np.mean(a_raw[start:end]) / truth_level),
        }

    report = {
        'data': data_zoh.summary(),
        'zoh_config': zoh_config.summary(),
        'foh_config': foh_config.summary(),
        'zoh_model': zoh_model.as_dict(),
        'foh_model': foh_model.as_dict(),
        'zoh_observability': zoh_model.observability_report().as_dict(),
        'foh_observability': foh_model.observability_report().as_dict(),
        'baselines': {
            'conversion': conversion,
            'pid_state_balance': {
                'acceleration_per_control_unit': conversion['acceleration_per_control_unit'],
            },
            'rc': filters['rc'],
            'bw': filters['bw'],
        },
        'windows': rows,
        'settled_window': settled_name,
        'tone_hz': float(args.tone_hz),
        'whole_record': {
            'zoh': evaluation.error_metrics(zoh_acceleration, truth),
            'foh': evaluation.error_metrics(foh_acceleration, truth),
            'baseline': evaluation.error_metrics(a_raw, truth),
            'rc': evaluation.error_metrics(a_rc, truth),
            'bw': evaluation.error_metrics(a_bw, truth),
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

    plot_overview(t, truth, zoh_acceleration, foh_acceleration, a_rc, a_bw,
                  os.path.join(args.out, '01-comparison-overview.png'))
    plot_zoom(t, truth, zoh_acceleration, foh_acceleration, a_rc, a_bw, windows,
              os.path.join(args.out, '02-comparison-zoom.png'))
    plot_error(t, truth, zoh_acceleration, foh_acceleration, a_raw, a_rc, a_bw,
               windows[settled_name], settled,
               os.path.join(args.out, '03-comparison-error.png'))
    if foh_metrics.get('jerk') is not None:
        plot_jerk(t, foh, windows[settled_name],
                  os.path.join(args.out, '04-comparison-jerk.png'))
    plot_asd(t, truth, zoh_acceleration, foh_acceleration, a_raw, a_rc, a_bw,
             windows[settled_name], data_zoh.dt,
             os.path.join(args.out, '05-comparison-asd.png'))

    lines = [f'# ZOH against FOH prior on {os.path.basename(args.data)}', '']
    lines.append(f"* samples: {len(data_zoh)}")
    lines.append(f"* q_a (ZOH): {zoh_config.q_a:.3e} (m/s^2)^2")
    lines.append(f"* q_j (FOH): {foh_config.q_j:.3e} (m/s^3)^2")
    lines.append(f"* RC cutoff: {filters['rc']['fc']:.0f} Hz, "
                 f"Butterworth cutoff: {filters['bw']['fc']:.0f} Hz")
    lines.append("* acceleration conversion: "
                 f"a = {conversion['acceleration_per_control_unit']:.6g} * u, "
                 f"units.u = {conversion['control_units_per_code']:.6g} per code")
    lines.append(f"* settled window: {settled_name} {windows[settled_name]}")
    lines.extend(['', '| window | ZOH | FOH | PID raw | RC | Butterworth | FOH/ZOH |',
                  '|---|---|---|---|---|---|---|'])
    for row in rows:
        lines.append('| {} | {:.4e} | {:.4e} | {:.4e} | {:.4e} | {:.4e} | {:.3f} |'.format(
            row['window'], row['zoh']['rmse'], row['foh']['rmse'],
            row['baseline']['rmse'], row['rc']['rmse'], row['bw']['rmse'],
            row.get('foh_over_zoh_rmse', float('nan'))))
    with open(os.path.join(args.out, 'comparison.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
