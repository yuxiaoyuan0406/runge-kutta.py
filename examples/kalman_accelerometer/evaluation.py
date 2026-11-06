'''
Error metrics, PID baseline, innovation diagnostics and figures.

The evaluation consumes the estimator history and, when available, the true
signals of the simulation.  True values never go back into the estimator: the
recursion only ever sees ``u[k]``.
'''
import json
import os
from typing import Dict, List, Optional

import numpy as np
import scipy.signal

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def error_metrics(estimate: np.ndarray, truth: np.ndarray,
                  start: int = 0, end: Optional[int] = None) -> Dict[str, float]:
    '''
    Bias, RMSE and the de-meaned error RMS of section 19.2.

    Args:
        estimate: Estimated quantity.
        truth: Reference quantity, same length.
        start: First sample of the evaluation window.
        end: Last sample (exclusive), ``None`` means the whole sequence.

    Returns:
        Dict[str, float]: ``bias``, ``rmse``, ``sigma_err`` and the sample count.
        ``sigma_err`` divides by ``N`` which is the convention of the task sheet.
    '''
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if estimate.shape != truth.shape:
        raise ValueError(f'estimate and truth differ in shape: {estimate.shape} vs {truth.shape}')

    stop = end if end is not None else estimate.size
    error = estimate[start:stop] - truth[start:stop]
    if error.size == 0:
        raise ValueError('the evaluation window is empty')

    bias = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    sigma_err = float(np.sqrt(np.mean((error - bias) ** 2)))
    return {
        'samples': int(error.size),
        'bias': bias,
        'rmse': rmse,
        'sigma_err': sigma_err,
    }


def pid_baseline(u: np.ndarray, K_u: float, mass: float,
                 gain: Optional[float] = None) -> np.ndarray:
    '''
    Static force balance baseline of section 6.2.

    ``a_PID[k] = -(K_u / m) u[k]`` when the user does not supply a calibrated
    gain.  The baseline is only a reference: during dynamic tests its error
    contains mechanical motion, not just readout noise.
    '''
    coefficient = -(K_u / mass) if gain is None else float(gain)
    return coefficient * np.asarray(u, dtype=np.float64)


def nis_statistics(innovation: np.ndarray, innovation_cov: np.ndarray) -> Dict[str, float]:
    '''
    Normalised innovation squared of section 19.2.

    For a matched model the normalised innovation should be zero mean, unit
    variance and uncorrelated across time.
    '''
    innovation = np.asarray(innovation, dtype=np.float64).ravel()
    innovation_cov = np.asarray(innovation_cov, dtype=np.float64).ravel()
    if innovation.size != innovation_cov.size:
        raise ValueError('innovation and its covariance differ in length')

    valid = innovation_cov > 0
    if not np.any(valid):
        return {'samples': 0, 'mean': float('nan'), 'variance': float('nan'),
                'z_mean': float('nan'), 'z_std': float('nan'), 'lag1': float('nan')}

    nis = innovation[valid] ** 2 / innovation_cov[valid]
    z = innovation[valid] / np.sqrt(innovation_cov[valid])

    lag1 = float('nan')
    if z.size > 1:
        lag1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])

    return {
        'samples': int(nis.size),
        'mean': float(np.mean(nis)),
        'variance': float(np.var(nis)),
        'z_mean': float(np.mean(z)),
        'z_std': float(np.std(z)),
        'lag1': lag1,
    }


def spectrum(x: np.ndarray, dt: float, nperseg: Optional[int] = None) -> Dict[str, np.ndarray]:
    '''
    One sided PSD and ASD with Welch averaging.

    Args:
        x: Signal.
        dt: Sampling period in seconds.
        nperseg: Segment length, chosen from the record length when omitted.

    Returns:
        Dict[str, np.ndarray]: ``f`` in Hz, ``psd`` in unit^2/Hz, ``asd`` in
        unit/sqrt(Hz).
    '''
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 8:
        raise ValueError('the signal is too short for a spectrum estimate')

    if nperseg is None:
        nperseg = int(2 ** np.floor(np.log2(max(8, min(x.size, 4096)))))
    nperseg = int(min(nperseg, x.size))
    noverlap = nperseg // 2

    frequencies, psd = scipy.signal.welch(
        x, fs=1. / dt, window='hann', nperseg=nperseg,
        noverlap=noverlap, detrend='constant', scaling='density',
        return_onesided=True)
    return {'f': frequencies, 'psd': psd, 'asd': np.sqrt(psd)}


def window_slice(t: np.ndarray, window: Optional[List[float]]) -> tuple:
    '''
    Convert a ``[start, end]`` window in seconds into index bounds.
    '''
    if not window:
        return 0, int(t.size)
    start, end = float(window[0]), float(window[1])
    mask = np.nonzero((t >= start) & (t <= end))[0]
    if mask.size == 0:
        raise ValueError(f'window [{start}, {end}] s contains no samples')
    return int(mask[0]), int(mask[-1]) + 1


def evaluate(t: np.ndarray, estimate: Dict[str, np.ndarray], data,
             config, model) -> Dict[str, object]:
    '''
    Build the numeric evaluation report of section 6.

    Args:
        t: Sample times.
        estimate: History returned by :meth:`kalman.ClosedLoopKalman.run`.
        data: :class:`data_adapter.SimulationData` with optional truths.
        config: Validated configuration.
        model: Closed loop model, used for the baseline coefficient.

    Returns:
        Dict[str, object]: Metrics per configured window, baseline comparison,
        innovation statistics and metadata.
    '''
    windows = config.test_windows or {'all': [float(t[0]), float(t[-1])]}

    acceleration = np.asarray(estimate['acceleration'], dtype=np.float64)
    baseline = (pid_baseline(data.u, config.K_u, config.m, config.baseline_gain)
                if data.a_true is not None else None)

    report: Dict[str, object] = {
        'windows': {},
        'innovation': nis_statistics(estimate['innovation'],
                                     estimate['innovation_cov']),
        'samples': int(t.size),
        'has_truth': bool(data.a_true is not None),
        'baseline_gain': (float(config.baseline_gain) if config.baseline_gain is not None
                          else float(-config.K_u / config.m)),
    }

    for name, window in windows.items():
        start, end = window_slice(t, window)
        entry: Dict[str, object] = {'window': [float(window[0]), float(window[1])]}

        if data.a_true is not None:
            entry['kalman'] = error_metrics(acceleration, data.a_true, start, end)
            if baseline is not None:
                entry['baseline'] = error_metrics(baseline, data.a_true, start, end)
                kalman_rmse = entry['kalman']['rmse']
                baseline_rmse = entry['baseline']['rmse']
                if kalman_rmse > 0 and baseline_rmse > 0:
                    entry['rmse_ratio_kalman_over_baseline'] = float(kalman_rmse / baseline_rmse)
        report['windows'][name] = entry

    if data.a_true is not None:
        start, end = window_slice(t, windows.get('steady'))
        steady = error_metrics(acceleration, data.a_true, start, end)
        if steady['rmse'] > 0:
            report['kalman_rmse'] = steady['rmse']
        report['steady_state'] = steady

    return report


def compare_spectra(t: np.ndarray, signals: Dict[str, np.ndarray],
                    dt: float) -> Dict[str, Dict[str, np.ndarray]]:
    '''
    Compute the ASD of several signals on one common frequency grid.
    '''
    spectra = {}
    for name, signal in signals.items():
        spectra[name] = spectrum(signal, dt)
    return spectra


def save_json(path: str, payload: Dict[str, object]):
    '''
    Write a JSON report, creating the directory when needed.
    '''
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value):
    '''
    Convert numpy scalars and arrays for ``json.dump``.
    '''
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    return str(value)


def write_markdown(path: str, report: Dict[str, object], title: str = 'Kalman evaluation'):
    '''
    Write a short human readable summary next to the JSON report.
    '''
    lines = [f'# {title}', '']
    lines.append(f"* samples: {report.get('samples')}")
    lines.append(f"* has ground truth: {report.get('has_truth')}")
    lines.append(f"* baseline gain: {report.get('baseline_gain')}")

    innovation = report.get('innovation') or {}
    lines.extend(['', '## Innovation', '',
                  f"* NIS mean: {innovation.get('mean')}",
                  f"* NIS variance: {innovation.get('variance')}",
                  f"* z mean: {innovation.get('z_mean')}",
                  f"* z std: {innovation.get('z_std')}",
                  f"* z lag-1 correlation: {innovation.get('lag1')}"])

    for name, entry in (report.get('windows') or {}).items():
        lines.extend(['', f'## Window `{name}` {entry.get("window")}'])
        for method in ('kalman', 'baseline'):
            metrics = entry.get(method)
            if not metrics:
                continue
            lines.append(
                f"* {method}: bias {metrics['bias']:.6e}, "
                f"RMSE {metrics['rmse']:.6e}, sigma_err {metrics['sigma_err']:.6e}")

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def plot_time(t: np.ndarray, series: Dict[str, np.ndarray], path: str,
              ylabel: str = 'Acceleration [m/s$^2$]', title: str = 'Acceleration estimate'):
    '''
    Plot several time series and save the figure.
    '''
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, values in series.items():
        axis.plot(t, values, label=label, linewidth=1.)
    axis.set_xlabel('Time [s]')
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, path)


def plot_spectrum(spectra: Dict[str, Dict[str, np.ndarray]], path: str,
                  title: str = 'Acceleration ASD'):
    '''
    Plot the amplitude spectral density of several signals.
    '''
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, item in spectra.items():
        axis.loglog(item['f'][1:], item['asd'][1:], label=label, linewidth=1.)
    axis.set_xlabel('Frequency [Hz]')
    axis.set_ylabel('ASD [(m/s$^2$)/$\\sqrt{Hz}$]')
    axis.set_title(title)
    axis.grid(True, which='both')
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, path)


def plot_diagnostics(t: np.ndarray, estimate: Dict[str, np.ndarray], path: str):
    '''
    Plot the innovation and the posterior standard deviations of the states.
    '''
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(t, estimate['innovation'], linewidth=1.)
    axes[0].set_ylabel('Innovation')
    axes[0].grid(True)

    variances = np.asarray(estimate['posterior_variance'])
    for column, label in enumerate(('x', 'v', 'a')):
        axes[1].semilogy(t, np.sqrt(np.maximum(variances[:, column], 1e-300)),
                         label=f'$\\sigma_{{{label}}}$', linewidth=1.)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Posterior std')
    axes[1].grid(True, which='both')
    axes[1].legend()

    figure.tight_layout()
    _save_figure(figure, path)


def _save_figure(figure, path: str):
    '''
    Save and close a figure, creating the directory when needed.
    '''
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def summarize(report: Dict[str, object]) -> str:
    '''
    One line summary of the steady state error ratio.
    '''
    windows = report.get('windows') or {}
    steady = windows.get('steady') or {}
    if 'kalman' in steady and 'baseline' in steady:
        return (f"RMSE Kalman {steady['kalman']['rmse']:.4e} vs "
                f"PID baseline {steady['baseline']['rmse']:.4e}")
    if 'kalman' in steady:
        return f"RMSE Kalman {steady['kalman']['rmse']:.4e}"
    return 'no truth available, nothing to compare'
