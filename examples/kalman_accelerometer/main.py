'''
Command line entry point of the closed loop Kalman example.

Run from the project root::

    python examples/kalman_accelerometer/main.py params --config <file>
    python examples/kalman_accelerometer/main.py check  --config <file>
    python examples/kalman_accelerometer/main.py run    --config <file> --data <dir>
    python examples/kalman_accelerometer/main.py scan   --config <file> --data <dir>

``params`` only reports which section 9 entries still need a value; it never
fills a number in.  ``check`` runs the algebraic verification of section 19.1.
'''
import argparse
import json
import logging
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.kalman_accelerometer import config as config_module
from examples.kalman_accelerometer import data_adapter, evaluation, verification
from examples.kalman_accelerometer.kalman import ClosedLoopKalman
from examples.kalman_accelerometer.model import ClosedLoopModel


def get_logger(level: int = logging.INFO) -> logging.Logger:
    '''
    Create the console logger used by this script.
    '''
    logger = logging.getLogger('kalman-accelerometer')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


def argue_parser() -> argparse.ArgumentParser:
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Closed loop MEMS accelerometer Kalman filter.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print extra information.')

    subparsers = parser.add_subparsers(dest='command', required=True)

    params = subparsers.add_parser('params', help='Report the parameters still missing.')
    params.add_argument('--config', type=str, required=True,
                        help='Configuration file in JSON format.')

    check = subparsers.add_parser('check', help='Run the algebraic verification.')
    check.add_argument('--config', type=str, required=True,
                       help='Configuration file in JSON format.')
    check.add_argument('--out', type=str, default='',
                       help='Optional path of the JSON verification report.')

    run = subparsers.add_parser('run', help='Filter one measurement sequence.')
    run.add_argument('--config', type=str, required=True,
                     help='Configuration file in JSON format.')
    run.add_argument('--data', type=str, default='',
                     help='Override data_path of the configuration.')
    run.add_argument('--out', type=str, default='',
                     help='Override output_dir of the configuration.')

    scan = subparsers.add_parser('scan', help='Repeat run for every q_a of q_a_grid.')
    scan.add_argument('--config', type=str, required=True,
                      help='Configuration file in JSON format.')
    scan.add_argument('--data', type=str, default='',
                      help='Override data_path of the configuration.')
    scan.add_argument('--out', type=str, default='',
                      help='Override output_dir of the configuration.')

    reference = subparsers.add_parser(
        'reference',
        help='Verify, or explicitly capture, the phase 1 ZOH reference record.')
    reference.add_argument('--capture', action='store_true', default=False,
                           help='Overwrite the reference with the current ZOH '
                                'implementation.  Only the phase 1 program is '
                                'allowed to do this.')
    reference.add_argument('--base-dir', type=str, default='',
                           help='Directory the case paths are relative to, '
                                'defaults to the repository root.')
    reference.add_argument('--reference-dir', type=str, default='',
                           help='Directory holding the reference record.')
    reference.add_argument('--out', type=str, default='',
                           help='Optional path of the JSON report.')

    return parser


def load(config_path: str, data_path: str = '', out_dir: str = ''):
    '''
    Load a configuration and apply the command line overrides.
    '''
    config = config_module.load_config(config_path)
    if data_path:
        config.data_path = data_path
    if out_dir:
        config.output_dir = out_dir
    return config


def command_params(args) -> int:
    '''
    Report the completeness of the section 9 parameter block.
    '''
    report = config_module.missing_parameter_report(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report['ok']:
        logger.info('All section 9 parameters are present.')
        return 0
    logger.warning('%d parameters still need a value.', len(report['missing']))
    return 0


def command_check(args) -> int:
    '''
    Run the algebraic verification of section 19.1.
    '''
    config = load(args.config)
    logger.info('Running the algebraic verification.')
    results = verification.run_all(config)

    text = json.dumps(results, indent=2, ensure_ascii=False,
                      default=evaluation._json_default)
    if args.out:
        evaluation.save_json(args.out, results)
        logger.info('Verification report written to `%s`.', args.out)
    else:
        print(text)

    if not results['passed']:
        logger.error('The algebraic verification failed.')
        return 1
    logger.info('The algebraic verification passed.')
    return 0


def filter_sequence(config) -> tuple:
    '''
    Run the estimator on the configured data and return its history.
    '''
    data = data_adapter.load_data(config)
    logger.info('Loaded %d samples from `%s`.', len(data), data.source)
    for message in data.warnings:
        logger.warning(message)

    estimator = ClosedLoopKalman.initialize(config)
    history = estimator.run(data.u)

    finite = np.all(np.isfinite(history['state']))
    if not finite:
        raise RuntimeError('the estimate contains non finite entries')
    return data, history, estimator


def command_run(args) -> int:
    '''
    Filter the data, evaluate the estimate and write the report.
    '''
    config = load(args.config, args.data, args.out)
    data, history, estimator = filter_sequence(config)
    model = estimator.model

    report = model.observability_report()
    if not report.detectable:
        logger.warning('The closed loop pair is not detectable, the estimate may '
                       'not converge for every initial condition.')
    for note in report.notes:
        logger.warning(note)

    metrics = evaluation.evaluate(data.t, history, data, config, model)

    payload = {
        'config': config.summary(),
        'data': data.summary(),
        'model': model.as_dict(),
        'observability': report.as_dict(),
        'metrics': metrics,
    }
    output_dir = config.output_dir
    evaluation.save_json(os.path.join(output_dir, 'report.json'), payload)
    evaluation.write_markdown(os.path.join(output_dir, 'report.md'), metrics,
                              title=f'Closed loop Kalman evaluation ({config.prior_model})')

    fields = {
        't': data.t,
        'u': data.u,
        'a_hat': history['acceleration'],
        'x_hat': history['displacement'],
        'v_hat': history['velocity'],
        'state': history['state'],
        'innovation': history['innovation'],
        'innovation_cov': history['innovation_cov'],
        'gain': history['gain'],
    }
    if model.has_jerk:
        # Section 5: the phase 1 fields keep their names, units and index; the
        # Jerk diagnostics are appended.
        fields.update({
            'jerk_hat': history['jerk'],
            'jerk_variance': history['jerk_variance'],
            'jerk_accel_correlation': history['jerk_accel_correlation'],
            'posterior_variance': history['posterior_variance'],
            'prior_variance': history['prior_variance'],
        })
    np.savez(os.path.join(output_dir, 'estimate.npz'), **fields)

    series = {'Kalman estimate': history['acceleration']}
    if data.a_true is not None:
        series['True acceleration'] = data.a_true
        series['PID baseline'] = evaluation.pid_baseline(
            data.u, config.K_u, config.m, config.baseline_gain)
    evaluation.plot_time(data.t, series, os.path.join(output_dir, 'acceleration-time.png'))
    evaluation.plot_diagnostics(data.t, history, os.path.join(output_dir, 'diagnostics.png'))

    if data.a_true is not None:
        spectra = evaluation.compare_spectra(
            data.t,
            {'Kalman error': history['acceleration'] - data.a_true,
             'Truth': data.a_true},
            data.dt)
        evaluation.plot_spectrum(spectra, os.path.join(output_dir, 'acceleration-asd.png'))

    logger.info('%s', evaluation.summarize(metrics))
    logger.info('Results written to `%s`.', output_dir)
    return 0


def command_scan(args) -> int:
    '''
    Repeat the run for every q_a of ``q_a_grid`` without touching the data.

    The sweep belongs to the ZOH path of phase 1.  ``q_j`` is a different
    physical quantity and section 7 of the V2.1 task sheet forbids an automatic
    ``q_j`` scan, so an FOH configuration is rejected instead of being mapped
    onto ``q_a``.
    '''
    config = load(args.config, args.data, args.out)
    if config.prior_model == 'FOH':
        logger.error('scan sweeps q_a, which is the ZOH parameter.  The FOH path '
                     'uses q_j and section 7 of the V2.1 task sheet forbids an '
                     'automatic q_j sweep; edit the configuration by hand.')
        return 2

    base_model = ClosedLoopModel(config)
    data = data_adapter.load_data(config)

    logger.info('Scanning %d values of q_a.', len(config.q_a_grid))
    results = []
    for q_a in config.q_a_grid:
        model = base_model.with_q_a(q_a)
        estimator = ClosedLoopKalman(model, config.prior_mean,
                                     config.prior_covariance, config.tolerances)
        history = estimator.run(data.u)
        metrics = evaluation.evaluate(data.t, history, data, config, model)
        entry = {'q_a': float(q_a), 'metrics': metrics}
        results.append(entry)
        logger.info('q_a = %.3e: %s', q_a, evaluation.summarize(metrics))

    output_dir = config.output_dir
    payload = {'config': config.summary(), 'data': data.summary(), 'scan': results}
    evaluation.save_json(os.path.join(output_dir, 'q_a-scan.json'), payload)

    if data.a_true is not None and config.q_a_grid:
        _plot_scan(results, os.path.join(output_dir, 'q_a-scan.png'))

    logger.info('Scan written to `%s`.', output_dir)
    return 0


def _plot_scan(results: list, path: str):
    '''
    Plot RMSE against q_a for the steady state window when truth is available.
    '''
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    q_values = []
    rmse_values = []
    for entry in results:
        steady = (entry['metrics']['windows'].get('steady') or {}).get('kalman')
        if not steady:
            continue
        q_values.append(entry['q_a'])
        rmse_values.append(steady['rmse'])

    if not q_values:
        logger.warning('No steady state window, the q_a scan figure is skipped.')
        return

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.loglog(q_values, rmse_values, 'o-')
    axis.set_xlabel('$q_a$ [(m/s$^2$)$^2$]')
    axis.set_ylabel('Steady state RMSE [(m/s$^2$)]')
    axis.set_title('Accuracy versus process noise')
    axis.grid(True, which='both')
    figure.tight_layout()
    evaluation._save_figure(figure, path)


def command_reference(args) -> int:
    '''
    Capture or verify the phase 1 ZOH reference record of section 5.
    '''
    from examples.kalman_accelerometer import regression

    base_dir = args.base_dir or regression.ROOT
    reference_dir = args.reference_dir or regression.REFERENCE_DIR

    if args.capture:
        manifest = regression.capture(base_dir=base_dir, directory=reference_dir)
        logger.info('Reference record captured from the current implementation: '
                    '%d cases, %s.', len(manifest['cases']), manifest['file'])
        return 0

    arrays, manifest = regression.load(reference_dir)
    report = regression.compare(arrays, manifest, base_dir=base_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False,
                      default=evaluation._json_default)
    if args.out:
        evaluation.save_json(args.out, report)
        logger.info('Regression report written to `%s`.', args.out)
    else:
        print(text)
    logger.info('%s', regression.summarize(report))
    return 0 if report['passed'] else 1


def main(argv=None) -> int:
    '''
    Dispatch the selected subcommand.
    '''
    parser = argue_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    handlers = {
        'params': command_params,
        'check': command_check,
        'run': command_run,
        'scan': command_scan,
        'reference': command_reference,
    }
    return handlers[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
