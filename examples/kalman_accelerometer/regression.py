'''
Phase 1 (ZOH) reference record for the compatibility regression of section 5.

The FOH upgrade of phase 2 must not change the five state ZOH path.  This
module stores the phase 1 output of four simulation records -- estimated
states, covariances, innovations, gains and the evaluation metrics -- so the
upgraded code can be checked against it without keeping a copy of the old
program next to the new one.

The record was captured with the phase 1 implementation, that is *before* the
FOH work.  Re-capturing it with the new code would defeat its purpose, which is
why :func:`capture` is a separate, explicit operation rather than something the
test suite calls.

Declared numerical tolerance: the ZOH arithmetic path is untouched by the FOH
work, so a fresh run on the same machine reproduces the record bit for bit and
the observed difference is zero.  The tolerance below only absorbs LAPACK and
build differences if the record is compared on another machine.
'''
import hashlib
import json
import os

import numpy as np

from . import config as config_module
from . import data_adapter, evaluation
from .kalman import ClosedLoopKalman

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
'''Repository root, used to resolve the relative data paths of :data:`CASES`.'''

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
'''Directory of this package, used to resolve the relative configuration paths.'''

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'tests', 'fixtures')
REFERENCE_PATH = os.path.join(REFERENCE_DIR, 'zoh-reference.npz')
MANIFEST_PATH = os.path.join(REFERENCE_DIR, 'zoh-reference.json')

TOLERANCE = {
    'array_relative': 1e-12,
    'metric_relative': 1e-12,
    'absolute_floor': 1e-300,
    'note': ('Relative difference of the maximum absolute deviation over the '
             'maximum absolute reference value.  1e-12 is about 4500 machine '
             'epsilons, far above BLAS rounding and far below any change that '
             'would alter the estimate.'),
}

CASES = (
    {'name': 'unit_step_1', 'config': 'configs/xiong-2018.json',
     'data': 'data/20261026-unit_step_1'},
    {'name': '125hz_sine_1', 'config': 'configs/xiong-2018-sine.json',
     'data': 'data/20261026-125hz-sine-1'},
    {'name': '125hz_sin_0p1', 'config': 'configs/xiong-2018-sine-0p1.json',
     'data': 'data/20261027-125hz-sin-0.1'},
    {'name': '12p5hz_sine_0p1', 'config': 'configs/xiong-2018-sine-12p5.json',
     'data': 'data/20261027-12.5hz-sine-0.1'},
)
'''The four phase 1 records, with the configuration used for each of them.'''

ARRAYS = ('acceleration', 'displacement', 'velocity', 'innovation',
          'innovation_cov', 'state', 'posterior_variance', 'prior_variance',
          'gain', 'next_gain')
'''Per sample arrays compared by the regression.  ``index`` is implied.'''


def _relative_difference(reference: np.ndarray, actual: np.ndarray) -> float:
    '''
    Maximum absolute deviation divided by the maximum absolute reference value.
    '''
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f'shape mismatch: {reference.shape} vs {actual.shape}')
    if reference.size == 0:
        return 0.
    deviation = float(np.max(np.abs(actual - reference)))
    scale = float(np.max(np.abs(reference)))
    floor = float(TOLERANCE['absolute_floor'])
    return deviation / max(scale, floor)


def _compare_numbers(reference, actual, path: str, tolerance: float,
                     rows: list):
    '''
    Recursively compare the numbers of two JSON friendly structures.

    Keys that only exist on one side are reported, except the free text
    ``notes`` list of the observability report which is documentation.
    '''
    if isinstance(reference, dict):
        if not isinstance(actual, dict):
            rows.append({'path': path, 'status': 'type mismatch'})
            return
        for key in reference:
            if key == 'notes':
                continue
            if key not in actual:
                rows.append({'path': f'{path}.{key}', 'status': 'missing'})
                continue
            _compare_numbers(reference[key], actual[key], f'{path}.{key}',
                             tolerance, rows)
        return
    if isinstance(reference, (list, tuple)):
        actual_list = list(actual) if isinstance(actual, (list, tuple)) else None
        if actual_list is None or len(actual_list) != len(reference):
            rows.append({'path': path, 'status': 'shape mismatch'})
            return
        for index, (left, right) in enumerate(zip(reference, actual_list)):
            _compare_numbers(left, right, f'{path}[{index}]', tolerance, rows)
        return
    if reference is None and actual is None:
        return
    if isinstance(reference, str):
        if reference != actual:
            rows.append({'path': path, 'status': 'text differs',
                         'reference': reference, 'actual': actual})
        return
    try:
        left = float(reference)
        right = float(actual)
    except (TypeError, ValueError):
        rows.append({'path': path, 'status': 'non numeric'})
        return
    deviation = abs(right - left)
    scale = max(abs(left), abs(right), float(TOLERANCE['absolute_floor']))
    relative = deviation / scale
    rows.append({'path': path, 'status': 'ok' if relative <= tolerance else 'differs',
                 'relative': relative})


def run_case(case: dict, base_dir: str = ROOT) -> dict:
    '''
    Re-run one ZOH case with the current code.

    Args:
        case: One entry of :data:`CASES`.
        base_dir: Directory the relative configuration and data paths use.

    Returns:
        dict: ``history``, ``metrics``, ``observability``, ``summary`` and
        ``data`` of the run.
    '''
    config = config_module.load_config(os.path.join(PACKAGE_DIR, case['config']))
    prior_model = getattr(config, 'prior_model', 'ZOH')
    if prior_model != 'ZOH':
        raise ValueError(f'{case["config"]} is not a ZOH configuration '
                         f'(prior_model = {prior_model!r})')
    config.data_path = os.path.join(base_dir, case['data'])

    estimator = ClosedLoopKalman.initialize(config)
    model = estimator.model
    data = data_adapter.load_data(config)
    history = estimator.run(data.u)
    return {
        'history': history,
        'model': model,
        'data': data,
        'metrics': evaluation.evaluate(data.t, history, data, config, model),
        'observability': model.observability_report().as_dict(),
        'summary': config.summary(),
        'data_summary': data.summary(),
    }


def capture(cases=CASES, base_dir: str = ROOT, directory: str = REFERENCE_DIR) -> dict:
    '''
    Write the reference record from the current ZOH implementation.

    Only the phase 1 program is allowed to call this: it records what the ZOH
    path produced before the FOH upgrade.

    Args:
        cases: Cases to record, defaults to :data:`CASES`.
        base_dir: Directory the relative paths of ``cases`` use.
        directory: Destination directory.

    Returns:
        dict: The manifest that was written.
    '''
    os.makedirs(directory, exist_ok=True)
    arrays = {}
    manifest = {
        'description': ('Phase 1 five state ZOH output, captured before the FOH '
                        'upgrade, for the compatibility regression of section 5.'),
        'tolerance': TOLERANCE,
        'cases': [],
    }

    for case in cases:
        result = run_case(case, base_dir)
        for key in ARRAYS:
            arrays[f'{case["name"]}/{key}'] = np.asarray(result['history'][key],
                                                         dtype=np.float64)
        manifest['cases'].append({
            'name': case['name'],
            'config': case['config'],
            'data': case['data'],
            'samples': int(result['data'].t.size),
            'config_summary': result['summary'],
            'data_summary': result['data_summary'],
            'metrics': result['metrics'],
            'observability': result['observability'],
            'arrays': list(ARRAYS),
        })

    path = os.path.join(directory, 'zoh-reference.npz')
    np.savez_compressed(path, **arrays)
    manifest['file'] = os.path.basename(path)
    manifest['file_sha256'] = sha256(path)
    with open(os.path.join(directory, 'zoh-reference.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False,
                  default=evaluation._json_default)
    return manifest


def load(directory: str = REFERENCE_DIR) -> tuple:
    '''
    Load the stored reference record.

    Returns:
        tuple: ``(arrays, manifest)``.

    Raises:
        FileNotFoundError: When the fixture is absent.
    '''
    arrays_path = os.path.join(directory, 'zoh-reference.npz')
    manifest_path = os.path.join(directory, 'zoh-reference.json')
    for path in (arrays_path, manifest_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'reference record not found: {path}')
    with np.load(arrays_path) as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float64)
                  for key in archive.files}
    with open(manifest_path, 'r', encoding='utf-8', errors='replace') as handle:
        manifest = json.load(handle)
    return arrays, manifest


def compare(arrays: dict, manifest: dict, base_dir: str = ROOT,
            cases=None) -> dict:
    '''
    Re-run the recorded ZOH cases and compare them with the reference.

    Args:
        arrays: Reference arrays returned by :func:`load`.
        manifest: Reference manifest returned by :func:`load`.
        base_dir: Directory the relative paths of the cases use.
        cases: Optional restriction by case name.

    Returns:
        dict: Per array relative deviations, per metric comparisons and an
        overall ``passed`` flag.
    '''
    tolerance = float(manifest.get('tolerance', TOLERANCE)['array_relative'])
    metric_tolerance = float(
        manifest.get('tolerance', TOLERANCE)['metric_relative'])
    selected = set(cases or ())

    report = {'tolerance': manifest.get('tolerance', TOLERANCE),
              'cases': [], 'problems': []}
    max_relative = 0.

    for case in manifest['cases']:
        name = case['name']
        if selected and name not in selected:
            continue
        run = run_case(case, base_dir)
        entry = {'name': name, 'arrays': {}, 'samples': int(run['data'].t.size)}

        for key in case['arrays']:
            reference = arrays[f'{name}/{key}']
            actual = np.asarray(run['history'][key], dtype=np.float64)
            if actual.shape != reference.shape:
                report['problems'].append(
                    f'{name}/{key}: shape {actual.shape} instead of {reference.shape}')
                entry['arrays'][key] = {'relative': float('inf')}
                max_relative = float('inf')
                continue
            relative = _relative_difference(reference, actual)
            max_relative = max(max_relative, relative)
            entry['arrays'][key] = {'relative': relative,
                                    'passed': relative <= tolerance}

        metric_rows = []
        _compare_numbers(case['metrics'], run['metrics'],
                         f'{name}.metrics', metric_tolerance, metric_rows)
        _compare_numbers(case['observability'], run['observability'],
                         f'{name}.observability', metric_tolerance, metric_rows)
        for row in metric_rows:
            if row['status'] != 'ok':
                report['problems'].append(f"{row['path']}: {row['status']}")
        entry['metrics_max_relative'] = max(
            [row.get('relative', 0.) for row in metric_rows
             if row['status'] != 'text differs'] or [0.])
        entry['metrics_passed'] = all(row['status'] == 'ok'
                                      or row['status'] == 'text differs'
                                      for row in metric_rows)
        report['cases'].append(entry)

    arrays_passed = all(
        item['passed']
        for case in report['cases']
        for item in case['arrays'].values())
    metrics_passed = all(case['metrics_passed'] for case in report['cases'])
    report['max_array_relative'] = max_relative
    report['passed'] = bool(arrays_passed and metrics_passed
                            and not report['problems'])
    return report


def sha256(path: str) -> str:
    '''
    Hexadecimal SHA-256 of a file.
    '''
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(report: dict) -> str:
    '''
    One line summary of a regression report.
    '''
    if report['passed']:
        return (f"ZOH regression passed: {len(report['cases'])} cases, "
                f"max array relative deviation {report['max_array_relative']:.3e}")
    return (f"ZOH regression FAILED: {len(report['problems'])} problems, "
            f"max array relative deviation {report['max_array_relative']:.3e}")
