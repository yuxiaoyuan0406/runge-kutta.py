'''
Configuration loading and validation for the closed-loop MEMS accelerometer
Kalman filter.

The key names follow section 9 of the development task sheet.  Every physical
quantity that the task sheet marks as "to be filled" has to come from the
user: a missing or inconsistent value raises :class:`ConfigError` instead of
being silently replaced by a made-up number.
'''
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

STATE_LABELS = ('x', 'v', 'a', 'I', 'h')
'''
Name of the five state components of X = [x, v, a, I, h], h = e[k-1].
'''

# Section 9 of the task sheet: key -> (unit, meaning).
PARAMETER_SPEC: Dict[str, tuple] = {
    'fs': ('Hz', 'Closed-loop sampling rate.'),
    'Ts': ('s', 'Closed-loop sampling period, must match fs.'),
    'm': ('kg', 'Equivalent mass.'),
    'c': ('N*s/m', 'Damping coefficient.'),
    'k': ('N/m', 'Spring constant.'),
    'K_u': ('N / control unit', 'Signed PID output to feedback force gain.'),
    'k_cv': ('digital / m', 'Displacement to analog gain.'),
    'k_adc': ('control unit / analog', 'Analog to digital gain.'),
    'K_p': ('control unit / digital error', 'Proportional gain, discrete.'),
    'K_i': ('control unit / digital error', 'Integral gain, discrete.'),
    'K_d': ('control unit / digital error', 'Derivative gain, discrete.'),
    'sigma_n2': ('(digital error)^2', 'Variance of the discrete white readout noise.'),
    'mechanical_noise_psd': ('see mechanical_noise_unit', 'Equivalent random force or acceleration PSD.'),
    'psd_convention': ('-', 'two_sided_psd | one_sided_psd | one_sided_asd.'),
    'q_a': ('(m/s^2)^2', 'Per-sample acceleration increment variance.'),
    'q_a_grid': ('(m/s^2)^2', 'q_a values used by the sensitivity sweep.'),
    'X0_hat': ('mixed', 'Prior mean of the initial state, five entries.'),
    'P0': ('mixed', 'Prior covariance of the initial state, 5x5.'),
}

PASS_THROUGH_KEYS = (
    'data_path',
    'data_format',
    'column_map',
    'units',
    'sample_alignment',
    'test_windows',
    'baseline_gain',
    'output_dir',
    'seed',
    'state_scales',
    'init_mode',
    'known_integrator_init',
    'tolerances',
)

DEFAULT_TOLERANCES = {
    'symmetry': 1e-12,
    'psd_eigen': 1e-10,
    'singular_relative': 1e-12,
    'innovation_relative': 1e-8,
    'rank_relative': 1e-10,
}
'''Numerical tolerances only, they carry no physical meaning.'''

VALID_PSD_CONVENTIONS = ('two_sided_psd', 'one_sided_psd', 'one_sided_asd')
VALID_NOISE_UNITS = ('force', 'accel')
VALID_INIT_MODES = ('k0', 'k_minus_1')


class ConfigError(ValueError):
    '''
    Raised when the configuration is incomplete, inconsistent or unphysical.
    '''

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        text = '; '.join(self.problems)
        super().__init__(f'configuration is not usable: {text}')


@dataclass
class Config:
    '''
    Validated configuration of the closed-loop Kalman filter.

    Attributes are named after section 9 of the task sheet; the derived ones
    (``k_s``, ``alpha``, ``beta``, ``q_b``, ``q_F``, ``r``, ``R``) are filled
    in by :func:`build_config` so that every later module sees one consistent
    set of numbers.
    '''

    fs: float
    Ts: float
    m: float
    c: float
    k: float
    K_u: float
    k_cv: float
    k_adc: float
    K_p: float
    K_i: float
    K_d: float
    sigma_n2: float
    mechanical_noise_psd: float
    psd_convention: str
    mechanical_noise_unit: str
    q_a: float
    q_a_grid: List[float]
    X0_hat: np.ndarray
    P0: np.ndarray

    k_s: float = 0.
    alpha: float = 0.
    beta: float = 0.
    q_b: float = 0.
    q_F: float = 0.
    r: float = 0.
    R: float = 0.

    init_mode: str = 'k0'
    known_integrator_init: Optional[bool] = None
    data_path: Optional[str] = None
    data_format: str = 'npy_dir'
    column_map: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, float] = field(default_factory=dict)
    sample_alignment: Dict[str, Any] = field(default_factory=dict)
    test_windows: Dict[str, List[float]] = field(default_factory=dict)
    baseline_gain: Optional[float] = None
    output_dir: str = 'data/kalman-accelerometer'
    seed: Optional[int] = None
    state_scales: Optional[np.ndarray] = None
    tolerances: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOLERANCES))

    def summary(self) -> Dict[str, Any]:
        '''
        Return a JSON friendly summary of the configuration.
        '''
        return {
            'fs': self.fs,
            'Ts': self.Ts,
            'm': self.m,
            'c': self.c,
            'k': self.k,
            'K_u': self.K_u,
            'k_cv': self.k_cv,
            'k_adc': self.k_adc,
            'k_s': self.k_s,
            'K_p': self.K_p,
            'K_i': self.K_i,
            'K_d': self.K_d,
            'alpha': self.alpha,
            'beta': self.beta,
            'sigma_n2': self.sigma_n2,
            'mechanical_noise_psd': self.mechanical_noise_psd,
            'psd_convention': self.psd_convention,
            'mechanical_noise_unit': self.mechanical_noise_unit,
            'q_b': self.q_b,
            'q_F': self.q_F,
            'q_a': self.q_a,
            'output_dir': self.output_dir,
        }


def _is_number(value: Any) -> bool:
    '''
    True when ``value`` is a finite real number.
    '''
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(number)


def _require_number(raw: Dict[str, Any], key: str, problems: List[str]) -> Optional[float]:
    '''
    Fetch a finite scalar, recording a problem when it is missing or invalid.
    '''
    if key not in raw or raw[key] is None:
        problems.append(f'missing required parameter "{key}" ({PARAMETER_SPEC[key][1]})')
        return None
    if not _is_number(raw[key]):
        problems.append(f'parameter "{key}" must be a finite number, got {raw[key]!r}')
        return None
    return float(raw[key])


def _as_matrix(value: Any, shape: tuple, name: str, problems: List[str]) -> Optional[np.ndarray]:
    '''
    Convert ``value`` to a finite float array of the requested shape.
    '''
    if value is None:
        problems.append(f'missing required parameter "{name}"')
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        problems.append(f'parameter "{name}" must be a numeric array')
        return None
    if array.shape != shape:
        problems.append(f'parameter "{name}" must have shape {shape}, got {array.shape}')
        return None
    if not np.all(np.isfinite(array)):
        problems.append(f'parameter "{name}" contains non finite entries')
        return None
    return array


def _check_symmetric(matrix: np.ndarray, name: str, tolerance: float, problems: List[str]):
    '''
    Record a problem when ``matrix`` is not symmetric within ``tolerance``.
    '''
    scale = max(1., float(np.max(np.abs(matrix))))
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > tolerance * scale:
        problems.append(f'parameter "{name}" is not symmetric (max asymmetry {asymmetry:.3e})')


def _check_psd(matrix: np.ndarray, name: str, tolerance: float, problems: List[str]):
    '''
    Record a problem when ``matrix`` has a significantly negative eigenvalue.
    '''
    symmetric = 0.5 * (matrix + matrix.T)
    eigen_min = float(np.min(np.linalg.eigvalsh(symmetric)))
    scale = max(1., float(np.max(np.abs(symmetric))))
    if eigen_min < -tolerance * scale:
        problems.append(f'parameter "{name}" is not positive semi definite (min eigenvalue {eigen_min:.3e})')


def _convert_noise(raw: Dict[str, Any], problems: List[str]) -> tuple:
    '''
    Convert the configured PSD into ``q_F`` (N^2/Hz) and ``q_b`` ((m/s^2)^2/Hz).

    Section 12.2: a two sided PSD equals the white noise intensity ``q``; a one
    sided PSD and a one sided ASD are halved after squaring the ASD.  The force
    and equivalent acceleration forms describe one and the same noise, so the
    value is converted, never added twice.
    '''
    convention = raw.get('psd_convention')
    if convention is None:
        problems.append('missing required parameter "psd_convention" '
                        f'(one of {VALID_PSD_CONVENTIONS})')
        return None, None
    if convention not in VALID_PSD_CONVENTIONS:
        problems.append(f'parameter "psd_convention" must be one of {VALID_PSD_CONVENTIONS}, '
                        f'got {convention!r}')
        return None, None

    unit = raw.get('mechanical_noise_unit', 'force')
    if unit not in VALID_NOISE_UNITS:
        problems.append(f'parameter "mechanical_noise_unit" must be one of {VALID_NOISE_UNITS}, '
                        f'got {unit!r}')
        return None, None

    value = _require_number(raw, 'mechanical_noise_psd', problems)
    mass = raw.get('m')
    if value is None or not _is_number(mass):
        return None, None

    spectral = value * value if convention == 'one_sided_asd' else value
    if convention != 'two_sided_psd':
        spectral = spectral / 2
    if spectral < 0:
        problems.append('parameter "mechanical_noise_psd" must not be negative')
        return None, None

    mass = float(mass)
    if mass <= 0:
        # The mass problem itself is recorded by the caller.
        return None, None
    if unit == 'force':
        return spectral, spectral / (mass * mass)
    return spectral * mass * mass, spectral


def build_config(raw: Dict[str, Any]) -> Config:
    '''
    Validate a raw configuration dictionary and build a :class:`Config`.

    Args:
        raw: Parsed JSON content, keys as described by section 9.

    Returns:
        Config: Validated configuration with derived quantities filled in.

    Raises:
        ConfigError: When any required parameter is missing, non finite or
            internally inconsistent.  All problems are reported at once.
    '''
    problems: List[str] = []
    tolerances = dict(DEFAULT_TOLERANCES)
    tolerances.update(raw.get('tolerances') or {})

    # --- sampling -----------------------------------------------------
    fs = _require_number(raw, 'fs', problems)
    Ts = _require_number(raw, 'Ts', problems)
    if fs is not None and Ts is not None:
        if fs <= 0 or Ts <= 0:
            problems.append('parameters "fs" and "Ts" must be positive')
        elif abs(fs * Ts - 1.) > 1e-9 * max(1., abs(fs * Ts)):
            problems.append(f'parameters "fs" ({fs}) and "Ts" ({Ts}) disagree: fs*Ts = {fs * Ts}')

    # --- mechanical ---------------------------------------------------
    mass = _require_number(raw, 'm', problems)
    damping = _require_number(raw, 'c', problems)
    stiffness = _require_number(raw, 'k', problems)
    if mass is not None and mass <= 0:
        problems.append(f'parameter "m" must be positive, got {mass}')
    if stiffness is not None and stiffness < 0:
        problems.append(f'parameter "k" must not be negative, got {stiffness}')
    if damping is not None and damping < 0:
        problems.append(f'parameter "c" must not be negative, got {damping}')

    # --- feedback and readout -----------------------------------------
    K_u = _require_number(raw, 'K_u', problems)
    k_cv = _require_number(raw, 'k_cv', problems)
    k_adc = _require_number(raw, 'k_adc', problems)

    # --- PID ----------------------------------------------------------
    K_p = _require_number(raw, 'K_p', problems)
    K_i = _require_number(raw, 'K_i', problems)
    K_d = _require_number(raw, 'K_d', problems)

    sigma_n2 = _require_number(raw, 'sigma_n2', problems)
    if sigma_n2 is not None and sigma_n2 < 0:
        problems.append(f'parameter "sigma_n2" must not be negative, got {sigma_n2}')

    q_a = _require_number(raw, 'q_a', problems)
    if q_a is not None and q_a < 0:
        problems.append(f'parameter "q_a" must not be negative, got {q_a}')

    q_a_grid = raw.get('q_a_grid')
    if q_a_grid is None:
        q_a_grid = [] if q_a is None else [q_a]
    else:
        values = []
        for entry in q_a_grid:
            if not _is_number(entry) or float(entry) < 0:
                problems.append(f'parameter "q_a_grid" must hold non negative numbers, got {entry!r}')
                break
            values.append(float(entry))
        q_a_grid = values

    # --- noise spectral density ---------------------------------------
    q_F, q_b = _convert_noise(raw, problems)

    # --- initial state -------------------------------------------------
    X0 = _as_matrix(raw.get('X0_hat'), (5,), 'X0_hat', problems)
    P0 = _as_matrix(raw.get('P0'), (5, 5), 'P0', problems)

    init_mode = raw.get('init_mode', 'k0')
    if init_mode not in VALID_INIT_MODES:
        problems.append(f'parameter "init_mode" must be one of {VALID_INIT_MODES}, '
                        f'got {init_mode!r}')
    elif init_mode == 'k_minus_1':
        # Section 16: I[0] = I[-1] + K_i * e[-1].
        if X0 is not None and P0 is not None and K_i is not None:
            T_init = np.eye(5)
            T_init[3, 4] = K_i
            X0 = T_init @ X0
            P0 = T_init @ P0 @ T_init.T
            P0 = 0.5 * (P0 + P0.T)

    if P0 is not None:
        _check_symmetric(P0, 'P0', tolerances['symmetry'], problems)
        _check_psd(P0, 'P0', tolerances['psd_eigen'], problems)

    if X0 is not None and P0 is not None:
        variances = np.diag(P0)
        if np.any(variances < 0):
            problems.append('parameter "P0" has a negative diagonal entry')

    if problems:
        raise ConfigError(problems)

    state_scales = raw.get('state_scales')
    if state_scales is not None:
        scales = np.asarray(state_scales, dtype=np.float64)
        if scales.shape != (5,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ConfigError(['parameter "state_scales" must hold five positive numbers'])
        state_scales = scales

    config = Config(
        fs=float(fs),
        Ts=float(Ts),
        m=float(mass),
        c=float(damping),
        k=float(stiffness),
        K_u=float(K_u),
        k_cv=float(k_cv),
        k_adc=float(k_adc),
        K_p=float(K_p),
        K_i=float(K_i),
        K_d=float(K_d),
        sigma_n2=float(sigma_n2),
        mechanical_noise_psd=float(raw['mechanical_noise_psd']),
        psd_convention=str(raw['psd_convention']),
        mechanical_noise_unit=str(raw.get('mechanical_noise_unit', 'force')),
        q_a=float(q_a),
        q_a_grid=[float(value) for value in q_a_grid],
        X0_hat=X0,
        P0=P0,
        init_mode=init_mode,
        state_scales=state_scales,
        tolerances=tolerances,
    )

    config.k_s = config.k_cv * config.k_adc
    config.alpha = config.K_p + config.K_d
    config.beta = -config.K_d
    config.q_F = float(q_F)
    config.q_b = float(q_b)
    config.r = config.sigma_n2
    config.R = config.alpha ** 2 * config.r

    config.known_integrator_init = raw.get('known_integrator_init')
    config.data_path = raw.get('data_path')
    config.data_format = raw.get('data_format', 'npy_dir')
    config.column_map = dict(raw.get('column_map') or {})
    config.units = dict(raw.get('units') or {})
    config.sample_alignment = dict(raw.get('sample_alignment') or {})
    config.test_windows = dict(raw.get('test_windows') or {})
    config.baseline_gain = raw.get('baseline_gain')
    config.output_dir = raw.get('output_dir', 'data/kalman-accelerometer')
    config.seed = raw.get('seed')

    return config


def load_config(path: str) -> Config:
    '''
    Load and validate a JSON configuration file.

    Args:
        path: Path of the configuration file.

    Returns:
        Config: Validated configuration.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ConfigError: When the content is incomplete or inconsistent.
    '''
    if not os.path.isfile(path):
        raise FileNotFoundError(f'configuration file not found: {path}')
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ConfigError([f'{path}: top level JSON value must be an object'])
    return build_config(raw)


def missing_parameter_report(path: str) -> Dict[str, Any]:
    '''
    Report the parameters of a configuration file that still need a value.

    This is the "read section 9 and report completeness" step of the task
    sheet.  It never raises on missing values: it reports them.

    Args:
        path: Path of the configuration file.

    Returns:
        Dict[str, Any]: ``ok`` flag plus the list of missing keys.
    '''
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        raw = json.load(handle)

    missing = [key for key in PARAMETER_SPEC if raw.get(key) is None]
    return {
        'path': path,
        'ok': not missing,
        'missing': missing,
        'missing_detail': [{'key': key, 'unit': PARAMETER_SPEC[key][0],
                            'meaning': PARAMETER_SPEC[key][1]} for key in missing],
    }
