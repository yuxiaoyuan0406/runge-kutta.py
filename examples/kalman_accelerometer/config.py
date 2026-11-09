'''
Configuration loading and validation for the closed-loop MEMS accelerometer
Kalman filter.

The key names follow section 9 of the development task sheet (V1.1) plus the
phase 2 additions of section 6: ``prior_model``, ``q_j``, ``jerk0_hat``,
``P0_FOH``, ``jerk_init_mode``, ``foh_output_fields``, ``zoh_reference``.

Every physical quantity that the task sheet marks as "to be filled" has to come
from the user: a missing or inconsistent value raises :class:`ConfigError`
instead of being silently replaced by a made-up number.

Two estimators are described by one configuration schema:

``ZOH``
    the phase 1 five state model ``[x, v, a, I, h]`` with the acceleration
    random walk ``q_a``;
``FOH``
    the phase 2 six state model ``[x, v, a, j, I, h]`` with the jerk random
    walk ``q_j``.  ``q_a`` is *not* read on this path: the two variances belong
    to different physical quantities and must never be exchanged.
'''
import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

STATE_LABELS = ('x', 'v', 'a', 'I', 'h')
'''Name of the five state components of the ZOH state ``[x, v, a, I, h]``.'''

STATE_LABELS_FOH = ('x', 'v', 'a', 'j', 'I', 'h')
'''Name of the six state components of the FOH state ``[x, v, a, j, I, h]``.'''

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
    'prior_model': ('-', 'ZOH (phase 1) or FOH (phase 2) acceleration prior.'),
    'q_a': ('(m/s^2)^2', 'Per-sample acceleration increment variance, ZOH only.'),
    'q_a_grid': ('(m/s^2)^2', 'q_a values used by the ZOH sensitivity sweep.'),
    'q_j': ('(m/s^3)^2', 'Per-sample Jerk increment variance, FOH only.'),
    'jerk0_hat': ('m/s^3', 'Prior mean of the Jerk state, FOH only.'),
    'P0_FOH': ('mixed', 'Prior covariance of [x, v, a, j, I, h], 6x6, FOH only.'),
    'jerk_init_mode': ('-', 'known | unknown Jerk initial value, FOH only.'),
    'X0_hat': ('mixed', 'Prior mean of the initial state, five entries (both models).'),
    'P0': ('mixed', 'Prior covariance of the initial state, 5x5 (ZOH, or FOH assembly).'),
}

ZOH_ONLY_KEYS = ('q_a', 'q_a_grid')
'''Keys required by the ZOH path only.'''

FOH_ONLY_KEYS = ('q_j', 'jerk0_hat', 'jerk_init_mode')
'''Keys required by the FOH path only.'''

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
    'foh_output_fields',
    'zoh_reference',
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
VALID_PRIOR_MODELS = ('ZOH', 'FOH')
VALID_JERK_INIT_MODES = ('known', 'unknown')

DEFAULT_FOH_OUTPUT_FIELDS = ('jerk_hat', 'jerk_variance',
                             'jerk_accel_correlation',
                             'jerk_posterior_variance')
'''Diagnostic fields appended to the FOH output; the phase 1 fields stay.'''

STATE_INDEX = {
    'ZOH': {'x': 0, 'v': 1, 'a': 2, 'I': 3, 'h': 4},
    'FOH': {'x': 0, 'v': 1, 'a': 2, 'j': 3, 'I': 4, 'h': 5},
}
'''Position of every state component in the assembled state vector.'''


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

    ``X0_hat`` and ``P0`` hold the five state prior of the ZOH path.  The FOH
    path uses ``X0_FOH`` and ``P0_FOH``; :attr:`prior_mean` and
    :attr:`prior_covariance` return the pair that belongs to the selected
    prior model, so downstream code never has to branch by itself.
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
    X0_hat: Optional[np.ndarray]
    P0: Optional[np.ndarray]

    k_s: float = 0.
    alpha: float = 0.
    beta: float = 0.
    q_b: float = 0.
    q_F: float = 0.
    r: float = 0.
    R: float = 0.

    prior_model: str = 'ZOH'
    q_j: float = 0.
    jerk0_hat: float = 0.
    jerk_init_mode: str = 'known'
    X0_FOH: Optional[np.ndarray] = None
    P0_FOH: Optional[np.ndarray] = None
    foh_output_fields: List[str] = field(default_factory=list)
    zoh_reference: Dict[str, Any] = field(default_factory=dict)

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

    # ------------------------------------------------------------------
    # prior model dependent views
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        '''
        Number of states of the selected prior model, five or six.
        '''
        return 6 if self.prior_model == 'FOH' else 5

    @property
    def state_labels(self) -> tuple:
        '''
        Names of the state components in the selected order.
        '''
        return STATE_LABELS_FOH if self.prior_model == 'FOH' else STATE_LABELS

    @property
    def index(self) -> Dict[str, int]:
        '''
        Position of every state component in the selected state vector.
        '''
        return STATE_INDEX[self.prior_model]

    @property
    def has_jerk(self) -> bool:
        '''
        True on the FOH path, where a Jerk state exists.
        '''
        return self.prior_model == 'FOH'

    @property
    def prior_mean(self) -> np.ndarray:
        '''
        Prior mean of the selected model, shape ``(dim,)``.
        '''
        array = self.X0_FOH if self.has_jerk else self.X0_hat
        if array is None:
            raise ConfigError([f'the {self.prior_model} prior mean is not configured'])
        return np.asarray(array, dtype=np.float64)

    @property
    def prior_covariance(self) -> np.ndarray:
        '''
        Prior covariance of the selected model, shape ``(dim, dim)``.
        '''
        array = self.P0_FOH if self.has_jerk else self.P0
        if array is None:
            raise ConfigError([f'the {self.prior_model} prior covariance is not configured'])
        return np.asarray(array, dtype=np.float64)

    def summary(self) -> Dict[str, Any]:
        '''
        Return a JSON friendly summary of the configuration.
        '''
        summary = {
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
            'prior_model': self.prior_model,
            'state_labels': list(self.state_labels),
            'output_dir': self.output_dir,
        }
        if self.has_jerk:
            summary.update({
                'q_j': self.q_j,
                'jerk0_hat': self.jerk0_hat,
                'jerk_init_mode': self.jerk_init_mode,
                'foh_output_fields': list(self.foh_output_fields),
                'zoh_reference': dict(self.zoh_reference),
            })
        else:
            summary['q_a'] = self.q_a
        return summary


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


def _optional_number(raw: Dict[str, Any], key: str, problems: List[str]) -> Optional[float]:
    '''
    Fetch a finite scalar that may be absent.
    '''
    if key not in raw or raw[key] is None:
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


def _as_vector_any(value: Any, shapes: tuple, name: str,
                   problems: List[str]) -> Optional[np.ndarray]:
    '''
    Convert ``value`` to a finite float vector of one of the accepted shapes.
    '''
    if value is None:
        problems.append(f'missing required parameter "{name}"')
        return None
    try:
        array = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        problems.append(f'parameter "{name}" must be a numeric array')
        return None
    if array.shape not in shapes:
        problems.append(f'parameter "{name}" must have shape one of {shapes}, '
                        f'got {array.shape}')
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


def _build_foh_prior(raw: Dict[str, Any], problems: List[str], tolerances: Dict[str, float],
                     K_i: Optional[float], q_a: Optional[float]) -> tuple:
    '''
    Assemble the six state FOH prior of section 15.1.

    Two legal routes are accepted:

    ``P0_FOH``
        the complete 6x6 covariance, including any Jerk cross covariance;
    ``P0`` + ``jerk_var0``
        the phase 1 5x5 prior embedded with the selection matrix ``S`` of
        section 15.1 plus a Jerk variance.  This is an assembly convenience
        only: it does not invent a physical value, the user has to supply it.

    Args:
        raw: Parsed configuration.
        problems: Problem collector.
        tolerances: Numerical tolerances.
        K_i: Integral gain, used by ``init_mode = k_minus_1``.
        q_a: Parsed ``q_a`` if it was present, ignored on this path.

    Returns:
        tuple: ``(X0_FOH, P0_FOH, jerk0_hat, jerk_init_mode)``; entries are
        ``None`` when a problem was recorded.
    '''
    jerk0 = _optional_number(raw, 'jerk0_hat', problems)

    mode = raw.get('jerk_init_mode')
    if mode is None:
        problems.append('missing required parameter "jerk_init_mode" '
                        f'(one of {VALID_JERK_INIT_MODES})')
        mode = None
    elif mode not in VALID_JERK_INIT_MODES:
        problems.append(f'parameter "jerk_init_mode" must be one of '
                        f'{VALID_JERK_INIT_MODES}, got {mode!r}')
        mode = None

    # --- mean ---------------------------------------------------------
    X0_FOH = None
    mean_raw = raw.get('X0_hat')
    if mean_raw is None:
        problems.append('missing required parameter "X0_hat"')
    else:
        try:
            mean = np.asarray(mean_raw, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            problems.append('parameter "X0_hat" must be a numeric array')
            mean = None
        if mean is not None and not np.all(np.isfinite(mean)):
            problems.append('parameter "X0_hat" contains non finite entries')
            mean = None
        if mean is not None:
            if mean.shape == (6,):
                X0_FOH = mean.copy()
                if jerk0 is None:
                    jerk0 = float(mean[3])
            elif mean.shape == (5,):
                if jerk0 is None:
                    problems.append('parameter "jerk0_hat" is required when '
                                    'X0_hat holds the five common states only')
                    X0_FOH = None
                else:
                    X0_FOH = np.insert(mean, 3, jerk0)
            else:
                problems.append(f'parameter "X0_hat" must have 5 or 6 entries for '
                                f'the FOH prior, got {mean.shape}')

    # --- covariance ---------------------------------------------------
    selection = np.zeros((6, 5), dtype=np.float64)
    for source, target in enumerate((0, 1, 2, 4, 5)):
        selection[target, source] = 1.

    P0_FOH = None
    if raw.get('P0_FOH') is not None:
        P0_FOH = _as_matrix(raw.get('P0_FOH'), (6, 6), 'P0_FOH', problems)
    else:
        P0_five = None
        if raw.get('P0') is not None:
            P0_five = _as_matrix(raw.get('P0'), (5, 5), 'P0', problems)
        else:
            problems.append('missing required parameter "P0_FOH" (6x6), or '
                            '"P0" (5x5) together with "jerk_var0"')
        if P0_five is not None:
            jerk_variance = _optional_number(raw, 'jerk_var0', problems)
            if jerk_variance is None:
                problems.append('parameter "jerk_var0" is required to embed the '
                                'five state prior P0 into the six state P0_FOH')
            elif jerk_variance < 0:
                problems.append(f'parameter "jerk_var0" must not be negative, '
                                f'got {jerk_variance}')
            else:
                P0_FOH = selection @ P0_five @ selection.T
                P0_FOH[3, 3] = jerk_variance

    if P0_FOH is not None:
        _check_symmetric(P0_FOH, 'P0_FOH', tolerances['symmetry'], problems)
        _check_psd(P0_FOH, 'P0_FOH', tolerances['psd_eigen'], problems)
        if np.any(np.diag(P0_FOH) < 0):
            problems.append('parameter "P0_FOH" has a negative diagonal entry')

        # The Jerk row and column encode how well the initial Jerk is known.
        jerk_row = np.max(np.abs(P0_FOH[3, :]))
        jerk_column = np.max(np.abs(P0_FOH[:, 3]))
        known_tolerance = tolerances['symmetry'] * max(1., float(np.max(np.abs(P0_FOH))))
        if mode == 'known':
            if max(jerk_row, jerk_column) > known_tolerance:
                problems.append(
                    'jerk_init_mode = "known" requires zero Jerk variance and zero '
                    f'Jerk cross covariances in P0_FOH (found {max(jerk_row, jerk_column):.3e})')
        elif mode == 'unknown':
            if P0_FOH[3, 3] <= 0:
                problems.append(
                    'jerk_init_mode = "unknown" requires a finite positive Jerk '
                    'variance in P0_FOH (jerk_var0 or P0_FOH[3, 3])')

    # --- section 16 index shift --------------------------------------
    if (raw.get('init_mode', 'k0') == 'k_minus_1' and X0_FOH is not None
            and P0_FOH is not None and K_i is not None):
        T_init = np.eye(6)
        T_init[4, 5] = K_i
        X0_FOH = T_init @ X0_FOH
        P0_FOH = T_init @ P0_FOH @ T_init.T
        P0_FOH = 0.5 * (P0_FOH + P0_FOH.T)

    return X0_FOH, P0_FOH, jerk0, mode


def build_config(raw: Dict[str, Any]) -> Config:
    '''
    Validate a raw configuration dictionary and build a :class:`Config`.

    Args:
        raw: Parsed JSON content, keys as described by section 9 (V1.1) and
            section 6 (V2.1).

    Returns:
        Config: Validated configuration with derived quantities filled in.

    Raises:
        ConfigError: When any required parameter is missing, non finite or
            internally inconsistent.  All problems are reported at once.
    '''
    problems: List[str] = []
    tolerances = dict(DEFAULT_TOLERANCES)
    tolerances.update(raw.get('tolerances') or {})

    # --- prior model ---------------------------------------------------
    prior_model = raw.get('prior_model')
    if prior_model is None:
        prior_model = 'ZOH'
    if not isinstance(prior_model, str) or prior_model.strip().upper() not in VALID_PRIOR_MODELS:
        problems.append(f'parameter "prior_model" must be one of {VALID_PRIOR_MODELS}, '
                        f'got {prior_model!r}')
        prior_model = 'ZOH'
    prior_model = prior_model.strip().upper()
    foh = prior_model == 'FOH'

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

    # --- process noise ------------------------------------------------
    # q_a belongs to the ZOH path, q_j to the FOH path.  Neither is inherited
    # by the other model: the numbers describe different physical quantities.
    q_a = _optional_number(raw, 'q_a', problems)
    if q_a is not None and q_a < 0:
        problems.append(f'parameter "q_a" must not be negative, got {q_a}')
    q_j = None
    if foh:
        q_j = _require_number(raw, 'q_j', problems)
        if q_j is not None and q_j < 0:
            problems.append(f'parameter "q_j" must not be negative, got {q_j}')
    else:
        if q_a is None and not any('parameter "q_a"' in item for item in problems):
            problems.append('missing required parameter "q_a" '
                            '(per sample acceleration increment variance, ZOH path)')
        q_j = _optional_number(raw, 'q_j', problems)
        if q_j is not None and q_j < 0:
            problems.append(f'parameter "q_j" must not be negative, got {q_j}')

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
    init_mode = raw.get('init_mode', 'k0')
    if init_mode not in VALID_INIT_MODES:
        problems.append(f'parameter "init_mode" must be one of {VALID_INIT_MODES}, '
                        f'got {init_mode!r}')

    X0_FOH = None
    P0_FOH = None
    jerk0_hat = 0.
    jerk_init_mode = 'known'

    if foh:
        X0_FOH, P0_FOH, jerk0_hat, jerk_init_mode = _build_foh_prior(
            raw, problems, tolerances, K_i, q_a)
        X0 = None
        P0 = None
    else:
        X0 = _as_matrix(raw.get('X0_hat'), (5,), 'X0_hat', problems)
        P0 = _as_matrix(raw.get('P0'), (5, 5), 'P0', problems)

        if init_mode == 'k_minus_1' and X0 is not None and P0 is not None and K_i is not None:
            # Section 16: I[0] = I[-1] + K_i * e[-1].
            T_init = np.eye(5)
            T_init[3, 4] = K_i
            X0 = T_init @ X0
            P0 = T_init @ P0 @ T_init.T
            P0 = 0.5 * (P0 + P0.T)

        if P0 is not None:
            _check_symmetric(P0, 'P0', tolerances['symmetry'], problems)
            _check_psd(P0, 'P0', tolerances['psd_eigen'], problems)

        if X0 is not None and P0 is not None:
            if np.any(np.diag(P0) < 0):
                problems.append('parameter "P0" has a negative diagonal entry')

    if problems:
        raise ConfigError(problems)

    state_scales = raw.get('state_scales')
    if state_scales is not None:
        scales = np.asarray(state_scales, dtype=np.float64)
        dimension = 6 if foh else 5
        if scales.shape != (dimension,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ConfigError([f'parameter "state_scales" must hold {dimension} '
                               'positive numbers'])
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
        q_a=float(q_a) if q_a is not None else 0.,
        q_a_grid=[float(value) for value in q_a_grid],
        X0_hat=None if X0 is None else np.asarray(X0, dtype=np.float64),
        P0=None if P0 is None else np.asarray(P0, dtype=np.float64),
        prior_model=prior_model,
        q_j=float(q_j) if q_j is not None else 0.,
        jerk0_hat=float(jerk0_hat) if jerk0_hat is not None else 0.,
        jerk_init_mode=jerk_init_mode or 'known',
        X0_FOH=None if X0_FOH is None else np.asarray(X0_FOH, dtype=np.float64),
        P0_FOH=None if P0_FOH is None else np.asarray(P0_FOH, dtype=np.float64),
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

    config.foh_output_fields = list(raw.get('foh_output_fields')
                                    or DEFAULT_FOH_OUTPUT_FIELDS)
    config.zoh_reference = dict(raw.get('zoh_reference') or {})
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


def with_prior_model(config: Config, prior_model: str, **overrides) -> Config:
    '''
    Return a copy of ``config`` switched to another prior model.

    Used by the verification module to build the ZOH counterpart of an FOH
    configuration without touching the files on disk.

    Args:
        config: Source configuration.
        prior_model: ``'ZOH'`` or ``'FOH'``.
        **overrides: Attribute values set on the copy afterwards.

    Returns:
        Config: The modified copy.
    '''
    clone = copy.deepcopy(config)
    clone.prior_model = prior_model.strip().upper()
    for key, value in overrides.items():
        setattr(clone, key, value)
    return clone


def missing_parameter_report(path: str) -> Dict[str, Any]:
    '''
    Report the parameters of a configuration file that still need a value.

    This is the "read section 9 and report completeness" step of the task
    sheet.  It never raises on missing values: it reports them.  The required
    set depends on ``prior_model``: ``q_a`` belongs to ZOH, ``q_j``,
    ``jerk0_hat`` and ``jerk_init_mode`` to FOH.

    Args:
        path: Path of the configuration file.

    Returns:
        Dict[str, Any]: ``ok`` flag plus the list of missing keys.
    '''
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        raw = json.load(handle)

    prior_model = raw.get('prior_model') or 'ZOH'
    if not isinstance(prior_model, str) or prior_model.strip().upper() not in VALID_PRIOR_MODELS:
        prior_model = 'ZOH'
    prior_model = prior_model.strip().upper()

    keys = [key for key in PARAMETER_SPEC]
    if prior_model == 'FOH':
        required = [key for key in keys if key not in ZOH_ONLY_KEYS and key != 'P0']
    else:
        required = [key for key in keys if key not in FOH_ONLY_KEYS and key != 'P0_FOH']

    missing = [key for key in required if raw.get(key) is None]
    if prior_model == 'FOH' and raw.get('P0') is not None:
        # The legal assembly route of section 15.1 embeds the five state prior
        # with the selection matrix, so P0 alone can carry the covariance.
        missing = [key for key in missing if key != 'P0_FOH']
    return {
        'path': path,
        'prior_model': prior_model,
        'ok': not missing,
        'missing': missing,
        'missing_detail': [{'key': key, 'unit': PARAMETER_SPEC[key][0],
                            'meaning': PARAMETER_SPEC[key][1]} for key in missing],
    }
