'''
Shared configuration and validation of the baseline low pass filters.

The two filters of this package are independent references for the Kalman
estimator of ``examples/kalman_accelerometer``.  Both only see the current and
the past samples of an acceleration sequence that is already expressed in
m/s^2: no unit conversion, no detrending, no zero phase filtering and no
Kalman import happen here.
'''
import math
import numbers
import warnings

import numpy as np

ZERO_STATE = 'zero_state'
STEADY_STATE = 'steady_state'
INIT_MODES = (ZERO_STATE, STEADY_STATE)

DEFAULT_FS = 128000.0
DEFAULT_FC_DIVISOR = 128.0


def _as_float(value, name):
    '''
    Convert one real scalar to a finite Python float.
    '''
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f'{name} must be a real number, got {type(value).__name__}')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return result


def validate_fs(fs):
    '''
    Check the sampling frequency, in Hz.
    '''
    value = _as_float(fs, 'fs')
    if value <= 0.0:
        raise ValueError(f'fs must be strictly positive, got {value} Hz')
    return value


def validate_fc(fs, fc, name='fc'):
    '''
    Check one cutoff frequency against a sampling frequency, in Hz.

    ``fc`` must be strictly inside ``(0, fs / 2)``: a cutoff at the Nyquist
    frequency is refused instead of being silently clamped.
    '''
    value = _as_float(fc, name)
    nyquist = fs / 2.0
    if value <= 0.0:
        raise ValueError(f'{name} must be strictly positive, got {value} Hz')
    if value >= nyquist:
        raise ValueError(
            f'{name} must stay below the Nyquist frequency fs/2 = {nyquist} Hz, '
            f'got {value} Hz')
    return value


def resolve_fc(fs, fc, name='fc'):
    '''
    Return ``fc`` or the documented default when it is not given.

    The default is ``fs / 128`` and it is never silent: a ``UserWarning`` is
    emitted so that a benchmark cannot be run on an unconfirmed cutoff without
    noticing it.
    '''
    if fc is None:
        default = fs / DEFAULT_FC_DIVISOR
        warnings.warn(
            f'{name} was not given, using the default {name} = fs/'
            f'{DEFAULT_FC_DIVISOR:g} = {default:g} Hz; pass an explicit cutoff '
            'frequency for a configured filter',
            UserWarning,
            stacklevel=3)
        return validate_fc(fs, default, name)
    return validate_fc(fs, fc, name)


def check_init_mode(init_mode):
    '''
    Check the initialisation mode name.
    '''
    if init_mode not in INIT_MODES:
        raise ValueError(
            f'init_mode must be one of {INIT_MODES}, got {init_mode!r}')
    return init_mode


def check_initialisation(init_mode, initial_value):
    '''
    Return the validated ``(init_mode, initial_value)`` pair.

    ``steady_state`` needs an explicit value: the filter must not read a future
    sample to guess it.  ``zero_state`` refuses a value instead of ignoring it.
    '''
    mode = check_init_mode(init_mode)
    if mode == ZERO_STATE:
        if initial_value is not None:
            raise ValueError(
                'zero_state forbids initial_value; pass '
                'init_mode="steady_state" to start from a constant steady state')
        return mode, None
    if initial_value is None:
        raise ValueError(
            'steady_state requires an explicit initial_value; the filter must '
            'not read a future sample')
    return mode, _as_float(initial_value, 'initial_value')


def check_sample(a_raw, name='a_raw'):
    '''
    Check one input sample.
    '''
    return _as_float(a_raw, name)


def check_sequence(samples, name='a_raw'):
    '''
    Check a one dimensional real finite input sequence.
    '''
    values = np.asarray(samples)
    if np.iscomplexobj(values):
        raise ValueError(f'{name} must be real, got a complex sequence')
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f'{name} must be one dimensional, got shape {values.shape}')
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must be finite')
    return values


def prepare_frequencies(frequencies, fs):
    '''
    Return ``(f, is_scalar)`` for the frequency response evaluation.

    Frequencies are limited to ``[0, fs/2]``: beyond Nyquist the digital
    response is only an alias of the one below it.
    '''
    values = np.asarray(frequencies)
    if np.iscomplexobj(values):
        raise ValueError('f must be real')
    values = np.asarray(values, dtype=np.float64)
    is_scalar = values.ndim == 0
    values = np.atleast_1d(values)
    if values.ndim != 1:
        raise ValueError(f'f must be one dimensional, got shape {values.shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError('f must be finite')
    nyquist = fs / 2.0
    if np.any(values < 0.0) or np.any(values > nyquist):
        raise ValueError(f'f must lie in [0, fs/2] = [0, {nyquist}] Hz')
    return values, is_scalar


class BaselineFilter:
    '''
    Common configuration, reset and sequence behaviour of the two filters.

    Subclasses provide ``_compute_coefficients``, ``_reset_state``,
    ``process_sample``, ``coefficients`` and ``frequency_response``; the
    cutoff keyword reported by ``resolve_fc`` is ``cutoff_name``.
    '''

    cutoff_name = 'fc'

    def __init__(self, fs=DEFAULT_FS, fc=None, init_mode=ZERO_STATE,
                 initial_value=None):
        self._fs = validate_fs(fs)
        self._fc = resolve_fc(self._fs, fc, self.cutoff_name)
        self._stored_mode = ZERO_STATE
        self._stored_initial = None
        self._compute_coefficients()
        self._reset_state(ZERO_STATE, None)
        self.reset(init_mode, initial_value)

    # -- configuration ----------------------------------------------------

    @property
    def fs(self):
        '''Sampling frequency, in Hz.'''
        return self._fs

    @property
    def fc(self):
        '''Configured cutoff frequency, in Hz.'''
        return self._fc

    def reconfigure(self, fs=None, fc=None):
        '''
        Change ``fs`` and/or ``fc``, recompute the coefficients and reset.

        A parameter left as ``None`` keeps its current value.  The coefficients
        are never changed in the middle of a data stream: every call resets the
        state, and the initialisation mode is kept.
        '''
        new_fs = self._fs if fs is None else validate_fs(fs)
        candidate = self._fc if fc is None else fc
        new_fc = validate_fc(new_fs, candidate, self.cutoff_name)
        self._fs, self._fc = new_fs, new_fc
        self._compute_coefficients()
        self._reset_state(self._stored_mode, self._stored_initial)
        return self

    # -- state ------------------------------------------------------------

    def reset(self, init_mode=None, initial_value=None):
        '''
        Clear the history and restart from the requested initialisation.

        Without arguments the mode and value of the last initialisation are
        reused.  ``fs``, ``fc`` and the coefficients are not touched.
        '''
        if init_mode is None:
            init_mode = self._stored_mode
            initial_value = self._stored_initial
        mode, value = check_initialisation(init_mode, initial_value)
        self._stored_mode, self._stored_initial = mode, value
        self._reset_state(mode, value)
        return self

    @property
    def init_mode(self):
        '''Initialisation mode used by the last ``reset``.'''
        return self._stored_mode

    # -- sequence ---------------------------------------------------------

    def process_array(self, samples):
        '''
        Filter a whole sequence, sample by sample.

        The internal state advances exactly as for repeated ``process_sample``
        calls, so the result is bit identical to a loop and the filter stays
        causal: the output at index ``k`` never depends on ``samples[k+1:]``.
        '''
        values = check_sequence(samples)
        output = np.empty(values.shape[0], dtype=np.float64)
        for index in range(values.shape[0]):
            output[index] = self.process_sample(values[index])
        return output

    # -- hooks ------------------------------------------------------------

    def _compute_coefficients(self):
        raise NotImplementedError

    def _reset_state(self, mode, value):
        raise NotImplementedError

    def process_sample(self, a_raw):
        raise NotImplementedError

    @property
    def coefficients(self):
        raise NotImplementedError

    def frequency_response(self, frequencies):
        raise NotImplementedError

    def __repr__(self):
        return (f'{type(self).__name__}(fs={self._fs:g}, fc={self._fc:g}, '
                f'init_mode={self._stored_mode!r})')
