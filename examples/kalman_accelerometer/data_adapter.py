'''
Data adapter: load the existing simulation files and hand the estimator a
clean ``(t, u)`` sequence.

This module is deliberately isolated from the estimator: the core recursion
only ever sees one scalar ``u[k]``.  Units, column names, truncation and
alignment are resolved here, and every inconsistency is reported instead of
being silently repaired.

Supported layouts
-----------------
``npy_dir``
    ``data_path`` is a directory, ``column_map`` values are paths relative to
    it, for example ``pid/pid.npy`` or ``mass_block/position.npy``.  This is
    the layout written by ``examples/xiong-2018/main.py``.
``npz``
    ``data_path`` is one ``.npz`` archive, ``column_map`` values are array
    keys inside it.
'''
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


class DataError(ValueError):
    '''
    Raised when the input data is missing, misaligned or inconsistent.
    '''


REQUIRED_COLUMNS = ('t', 'u')
OPTIONAL_COLUMNS = ('a_true', 'x_true', 'v_true', 'e_debug')

DEFAULT_COLUMN_MAP = {
    't': 'pid/time.npy',
    'u': 'pid/pid.npy',
    'x_true': 'mass_block/position.npy',
    'v_true': 'mass_block/velocity.npy',
}


@dataclass
class SimulationData:
    '''
    One aligned, unit converted measurement sequence.

    Attributes:
        t: Sample times in seconds, strictly increasing, shape ``(N,)``.
        u: PID outputs ``u[k]``, shape ``(N,)``.
        a_true: Optional true external acceleration, shape ``(N,)``.
        x_true: Optional true displacement, shape ``(N,)``.
        v_true: Optional true velocity, shape ``(N,)``.
        e_debug: Optional debug dump of the PID error ``e[k]``.
        dt: Sampling period taken from the data.
        source: Human readable description of where the data came from.
        warnings: Everything that was reported but did not stop the load.
    '''

    t: np.ndarray
    u: np.ndarray
    a_true: Optional[np.ndarray] = None
    x_true: Optional[np.ndarray] = None
    v_true: Optional[np.ndarray] = None
    e_debug: Optional[np.ndarray] = None
    dt: float = 0.
    source: str = ''
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.t.size)

    def window(self, start: float, end: float) -> 'SimulationData':
        '''
        Slice the sequence by time, end inclusive.

        Args:
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            SimulationData: A view with the same fields restricted to the window.
        '''
        if start > end:
            raise DataError(f'empty window: start {start} > end {end}')
        mask = (self.t >= start) & (self.t <= end)
        if not np.any(mask):
            raise DataError(f'no samples inside window [{start}, {end}] s')
        return self._select(mask)

    def index_window(self, first: int, last: int) -> 'SimulationData':
        '''
        Slice the sequence by sample index, end inclusive.
        '''
        if first < 0 or last >= len(self) or first > last:
            raise DataError(f'invalid index window [{first}, {last}] for {len(self)} samples')
        mask = np.zeros(len(self), dtype=bool)
        mask[first:last + 1] = True
        return self._select(mask)

    def _select(self, mask: np.ndarray) -> 'SimulationData':
        '''
        Apply a boolean mask to every present column.
        '''
        def pick(array):
            return None if array is None else array[mask]

        t = self.t[mask]
        dt = float(np.mean(np.diff(t))) if t.size > 1 else self.dt
        return SimulationData(
            t=t,
            u=pick(self.u),
            a_true=pick(self.a_true),
            x_true=pick(self.x_true),
            v_true=pick(self.v_true),
            e_debug=pick(self.e_debug),
            dt=dt,
            source=self.source,
            warnings=list(self.warnings),
        )

    def summary(self) -> Dict[str, object]:
        '''
        JSON friendly description of the loaded sequence.
        '''
        return {
            'source': self.source,
            'samples': len(self),
            'dt': self.dt,
            'fs': 1. / self.dt if self.dt else None,
            'duration': float(self.t[-1] - self.t[0]) if len(self) else 0.,
            'has_a_true': self.a_true is not None,
            'has_x_true': self.x_true is not None,
            'has_v_true': self.v_true is not None,
            'warnings': list(self.warnings),
        }


def _load_column(data_path: str, data_format: str, location: str) -> np.ndarray:
    '''
    Read one column from the configured layout.
    '''
    if data_format == 'npy_dir':
        path = os.path.join(data_path, location)
        if not os.path.isfile(path):
            raise DataError(f'column file not found: {path}')
        return np.load(path)

    if data_format == 'npz':
        with np.load(data_path) as archive:
            if location not in archive:
                raise DataError(f'column "{location}" not found in {data_path}')
            return archive[location]

    raise DataError(f'unsupported data_format "{data_format}", '
                    "use 'npy_dir' or 'npz'")


def _apply_alignment(config, data: SimulationData) -> SimulationData:
    '''
    Apply the configured offset, truncation and time shift.
    '''
    # Every present column has to agree on the sample count before any slice,
    # otherwise a boolean mask would fail with an unrelated IndexError.
    lengths = {'t': data.t.size}
    for name in ('u', 'a_true', 'x_true', 'v_true', 'e_debug'):
        array = getattr(data, name)
        if array is not None:
            lengths[name] = array.size
    if len(set(lengths.values())) > 1:
        raise DataError(f'the columns disagree on the sample count: {lengths}')

    alignment = config.sample_alignment or {}
    offset = int(alignment.get('offset', 0) or 0)
    length = alignment.get('length')
    time_offset = float(alignment.get('time_offset', 0.) or 0.)

    if offset < 0:
        raise DataError(f'sample_alignment.offset must not be negative, got {offset}')
    if offset >= len(data):
        raise DataError(f'sample_alignment.offset {offset} is beyond the {len(data)} samples')

    last = len(data) - 1 if length is None else offset + int(length) - 1
    if last >= len(data):
        raise DataError(f'sample_alignment.length {length} exceeds the available samples')

    trimmed = data.index_window(offset, last)
    if time_offset:
        trimmed.t = trimmed.t + time_offset
    return trimmed


def _check_timebase(config, data: SimulationData) -> List[str]:
    '''
    Verify finiteness, monotonicity and the sampling period against ``Ts``.
    '''
    warnings: List[str] = []
    for name in ('t', 'u', 'a_true', 'x_true', 'v_true', 'e_debug'):
        array = getattr(data, name)
        if array is None:
            continue
        if array.ndim != 1:
            raise DataError(f'column "{name}" must be one dimensional, got shape {array.shape}')
        if array.shape != data.t.shape:
            raise DataError(f'column "{name}" has {array.size} samples, '
                            f'time has {data.t.size}')
        if not np.all(np.isfinite(array)):
            raise DataError(f'column "{name}" contains non finite values')

    if data.t.size < 2:
        raise DataError('at least two samples are required')

    steps = np.diff(data.t)
    if np.any(steps <= 0):
        raise DataError('time column is not strictly increasing')

    dt = float(np.mean(steps))
    data.dt = dt
    spread = float(np.max(np.abs(steps - dt)) / dt)
    if spread > 1e-6:
        raise DataError(f'the sampling period is not uniform (relative spread {spread:.3e})')

    expected = 1. / config.fs
    mismatch = abs(dt - expected) / expected
    if mismatch > 1e-3:
        raise DataError(
            f'data sampling period {dt:.9e} s disagrees with the configured '
            f'fs = {config.fs:.9e} Hz (Ts = {expected:.9e} s); fix the '
            'configuration or the sample_alignment instead of rescaling the model')
    if mismatch > 1e-9:
        warnings.append(f'data dt {dt:.12e} s differs slightly from the configured '
                        f'Ts {expected:.12e} s (relative {mismatch:.3e})')
    return warnings


def load_data(config) -> SimulationData:
    '''
    Load the measurement sequence described by the configuration.

    Args:
        config: A :class:`config.Config` instance with ``data_path``,
            ``data_format`` and ``column_map`` filled in.

    Returns:
        SimulationData: Aligned and unit converted columns.

    Raises:
        DataError: When a required column is missing, the time base is broken
            or the sampling period contradicts the configuration.
    '''
    if not config.data_path:
        raise DataError('data_path is not configured')

    column_map: Dict[str, str] = dict(DEFAULT_COLUMN_MAP)
    column_map.update(config.column_map or {})

    missing = [key for key in REQUIRED_COLUMNS if key not in column_map]
    if missing:
        raise DataError(f'column_map is missing required entries: {missing}')

    units = config.units or {}
    arrays = {}
    skipped = []
    for name in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        location = column_map.get(name)
        if not location:
            continue
        try:
            raw = _load_column(config.data_path, config.data_format, location)
        except DataError:
            # A missing required column is fatal, a missing optional one is not:
            # a run may well provide the acceleration truth and no displacement.
            if name in REQUIRED_COLUMNS:
                raise
            skipped.append(name)
            continue
        values = np.asarray(raw, dtype=np.float64).ravel()
        factor = float(units.get(name, 1.))
        arrays[name] = values * factor

    data = SimulationData(
        t=arrays['t'],
        u=arrays['u'],
        a_true=arrays.get('a_true'),
        x_true=arrays.get('x_true'),
        v_true=arrays.get('v_true'),
        e_debug=arrays.get('e_debug'),
        source=f'{config.data_path} ({config.data_format})',
    )
    if skipped:
        data.warnings.append(f'optional columns not found and skipped: {skipped}')

    data = _apply_alignment(config, data)
    data.warnings.extend(_check_timebase(config, data))
    return data


def describe_layout(config) -> Dict[str, object]:
    '''
    Report which columns are available without loading the arrays.
    '''
    column_map: Dict[str, str] = dict(DEFAULT_COLUMN_MAP)
    column_map.update(config.column_map or {})
    available = {}
    for name, location in column_map.items():
        if config.data_format == 'npy_dir':
            available[name] = os.path.isfile(os.path.join(config.data_path or '', location))
        else:
            available[name] = None
    return {'data_path': config.data_path, 'data_format': config.data_format,
            'columns': column_map, 'available': available}
