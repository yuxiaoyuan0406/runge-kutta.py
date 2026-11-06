'''
Convert a MATLAB ``.mat`` file into a tree of ``.npy`` arrays.

Two kinds of input are supported:

* a plain MAT file (numeric arrays, structs, char and cell content).  It is
  read with :mod:`scipy.io` and written straight to ``.npy``;
* a Simulink ``SimulationOutput``.  Such an object is stored as an MCOS
  object, which ``scipy.io.loadmat`` can only identify, not decode, so the
  script asks MATLAB (batch mode) for one plain intermediate file first and
  then converts that.

Every run also writes ``manifest.json`` with the shape, dtype and sampling
period of each signal, so the result can be checked without opening the
arrays.

Usage::

    python mat_to_npy.py matlab_data/unit_step_1/out.mat -o data/unit_step_1
    python mat_to_npy.py out.mat -o out --layout flat
    python mat_to_npy.py out.mat -o out --exclude tout,logsout
'''
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io


class ConversionError(RuntimeError):
    '''
    Raised when the file cannot be converted without outside help.
    '''


# Signals of the closed loop example that map onto the repository layout.
REPO_LAYOUT = {
    'pid/time.npy': ('mems_out', 'time'),
    'pid/pid.npy': ('mems_out', 'data'),
    'mass_block/time.npy': ('a_true', 'time'),
    'mass_block/acceleration.npy': ('a_true', 'data'),
}

INTERMEDIATE_PREFIX = 'mat2npy_'


def find_matlab(explicit=None):
    '''
    Locate the MATLAB executable.

    Args:
        explicit: Path given on the command line or in ``MATLAB_EXE``.

    Returns:
        Optional[str]: Path of ``matlab`` or ``matlab.exe``, ``None`` if absent.
    '''
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get('MATLAB_EXE')
    if env:
        candidates.append(env)
    found = shutil.which('matlab')
    if found:
        candidates.append(found)

    if os.name == 'nt':
        patterns = [r'C:\Program Files\MATLAB\R*\bin\matlab.exe',
                    r'C:\Program Files (x86)\MATLAB\R*\bin\matlab.exe',
                    r'D:\Program Files\MATLAB\R*\bin\matlab.exe',
                    r'D:\MATLAB\R*\bin\matlab.exe']
    else:
        patterns = ['/usr/local/MATLAB/R*/bin/matlab',
                    '/opt/MATLAB/R*/bin/matlab',
                    '/Applications/MATLAB_R*.app/bin/matlab']
    for pattern in patterns:
        candidates.extend(sorted(glob.glob(pattern), reverse=True))
    candidates.extend(_matlab_from_registry())

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _matlab_from_registry():
    '''
    Read the MATLAB installation root from the Windows registry.
    '''
    if os.name != 'nt':
        return []
    try:
        import winreg
    except ImportError:
        return []

    roots = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, r'SOFTWARE\MathWorks\MATLAB') as key:
                index = 0
                while True:
                    try:
                        version = winreg.EnumKey(key, index)
                    except OSError:
                        break
                    index += 1
                    try:
                        with winreg.OpenKey(key, version) as sub:
                            root, _ = winreg.QueryValueEx(sub, 'MATLABROOT')
                            roots.append((version, root))
                    except OSError:
                        continue
        except OSError:
            continue

    paths = []
    for _, root in sorted(roots, reverse=True):
        paths.append(os.path.join(root, 'bin', 'matlab.exe'))
    return paths


def _is_opaque(value):
    '''
    True when scipy returned a MATLAB object it cannot decode.
    '''
    return type(value).__name__ == 'MatlabOpaque'


def _mcos_names(loaded):
    '''
    List the top level variables that scipy decoded as MCOS objects.

    scipy names such a variable ``None`` because the MCOS header hides the real
    name; the name is kept in the opaque object itself as ``s0``.
    '''
    names = []
    for name, value in loaded.items():
        if name.startswith('__') or not _is_opaque(value):
            continue
        names.append(_opaque_name(value) or name)
    return names


def _opaque_name(value):
    '''
    Recover the variable name stored inside a MatlabOpaque.
    '''
    try:
        record = value.ravel()[0]
        raw = record['s0']
        if isinstance(raw, np.ndarray):
            raw = raw.ravel()[0]
        if isinstance(raw, bytes):
            return raw.decode('utf-8', 'replace')
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return None


def _matlab_export_script() -> str:
    '''
    MATLAB function that flattens a MAT file into ``<name>_data`` variables.

    ``Simulink.SimulationOutput`` does not answer ``fieldnames``, it exposes
    its logged signals through ``who``; ``timeseries`` objects carry their own
    ``Data`` and ``Time``.  Everything is written as a double array so the
    Python side needs no MATLAB type knowledge.
    '''
    return r'''
function mat2npy_export(infile, outfile, skip)
% Flatten a MAT file into plain arrays for the Python converter.
if nargin < 3
    skip = {};
end
S = load(infile);
collector = struct();
names = fieldnames(S);
for i = 1:numel(names)
    collector = mat2npy_add(collector, mat2npy_safe(names{i}), S.(names{i}), skip);
end
save(outfile, '-struct', 'collector', '-v7');
end

function name = mat2npy_safe(name)
name = regexprep(name, '[^A-Za-z0-9_]', '_');
if isempty(name) || ~isletter(name(1))
    name = ['s_' name];
end
end

function c = mat2npy_add(c, name, v, skip)
if any(strcmp(name, skip))
    return
end
if isa(v, 'timeseries')
    c.([name '_data']) = double(v.Data);
    c.([name '_time']) = double(v.Time);
elseif isa(v, 'Simulink.SimulationOutput')
    subs = cellstr(string(who(v)));
    for j = 1:numel(subs)
        c = mat2npy_add(c, mat2npy_safe(subs{j}), v.(subs{j}), skip);
    end
elseif isstruct(v) && isscalar(v)
    f = fieldnames(v);
    for j = 1:numel(f)
        c = mat2npy_add(c, [name '_' mat2npy_safe(f{j})], v.(f{j}), skip);
    end
elseif isnumeric(v) && ~isempty(v)
    c.([name '_data']) = double(v);
end
end
'''


def export_with_matlab(path, matlab, workdir, exclude=()):
    '''
    Run the MATLAB helper and return the path of the plain intermediate file.

    Args:
        path: Input MAT file.
        matlab: MATLAB executable.
        workdir: Directory for the generated helper and the intermediate.
        exclude: Signal names MATLAB should not export.  Useful when one of the
            logged signals runs at the solver base rate and would dominate both
            the intermediate file and the output.

    Returns:
        str: Path of the plain intermediate MAT file.

    Raises:
        ConversionError: When MATLAB fails.
    '''
    helper = os.path.join(workdir, 'mat2npy_export.m')
    with open(helper, 'w', encoding='utf-8') as handle:
        handle.write(_matlab_export_script())

    intermediate = os.path.join(workdir, 'mat2npy_plain.mat')
    source = os.path.abspath(path).replace('\\', '/').replace("'", "''")
    target = intermediate.replace('\\', '/').replace("'", "''")
    folder = workdir.replace('\\', '/').replace("'", "''")
    skip = ', '.join(f"'{name}'" for name in exclude)
    command = (f"addpath('{folder}'); mat2npy_export('{source}', '{target}', "
               f"{{{skip}}});")

    print(f'running MATLAB to unpack the Simulink object: {matlab}')
    completed = subprocess.run([matlab, '-batch', command],
                               capture_output=True, text=True)
    if completed.returncode != 0 or not os.path.isfile(intermediate):
        tail = (completed.stderr or completed.stdout or '').strip()
        raise ConversionError(
            'MATLAB could not export the file.  Output:\n' + tail)
    return intermediate


def flatten(loaded, prefix=''):
    '''
    Flatten a scipy result into ``name -> array`` pairs.

    Structs become dotted names, char arrays are kept as text, empty values are
    skipped.  MCOS objects are reported by :func:`_mcos_names` instead.
    '''
    flattened = {}
    for name, value in loaded.items():
        if name.startswith('__'):
            continue
        flattened.update(_flatten_value(value, f'{prefix}{name}'))
    return flattened


def _flatten_value(value, full):
    '''
    Recursively flatten one value of a MAT file.
    '''
    result = {}
    if isinstance(value, np.ndarray) and value.dtype.names:
        # A struct comes back as a record array; scipy keeps it 1x1.
        if value.size == 0:
            return result
        record = value.ravel()[0]
        fields = value.dtype.names
        lower = {field.lower(): field for field in fields}
        if 'data' in lower and 'time' in lower:
            # MATLAB's signal convention: one signal with its own time base.
            result.update(_flatten_value(record[lower['data']], f'{full}_data'))
            result.update(_flatten_value(record[lower['time']], f'{full}_time'))
            return result
        for field in fields:
            result.update(_flatten_value(record[field], f'{full}.{field}'))
        return result

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return result
        if value.dtype.kind in 'US':
            result[full] = np.asarray(value)
        elif value.dtype.kind == 'O':
            for item in value.ravel():
                result.update(_flatten_value(item, full))
        else:
            result[full] = np.asarray(value, dtype=np.float64)
        return result

    if isinstance(value, (int, float)):
        result[full] = np.asarray([value], dtype=np.float64)
    return result


def _split_signals(flat):
    '''
    Group ``<name>_data`` and ``<name>_time`` into signal records.
    '''
    signals = {}
    for name, array in flat.items():
        if name.endswith('_data'):
            signals.setdefault(name[:-5], {})['data'] = array
        elif name.endswith('_time'):
            signals.setdefault(name[:-5], {})['time'] = array
        else:
            signals.setdefault(name, {})['data'] = array
    return {name: item for name, item in signals.items() if 'data' in item}


def _load(path, helper_dir=None, exclude=()):
    '''
    Load a MAT file, unpacking a Simulink object when necessary.

    Args:
        path: Input MAT file.
        helper_dir: Where to put the fallback MATLAB helper when MATLAB is
            missing.  Defaults to the directory of the input file.
        exclude: Signal names to skip while unpacking.
    '''
    loaded = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=True,
                              mat_dtype=False)
    opaque = _mcos_names(loaded)
    if not opaque:
        return flatten(loaded), None

    matlab = find_matlab()
    if matlab is None:
        directory = helper_dir or os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        helper = os.path.join(directory, 'mat_to_npy_export.m')
        with open(helper, 'w', encoding='utf-8') as handle:
            handle.write(_matlab_export_script())
        raise ConversionError(
            f'`{os.path.basename(path)}` stores {", ".join(opaque)} as a '
            'Simulink object (MCOS), which scipy cannot decode, and MATLAB was '
            'not found.\n'
            'Either pass --matlab-exe, or export it once with MATLAB:\n'
            f"  matlab -batch \"addpath('{os.path.dirname(helper)}'); "
            f"mat2npy_export('{path.replace(chr(92), '/')}', 'plain.mat');\"\n"
            f'A ready to use helper was written to {helper}.')

    workdir = tempfile.mkdtemp(prefix='mat2npy-')
    try:
        intermediate = export_with_matlab(path, matlab, workdir, exclude)
        loaded = scipy.io.loadmat(intermediate, squeeze_me=False,
                                  struct_as_record=True, mat_dtype=False)
        return flatten(loaded), matlab
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _as_vector(array: np.ndarray) -> np.ndarray:
    '''
    Turn a one column or one row signal into a 1-D array.

    The simulation writes 1-D ``.npy`` files, so the converted signals keep the
    same shape instead of staying ``(N, 1)``.
    '''
    array = np.asarray(array)
    if array.ndim == 2 and min(array.shape) == 1 and array.size > 1:
        return array.ravel()
    return array


def write_arrays(signals, outdir, layout):
    '''
    Write the conversion result.

    Args:
        signals: ``name -> {'data': array, 'time': array or None}``.
        outdir: Output directory.
        layout: ``repo`` writes the simulation layout as well, ``flat`` keeps
            only one file per signal.

    Returns:
        Dict[str, object]: The manifest.
    '''
    os.makedirs(outdir, exist_ok=True)
    rawdir = os.path.join(outdir, 'raw')
    os.makedirs(rawdir, exist_ok=True)

    entries = {}
    for name, item in sorted(signals.items()):
        data = _as_vector(np.asarray(item['data']))
        path = os.path.join(rawdir, f'{name}.npy')
        np.save(path, data)
        entry = {
            'file': os.path.relpath(path, outdir).replace('\\', '/'),
            'shape': list(data.shape),
            'dtype': str(data.dtype),
        }
        time = item.get('time')
        if time is not None:
            time = np.asarray(time, dtype=np.float64).ravel()
            time_path = os.path.join(rawdir, f'{name}_time.npy')
            np.save(time_path, time)
            entry['time_file'] = os.path.relpath(time_path, outdir).replace('\\', '/')
            if time.size > 1:
                steps = np.diff(time)
                entry['samples'] = int(time.size)
                entry['dt'] = float(np.median(steps))
                entry['fs'] = float(1. / np.median(steps)) if np.median(steps) else None
                entry['uniform'] = bool(np.allclose(steps, np.median(steps),
                                                    rtol=1e-9, atol=0.0))
        entries[name] = entry

    mapped = {}
    if layout == 'repo':
        for relative, (signal, part) in REPO_LAYOUT.items():
            item = signals.get(signal)
            if not item or part not in item or item[part] is None:
                continue
            target = os.path.join(outdir, *relative.split('/'))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            np.save(target, _as_vector(np.asarray(item[part])))
            mapped[relative] = signal

    manifest = {
        'source': None,
        'layout': layout,
        'signals': entries,
        'repo_layout': mapped,
    }
    with open(os.path.join(outdir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def convert(path, outdir, layout='repo', exclude=(), matlab=None):
    '''
    Convert one MAT file.

    Args:
        path: Input MAT file.
        outdir: Output directory.
        layout: ``repo`` or ``flat``.
        exclude: Signal names to skip.
        matlab: Explicit MATLAB executable, optional.

    Returns:
        Dict[str, object]: The manifest.
    '''
    if matlab:
        os.environ['MATLAB_EXE'] = matlab
    flat, used_matlab = _load(path, helper_dir=outdir, exclude=tuple(exclude))
    signals = _split_signals(flat)

    for name in exclude:
        signals.pop(name, None)

    if not signals:
        raise ConversionError(f'{path} contains no numeric signal')

    manifest = write_arrays(signals, outdir, layout)
    manifest['source'] = os.path.abspath(path)
    manifest['matlab'] = used_matlab
    with open(os.path.join(outdir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def argue_parser() -> argparse.ArgumentParser:
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Convert a MATLAB MAT file into .npy arrays.')
    parser.add_argument('input', type=str, help='Input .mat file.')
    parser.add_argument('-o', '--out', type=str, default=None,
                        help='Output directory, defaults to <input>_npy.')
    parser.add_argument('--layout', choices=('repo', 'flat'), default='repo',
                        help='repo also writes the pid/ and mass_block/ layout.')
    parser.add_argument('--exclude', type=str, default='',
                        help='Comma separated signal names to skip.')
    parser.add_argument('--matlab-exe', type=str, default=None,
                        help='MATLAB executable used to unpack Simulink objects.')
    parser.add_argument('--list', action='store_true',
                        help='Only print the signals that were found.')
    return parser


def main(argv=None) -> int:
    '''
    Command line entry point.
    '''
    args = argue_parser().parse_args(argv)
    if not os.path.isfile(args.input):
        print(f'input not found: {args.input}', file=sys.stderr)
        return 2

    outdir = args.out or f'{args.input}_npy'
    exclude = [item.strip() for item in args.exclude.split(',') if item.strip()]

    try:
        if args.list:
            flat, _ = _load(args.input)
            for name, item in sorted(_split_signals(flat).items()):
                data = np.asarray(item['data'])
                print(f'{name:24s} shape={data.shape} dtype={data.dtype}')
            return 0

        manifest = convert(args.input, outdir, args.layout, exclude, args.matlab_exe)
    except ConversionError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1

    print(f'converted {len(manifest["signals"])} signals into {outdir}')
    for name, entry in manifest['signals'].items():
        dt = entry.get('dt')
        suffix = f', dt={dt:.6e} s' if dt else ''
        print(f'  {name:24s} {entry["shape"]} {entry["dtype"]}{suffix}')
    if manifest['repo_layout']:
        print('repository layout:')
        for relative, signal in manifest['repo_layout'].items():
            print(f'  {relative:32s} <- {signal}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
