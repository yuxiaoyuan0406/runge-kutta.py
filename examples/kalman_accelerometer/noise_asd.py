'''
Noise ASD of the PID, Kalman and low pass outputs, in g/sqrt(Hz).

This is the closed loop noise evaluation of the repository's ``util.asd``
module (the Python port of ``bitstreamPSDfn_analogcl``): the *output signals*
themselves are transformed, without subtracting a reference, after converting
them from m/s^2 to g.  No error signal is formed.

The band used for the noise floor excludes the excitation and its harmonics,
because the ASD of the output contains the signal line as well; the default
2000-20000 Hz is above the 125 Hz and 12.5 Hz excitation and their low order
harmonics, and the 7-term Blackman-Harris window keeps the line leakage far
below the floor.

Usage (project root)::

    python examples/kalman_accelerometer/noise_asd.py \\
        --zoh-config examples/kalman_accelerometer/configs/xiong-2018-sine.json \\
        --foh-config examples/kalman_accelerometer/configs/xiong-2018-sine-foh.json \\
        --data data/20261111-125hz-sine-1 --band 2000,20000 \\
        -o data/noise-asd-125hz_sine_1
'''
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import util
from examples.kalman_accelerometer import config as config_module
from examples.kalman_accelerometer import data_adapter
from examples.kalman_accelerometer.compare_priors import pid_acceleration
from examples.kalman_accelerometer.kalman import ClosedLoopKalman
from util.asd import asd_fn, plot_asd
from util.baseline_filters import ButterworthFilter, RCFilter

STANDARD_GRAVITY = 9.80665
'''m/s^2 per g, the conversion used for every signal.'''

COLORS = {
    'PID': 'tab:red',
    'ZOH': 'tab:blue',
    'FOH': 'tab:green',
    'RC': 'tab:orange',
    'Butterworth': 'tab:purple',
}
LINESTYLES = {
    'PID': ':',
    'ZOH': '--',
    'FOH': '-',
    'RC': '-.',
    'Butterworth': (0, (1, 1)),
}


def argue_parser() -> argparse.ArgumentParser:
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Noise ASD (dBg/sqrt(Hz)) of the five method outputs.')
    parser.add_argument('--zoh-config', type=str, required=True)
    parser.add_argument('--foh-config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--rc-fc', type=float, default=1000.0)
    parser.add_argument('--bw-fc', type=float, default=1000.0)
    parser.add_argument('--band', type=str, default='2000,20000',
                        help='Noise floor band in Hz, "fl,fh".')
    parser.add_argument('-o', '--out', type=str, required=True,
                        help='Output directory (figure and JSON).')
    parser.add_argument('--label', type=str, default='',
                        help='Figure label, defaults to the parent directory name.')
    parser.add_argument('--floor-db', type=float, default=-160.0,
                        help='Lower y limit of the ASD figure.')
    parser.add_argument('--top-db', type=float, default=40.0,
                        help='Upper y limit of the ASD figure.')
    parser.add_argument('--dpi', type=int, default=200)
    return parser


def main(argv=None) -> int:
    '''
    Compute the ASD of every method output and write the figure and the JSON.
    '''
    args = argue_parser().parse_args(argv)
    fl, fh = (float(value) for value in args.band.split(','))

    zoh_config = config_module.load_config(args.zoh_config)
    foh_config = config_module.load_config(args.foh_config)
    zoh_config.data_path = args.data
    foh_config.data_path = args.data
    data = data_adapter.load_data(zoh_config)

    zoh = ClosedLoopKalman.initialize(zoh_config).run(data.u)
    foh = ClosedLoopKalman.initialize(foh_config).run(data.u)
    a_pid, conversion = pid_acceleration(zoh_config, data.u)
    rc = RCFilter(fs=zoh_config.fs, fc=args.rc_fc)
    bw = ButterworthFilter(fs=zoh_config.fs, fc=args.bw_fc)
    a_rc = rc.process_array(a_pid)
    a_bw = bw.process_array(a_pid)

    signals = {
        'PID': np.asarray(a_pid, dtype=np.float64),
        'ZOH': np.asarray(zoh['acceleration'], dtype=np.float64),
        'FOH': np.asarray(foh['acceleration'], dtype=np.float64),
        'RC': np.asarray(a_rc, dtype=np.float64),
        'Butterworth': np.asarray(a_bw, dtype=np.float64),
    }

    results = {}
    label = args.label or os.path.basename(os.path.dirname(
        os.path.abspath(args.data).rstrip('/\\'))) or os.path.basename(args.data)
    figure, axis = util.default_noise_plot_fig(title=f'Output ASD, {label}')
    for label, signal in signals.items():
        signal_g = signal / STANDARD_GRAVITY
        freq, asd, noise_floor = asd_fn(signal_g, fl, fh, fs=zoh_config.fs)
        results[label] = {
            'noise_floor_dbg_per_sqrt_hz': float(noise_floor),
            'noise_floor_linear_g_per_sqrt_hz': float(10.0 ** (noise_floor / 20.0)),
            'rms_g': float(np.std(signal_g)),
            'rms_m_s2': float(np.std(signal)),
        }
        plot_asd(freq, asd, None, ax=axis,
                 label=f'{label}  NF {noise_floor:.1f} dBg/$\\sqrt{{Hz}}$',
                 band=(fl, fh), floor_db=args.floor_db, title=None)
        # plot_asd draws one curve per call; colour and style it per method.
        axis.lines[-1].set_color(COLORS[label])
        axis.lines[-1].set_linestyle(LINESTYLES[label])

    axis.set_ylim(args.floor_db, args.top_db)
    axis.set_xlim(1.0, zoh_config.fs / 2.0)
    axis.axvspan(fl, fh, color='gray', alpha=0.06,
                 label=f'noise-floor band {fl:g}-{fh:g} Hz')
    axis.legend(loc='upper left', fontsize=7)
    axis.grid(True, which='both')

    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, 'noise-asd.png')
    figure.tight_layout()
    figure.savefig(png, dpi=args.dpi, bbox_inches='tight')
    plt.close(figure)

    report = {
        'data': data.summary(),
        'samples': int(data.t.size),
        'fs': float(zoh_config.fs),
        'signal_unit': 'g',
        'standard_gravity': STANDARD_GRAVITY,
        'band_hz': [fl, fh],
        'conversion': conversion,
        'filter_cutoffs_hz': {'rc': args.rc_fc, 'bw': args.bw_fc},
        'methods': results,
        'figure': png,
        'note': ('ASD of the output signals themselves (no reference '
                 'subtracted), unit g/sqrt(Hz); the noise floor is the power '
                 'mean of the PSD over the band, in dBg/sqrt(Hz).'),
    }
    with open(os.path.join(args.out, 'noise-asd.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f'{args.data}: {data.t.size} samples, fs={zoh_config.fs:g} Hz, '
          f'band {fl:g}-{fh:g} Hz')
    for label in signals:
        item = results[label]
        print(f'  {label:12s} noise floor {item["noise_floor_dbg_per_sqrt_hz"]:8.2f} '
              f'dBg/sqrt(Hz)  ({item["noise_floor_linear_g_per_sqrt_hz"]:.3e} g/sqrt(Hz))')
    print(f'wrote {png}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
