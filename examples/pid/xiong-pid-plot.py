'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys
import h5py

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import os

sys.path.append('.')
import util


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Plot simulation data. ')

    parser.add_argument('--data', type=str, help='Data directory to analysis.')

    return parser.parse_args()


if __name__ == '__main__':
    args = argue_parser()
    data_dir = args.data

    pid_data_path = os.path.join(data_dir, 'pid')
    pid_time = np.load(os.path.join(pid_data_path, 'time.npy'))
    pid_out = np.load(os.path.join(pid_data_path, 'pid.npy'))

    matlab_data_path = os.path.join(data_dir, 'matlab', 'pid.h5')
    with h5py.File(matlab_data_path, "r") as f:
        matlab_time = np.ravel(f["/t"][...])
        matlab_out = np.array(f["/y"][...])
        matlab_time = np.asarray(matlab_time).reshape(-1)
        matlab_out = np.asarray(matlab_out).reshape(-1)
        
    pid_out = util.Signal(pid_out,
                       t=pid_time,
                       color='blue',
                       linestyle='-',
                       label='My PID')
    matlab_out = util.Signal(matlab_out,
                       t=matlab_time,
                       color='green',
                       linestyle=':',
                       label='MATLAB')

    # util.Signal.plot_all([disp, velo], title='Simulation data', block=True)

    fig_time, ax_time = util.default_time_plot_fig()
    fig_freq, (ax_power, ax_phase) = util.default_freq_plot_fig()

    util.Signal.plot_all(lst=[pid_out, matlab_out], ax_time=ax_time, ax_power=ax_power, ax_phase=ax_phase, block=True)

    fig_time.savefig(os.path.join(args.data, 'pid-time.png'),
                     bbox_inches='tight', dpi=300)
    fig_freq.savefig(os.path.join(args.data, 'pid-freq.png'),
                     bbox_inches='tight', dpi=300)

