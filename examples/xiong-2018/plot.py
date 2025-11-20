'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import os

sys.path.append('.')
import util

# matplotlib.use('TkAgg')


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

    mass_block_data = os.path.join(args.data, 'mass_block')
    pid_data = os.path.join(args.data, 'pid')
    t = np.load(os.path.join(mass_block_data, 'time.npy'))
    dt = t[1] - t[0]
    disp = np.load(os.path.join(mass_block_data, 'position.npy'))
    velo = np.load(os.path.join(mass_block_data, 'velocity.npy'))

    pid_time = np.load(os.path.join(pid_data, 'time.npy'))
    pid  = np.load(os.path.join(pid_data, 'pid.npy'))

    disp = util.Signal(disp,
                       t=t,
                       color='blue',
                       linestyle='-',
                       label='Simulation displacement')
    velo = util.Signal(velo,
                       t=t,
                       color='green',
                       linestyle=':',
                       label='Simulation velocity')
    pid  = util.Signal(pid,
                       t=pid_time,
                       color='red',
                       linestyle='-',
                       label='Simulation PID')

    # util.Signal.plot_all([disp, velo], title='Simulation data', block=True)

    fig_time, ax_time = util.default_time_plot_fig()
    fig_freq, (ax_power, ax_phase) = util.default_freq_plot_fig()

    util.Signal.plot_all(lst=[disp,], ax_time=ax_time, ax_power=ax_power, ax_phase=ax_phase, block=False)

    fig_time.savefig(os.path.join(args.data, 'disp-time.png'),
                     bbox_inches='tight', dpi=300)
    fig_freq.savefig(os.path.join(args.data, 'disp-freq.png'),
                     bbox_inches='tight', dpi=300)

    fig_time, ax_time = util.default_time_plot_fig()
    fig_freq, (ax_power, ax_phase) = util.default_freq_plot_fig()

    util.Signal.plot_all(lst=[velo,], ax_time=ax_time, ax_power=ax_power, ax_phase=ax_phase, block=False)

    fig_time.savefig(os.path.join(args.data, 'velo-time.png'),
                     bbox_inches='tight', dpi=300)
    fig_freq.savefig(os.path.join(args.data, 'velo-freq.png'),
                     bbox_inches='tight', dpi=300)

    fig_time, ax_time = util.default_time_plot_fig()
    fig_freq, (ax_power, ax_phase) = util.default_freq_plot_fig()

    util.Signal.plot_all(lst=[pid,], ax_time=ax_time, ax_power=ax_power, ax_phase=ax_phase)

    fig_time.savefig(os.path.join(args.data, 'pid-time.png'),
                     bbox_inches='tight', dpi=300)
    fig_freq.savefig(os.path.join(args.data, 'pid-freq.png'),
                     bbox_inches='tight', dpi=300)
