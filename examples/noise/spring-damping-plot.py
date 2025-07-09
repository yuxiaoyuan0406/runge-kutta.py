'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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

    mass_block_data = os.path.join(args.data, 'nonzero_input')
    t = np.load(os.path.join(mass_block_data, 'time.npy'))
    disp = np.load(os.path.join(mass_block_data, 'position.npy'))
    disp = util.Signal(disp,
                       t=t,
                       label='Simulation displacement')

    kalman_data = os.path.join(args.data, 'kalman')
    t_k = np.load(os.path.join(mass_block_data, 'time.npy'))
    disp_k = np.load(os.path.join(mass_block_data, 'position.npy'))
    disp_k = util.Signal(disp_k,
                         t=t_k,
                         linestyle=':',
                         label='Kalman filtered displacement',)

    fig_time, ax_time = plt.subplots(figsize=(16, 9))
    ax_time.grid(True)
    ax_time.legend(loc='upper right')
    fig_freq, (ax_power, ax_phase) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    ax_power.grid(True)
    ax_phase.grid(True)
    ax_power.set_xscale('log')
    ax_phase.set_xscale('log')
    ax_power.set_ylabel('dB')
    ax_power.legend(loc='upper right')
    ax_phase.legend(loc='upper right')

    util.Signal.plot_all(ax_time=ax_time, ax_power=ax_power, ax_phase=ax_phase)

    fig_time.savefig(os.path.join(args.data, 'disp-time.png'),
                     bbox_inches='tight', dpi=300)
    fig_freq.savefig(os.path.join(args.data, 'disp-freq.png'),
                     bbox_inches='tight', dpi=300)

