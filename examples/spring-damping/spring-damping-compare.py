'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys

sys.path.append('.')
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

import util

# matplotlib.use('TkAgg')


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Analysis data file with fitting.')

    parser.add_argument('--file',
                        type=str,
                        help='Data file to analysis.',
                        default='data/20240729-103252/simulation-result.json')

    return parser.parse_args()


def get_disp(file_name, color=None, linestyle='-', label=''):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    mass_block_data = data['mass_block_state']
    return util.Signal(mass_block_data['position'],
                       t=mass_block_data['time'],
                       color=color,
                       linestyle=linestyle,
                       label=label)


if __name__ == '__main__':
    args = argue_parser()

    with open(args.file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    param = data['parameters']
    json.dump(param, sys.stdout, indent=2)
    # print(data['parameters'])

    m = param['mass']
    b = param['damping_coef']
    k = param['spring_coef']
    bm = b / m
    km = k / m

    w_n = np.sqrt(km)
    f_n = w_n / (2 * np.pi)
    zeta = bm / (2 * w_n)

    # print(f'Natural frequency: {f_n} Hz')
    # print(f'Damping ratio: {zeta}')

    mass_block_data = data['mass_block_state']
    t = np.array(mass_block_data['time'])

    def model(t, racio, decay, omega, phi):
        return racio * np.exp(-decay * t) * np.sin(omega * t + phi)

    racio = 1 / (w_n * np.sqrt(1 - zeta**2))
    decay = zeta * w_n
    omega = w_n * np.sqrt(1 - zeta**2)
    phi = 0
    initial_params = [racio, decay, omega, phi]

    analy = util.Signal(model(t, *initial_params),
                        t=t,
                        color='red',
                        linestyle='--',
                        label='Analytical result')

    simul_list = [
        get_disp('data/20240729-103252/simulation-result.json',
                 color='blue',
                 linestyle=':',
                 label='Rectangle'),
        get_disp('data/20240729-094902/simulation-result.json',
                 color='green',
                 linestyle='-',
                 label='Ideal pulse'),
    ]

    ax_time = None
    ax_power, ax_phase = None, None

    for sig in simul_list:
        ax_time = sig.plot_time_domain(ax_time)
        ax_power, ax_phase = sig.plot_freq_domain(ax_power=ax_power,
                                                  ax_phase=ax_phase)

    ax_time = analy.plot_time_domain(ax_time)
    ax_power, ax_phase = analy.plot_freq_domain(ax_power=ax_power,
                                                ax_phase=ax_phase)

    plt.show()
