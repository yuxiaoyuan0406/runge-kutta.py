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
        description='Analysis data file with fitting.')

    parser.add_argument('--data', type=str, help='Data directory to analysis.')

    return parser.parse_args()


if __name__ == '__main__':
    args = argue_parser()

    with open(os.path.join(args.data, 'param.json'), 'r',
              encoding='utf-8') as f:
        param = json.load(f)
        f.close()
    json.dump(param, sys.stdout, indent=2)

    m = param['mass']
    b = param['damping_coef']
    k = param['spring_coef']
    bm = b / m
    km = k / m

    w_n = np.sqrt(km)
    f_n = w_n / (2 * np.pi)
    zeta = bm / (2 * w_n)

    print(f'Natural frequency: {f_n} Hz')
    print(f'Damping ratio: {zeta}')

    mass_block_data = os.path.join(args.data, 'mass_block')
    t = np.load(os.path.join(mass_block_data, 'time.npy'))
    dt = t[1] - t[0]
    disp = np.load(os.path.join(mass_block_data, 'position.npy'))
    velo = np.load(os.path.join(mass_block_data, 'velocity.npy'))

    def model(t, racio, decay, omega, phi, offset):
        return racio * np.exp(-decay * t) * np.sin(omega * t + phi) + offset

    racio = 1 / (w_n * np.sqrt(1 - zeta**2))
    decay = zeta * w_n
    omega = w_n * np.sqrt(1 - zeta**2)
    phi = 0
    offset = 0
    initial_params = [racio, decay, omega, phi, offset]
    print(
        'Analytical solution of unit impulse response is:\n %.5e * exp(- %.5e * t) * sin(%.5e * t + %.5e) + %.5e'
        % tuple(initial_params))
    popt, pcov = curve_fit(model, t, disp, p0=initial_params)
    print(
        'Simulation of unit impulse response is:\n %.5e * exp(- %.5e * t) * sin(%.5e * t + %.5e) + %.5e'
        % tuple(popt))

    simul = util.Signal(disp, t=t, color='blue', linestyle='-', label='Simulation data')
    analy = util.Signal(model(t, *initial_params), t=t, color='red', linestyle='--', label='Analytical result')

    util.Signal.plot_all()
