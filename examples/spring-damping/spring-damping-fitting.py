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
    inpulse_params = [racio, decay, omega, phi, offset]
    step_params = [
        -racio / w_n, decay, omega,
        np.arctan2(omega, decay), 1 / (omega**2)
    ]

    popt, pcov = curve_fit(model, t, disp, p0=inpulse_params)

    print(
        'Analytical solution of unit impulse response is:\n %.5e * exp(- %.5e * t) * sin(%.5e * t + %.5e) + %.5e'
        % tuple(inpulse_params))
    print(
        'Analytical solution of unit step response is:\n %.5e * exp(- %.5e * t) * sin(%.5e * t + %.5e) + %.5e'
        % tuple(step_params))
    print(
        'Simulation response is:\n %.5e * exp(- %.5e * t) * sin(%.5e * t + %.5e) + %.5e'
        % tuple(popt))

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
    imp_rsp = util.Signal(model(t, *inpulse_params),
                          t=t,
                          color='red',
                          linestyle='--',
                          label='Analytical result for unit impulse')
    stp_rsp = util.Signal(model(t, *step_params),
                          t=t,
                          color='red',
                          linestyle='--',
                          label='Analytical result for unit step')

    util.Signal.plot_all([disp, velo], title='Simulation data', block=False)
    util.Signal.plot_all([disp, imp_rsp],
                         title='Impulse response',
                         block=False)
    util.Signal.plot_all([disp, stp_rsp], title='Step response', block=True)
