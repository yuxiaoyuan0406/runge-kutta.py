'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib

sys.path.append('.')
import util

# matplotlib.use('TkAgg')

def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Analysis data file with fitting.')

    parser.add_argument(
        '--file',
        type=str,
        help='Data file to analysis.')

    return parser.parse_args()

if __name__ == '__main__':
    args = argue_parser()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    param = data['parameters']
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

    mass_block_data = data['mass_block_state']
    t = np.array(mass_block_data['time'])
    dt = t[1] - t[0]
    disp = np.array(mass_block_data['position'])
    velo = np.array(mass_block_data['velocity'])

    def model(t, racio, decay, omega, phi):
        return racio * np.exp(-decay * t) * np.sin(omega * t + phi)

    racio = 1/(w_n*np.sqrt(1-zeta**2))
    decay = zeta * w_n
    omega = w_n * np.sqrt(1-zeta**2)
    phi = 0
    initial_params = [racio, decay, omega, phi]
    print('Analytical solution of unit impulse response is:\n %.5e * exp(- %.5e * t) * sin(%.5e * t + %.5e)' % tuple(initial_params))
    popt, pcov = curve_fit(model, t, disp, p0=initial_params)
    print('Simulation of unit impulse response is:\n %.5e * exp(- %.5e * t) * sin(%.5e * t + %.5e)' % tuple(popt))

    simul = util.Signal(disp, t=t, color='blue', linestyle='-', label='Simulation data')
    analy = util.Signal(model(t, *initial_params), t=t, color='red', linestyle='--', label='Analytical result')

    util.Signal.plot_all()
