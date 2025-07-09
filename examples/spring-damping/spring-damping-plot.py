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
    t = np.load(os.path.join(mass_block_data, 'time.npy'))
    dt = t[1] - t[0]
    disp = np.load(os.path.join(mass_block_data, 'position.npy'))
    velo = np.load(os.path.join(mass_block_data, 'velocity.npy'))

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

    util.Signal.plot_all([disp, velo], title='Simulation data', block=True)
