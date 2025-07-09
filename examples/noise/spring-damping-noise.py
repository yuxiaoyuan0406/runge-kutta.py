'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys

import simpy
import numpy as np
import matplotlib
import os
from tqdm import tqdm

import logging

sys.path.append('.')
from module import SpringDampingSystem
from module import SystemState
from module import Noise
import util

# matplotlib.use('TkAgg')
logger = util.default_logger(__name__, level=logging.INFO)


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Run simulation of a spring-damping system with noise.')

    parser.add_argument(
        '--param',
        type=str,
        help='An optional simulation parameters file in json format',
        default='parameters/default.json')
    parser.add_argument('--out',
                        type=str,
                        help='Data output directory.',
                        default=f'data/{util.formatted_date_time}-noise')
    parser.add_argument('--save',
                        action='store_true',
                        default=False,
                        help='whether to save the simulation result')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Print extra info')
    parser.add_argument('--show',
                        action='store_true',
                        default=False,
                        help='Plot result')

    return parser.parse_args()

if __name__ == '__main__':
    args = argue_parser()
    verbose = args.verbose

    if verbose:
        # print(f'Using parameters from file `{args.param}`.')
        logger.info(f'Using parameters from file `{args.param}`.')
    with open(args.param, 'r', encoding='utf-8', errors='replace') as f:
        param = json.load(f)
        f.close()

    runtime = param['runtime']
    dt = param['mechanic_dt']

    if verbose:
        logger.info(json.dumps(param, indent=2))


    def simulate(acce, noise_asd, name='Experiment'):
        # Create simulation enviorment
        env = simpy.Environment(0)

        # Initialize spring-damping module
        initial_state = np.array(param['initial_state'], dtype=np.float64)
        spring_system = SpringDampingSystem(
            env=env,
            system_state=SystemState(),
            mass=param['mass'],
            spring_coef=param['spring_coef'],
            damping_coef=param['damping_coef'],
            initial_state=initial_state,
            runtime=runtime,
            dt=dt,
            input_accel=acce,
            noise=Noise(noise_power=noise_asd, sample_time=dt, mean=0, seed=util.now_as_seed())
        )

        # Run simulation with a progress bar
        time_slice = 1000
        with tqdm(total=time_slice, desc='Running') as pbar:
            while env.now < runtime:
                env.run(until=env.now + runtime / time_slice)
                pbar.update(1)

        # Create a data object for analysis
        _disp = util.Signal(np.array(spring_system.simulation_data['position']),
                           t=np.array(spring_system.simulation_data['time']),
                           label=f'{name}: Position')
        _velo = util.Signal(np.array(spring_system.simulation_data['velocity']),
                           t=np.array(spring_system.simulation_data['time']),
                           label=f'{name}: Velocity')
        return _disp, _velo, spring_system

    # Run simulation
    @util.vectorize
    def unit_pulse(t):
        if 0 <= t < dt / 2:
            return 6. / dt
        return 0
    logger.info('Running simulation with unit pulse input and no noise.')
    disp, velo, sprint = simulate(unit_pulse, 0, name='No noise')
    logger.info('Simulation without noise finished.')
    # Zero input
    @util.vectorize
    def zero(t):
        return 0
    logger.info('Running simulation with zero input.')
    disp_0, velo_0, spring_0 = simulate(zero, param['noise_level'], name='Zero input')
    logger.info('Simulation with zero input finished.')

    # Non-zero input
    @util.vectorize
    def exte_accel(t):
        return 0.1 * 9.81 * np.sin(2 * np.pi * 50 * t)
    logger.info('Running simulation with non-zero input.')
    disp_1, velo_1, spring_1 = simulate(exte_accel, param['noise_level'], name='Non-zero input')
    logger.info('Simulation with non-zero input finished.')

    # Plot result
    if args.show:
        logger.info('Plotting simulation result.')
        util.Signal.plot_all([disp, disp_0, disp_1])
        logger.info('Simulation result plotted.')

    def save():
        """Save result
        """
        logger.info('Saving simulation result.')
        util.save_dict(os.path.join(args.out, 'param.json'), param)
        sprint.save(os.path.join(args.out, 'mass_block'))
        spring_0.save(os.path.join(args.out, 'zero_input'))
        spring_1.save(os.path.join(args.out, 'nonzero_input'))
        logger.info('Simulation result saved.')

    if args.save:
        save()
