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

sys.path.append('.')
from module import SpringDampingSystem
from module import SystemState
import util

matplotlib.use('TkAgg')


def external_accel(t):
    '''
    External Force, nothing unusual.
    '''
    # return 4e-1 * np.sin(2 * np.pi * 5e1 * t)
    return unit_pulse(t)
    # return unit_step(t)

@util.vectorize
def unit_pulse(x: float, offset: float = 0.)-> float:
    '''
    Unit pulse function.
    '''
    if x == offset:
        return 1.
    return 0.

@util.vectorize
def unit_step(x: float)-> float:
    '''
    Unit step function.
    '''
    if x >= 0.1:
        return 1.
    return 0.

def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Run simulation of a spring-damping system.')

    parser.add_argument(
        '--param',
        type=str,
        help='An optional simulation parameters file in json format',
        default='parameters/default.json')
    parser.add_argument('--out',
                        type=str,
                        help='Data output directory.',
                        default=f'data/{util.formatted_date_time}')
    parser.add_argument('--save',
                        action='store_true',
                        default=False,
                        help='whether to save the simulation result')

    return parser.parse_args()


if __name__ == '__main__':
    args = argue_parser()

    print(f'Using parameters from file `{args.param}`.')
    with open(args.param, 'r', encoding='utf-8', errors='replace') as f:
        param = json.load(f)
        f.close()
        json.dump(param, sys.stdout, indent=2)

    runtime = param['runtime']
    # runtime = 4
    dt = param['mechanic_dt']
    # dt = 1e-8

    @util.vectorize
    def exte_accel(t):
        '''
        external accel
        '''
        if 0 <= t and t < dt/2:
            return 6./dt
        return 0.

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
        input_accel=exte_accel,
    )

    # Run simulation with a progress bar
    time_slice = 1000
    with tqdm(total=time_slice, desc='Running') as pbar:
        while env.now < runtime:
            env.run(until=env.now + runtime / time_slice)
            pbar.update(1)

    # Create a data object for analysis
    disp = util.Signal(np.array(spring_system.simulation_data['position']),
                       t=np.array(spring_system.simulation_data['time']),
                       label='Position')
    velo = util.Signal(np.array(spring_system.simulation_data['velocity']),
                       t=np.array(spring_system.simulation_data['time']),
                       label='Velocity')

    # Plot result
    util.Signal.plot_all()

    def save():
        """Save result
        """
        util.save(os.path.join(args.out, 'param.json'), param)
        spring_system.save(os.path.join(args.out, 'mass_block'))

    if args.save:
        save()
