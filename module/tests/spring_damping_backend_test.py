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
import time

sys.path.append('.')
import module
from module import SpringDampingSystem
from module import SystemState
import util

# matplotlib.use('TkAgg')


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
    # parser.add_argument('--cpp',
    #                     action='store_true',
    #                     default=False,
    #                     help='whether to use cpp backend')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Print extra info')
    parser.add_argument('--show',
                        action='store_true',
                        default=False,
                        help='Plot result')
    parser.add_argument('--time',
                        type=float,
                        help='Simulation run time.',
                        default=None)

    return parser.parse_args()

logger = util.default_logger(__name__, level=util.logging.INFO)

if __name__ == '__main__':
    args = argue_parser()
    verbose = args.verbose
    # module.spring_damping.USING_CPP_BACKEND = args.cpp

    if verbose:
        print(f'Using parameters from file `{args.param}`.')
    with open(args.param, 'r', encoding='utf-8', errors='replace') as f:
        param = json.load(f)
        f.close()

    runtime: float
    if args.time:
        runtime = args.time
    else:
        runtime = param['runtime']
    dt = param['mechanic_dt']

    if verbose:
        print(json.dumps(param, indent=2))

    @util.vectorize
    def exte_accel(t):
        '''
        external accel
        '''
        if 0 <= t and t < dt/2:
            return 6./dt
        return 0.

    def run_simulation():
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
        return spring_system

    logger.info('Running python backend test.')
    module.spring_damping.USING_CPP_BACKEND = False
    t0 = time.perf_counter()
    py_backend = run_simulation()
    tx = time.perf_counter()
    py_time = tx-t0

    logger.info('Running C++ backend test.')
    module.spring_damping.USING_CPP_BACKEND = True
    t0 = time.perf_counter()
    c_backend = run_simulation()
    tx = time.perf_counter()
    cpp_time = tx-t0

    logger.info(f'Python backend time: {py_time}s')
    logger.info(f'C++ backend time: {cpp_time}s')
    logger.info(f'Time reduced by {(py_time - cpp_time) / py_time * 100}%')
    # Create a data object for analysis
    # disp = util.Signal(np.array(spring_system.simulation_data['position']),
    #                    t=np.array(spring_system.simulation_data['time']),
    #                    label='Position')
    # velo = util.Signal(np.array(spring_system.simulation_data['velocity']),
    #                    t=np.array(spring_system.simulation_data['time']),
    #                    label='Velocity')

    py_disp = util.Signal(np.array(py_backend.simulation_data['position']),
                t=np.array(py_backend.simulation_data['time']),
                label='py backend')
    
    c_disp = util.Signal(np.array(c_backend.simulation_data['position']),
                t=np.array(c_backend.simulation_data['time']),
                label='c backend')

    # Plot result
    if args.show:
        util.Signal.plot_all()

    def save():
        """Save result
        """
        util.save_dict(os.path.join(args.out, 'param.json'), param)
        py_backend.save(os.path.join(args.out, 'py', 'mass_block'))
        c_backend.save(os.path.join(args.out, 'cpp', 'mass_block'))

    if args.save:
        save()
