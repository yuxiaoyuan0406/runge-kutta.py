'''
Author: Yu Xiaoyuan
'''
import argparse
import sys
import subprocess
import multiprocessing
import os
import json
from tqdm import tqdm
import time

import simpy
import numpy as np

sys.path.append('.')

import util
from module import SpringDampingSystem
from module import SystemState


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description=
        'Run experiment of a spring-damping system, with simulation and curve fitting.'
    )

    parser.add_argument(
        '--param',
        type=str,
        help='An optional simulation parameters file in json format',
        default='parameters/default.json')
    parser.add_argument('--name',
                        type=str,
                        help='Experiment name.',
                        default='')
    return parser.parse_args()


def simulate(param: dict, desc='Running'):
    """Run spring-damping simulation

    Args:
        param (dict): System parameters.
        desc (str, optional): Description of process bar. Defaults to 'Running'.

    Returns:
        (np.ndarray, np.ndarray): (Displacement, time).
    """
    runtime = param['runtime']
    dt = param['mechanic_dt']
    
    def exte_accel(t):
        # if 0 <= t and t < dt / 2:
        #     return 6. / dt
        # return 0.
        return 1.
    
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
    with tqdm(total=time_slice, desc=desc) as pbar:
        while env.now < runtime:
            env.run(until=env.now + runtime / time_slice)
            pbar.update(1)

    # Create a data object for analysis
    # disp = util.Signal(np.array(spring_system.simulation_data['position']),
    #                    t=np.array(spring_system.simulation_data['time']),
    #                    label='Position')
    # velo = util.Signal(np.array(spring_system.simulation_data['velocity']),
    #                    t=np.array(spring_system.simulation_data['time']),
    #                    label='Velocity')
    disp = np.array(spring_system.simulation_data['position'])
    t=np.array(spring_system.simulation_data['time'])
    
    return disp, t


if __name__ == "__main__":
    args = argue_parser()

    if args.name == '':
        experiment_name = f'{util.formatted_date_time}-spr-dmp-multi-k'
    else:
        experiment_name = args.name
    resultDir = f'data/{experiment_name}'

    with open(args.param, 'r', encoding='utf-8') as f:
        param = json.load(f)
        f.close()
    param['runtime'] = 0.02
    # param['damping_coef'] = 0.02

    print(json.dumps(param, indent=2, sort_keys=True))

    m = param['mass']
    b = param['damping_coef']
    k = param['spring_coef']

    print('--- Running simulation ---')
    start_time = time.time()

    disp_list = []
    S = 10
    Q = np.sqrt(m * k) / b
    Q_list = np.logspace(-1, np.log10(3), S)
    for i in range(S):
        _Q = Q_list[i]
        _b = np.sqrt(m * k) / _Q
        p = dict(param)
        p['damping_coef'] = _b
        disp, t = simulate(p, desc=f'Q={_Q}')
        disp = util.Signal(disp, t=t, label=f'Q={_Q}', color = (i/S/2, i/S/2, i/S))
        disp_list.append(disp)

    end_time = time.time()
    print('--- Simulation finished ---')
    print(f'Simulation runtime: {end_time - start_time} (s).')

    util.Signal.plot_all(disp_list, title='Multi Q: Displacement')
