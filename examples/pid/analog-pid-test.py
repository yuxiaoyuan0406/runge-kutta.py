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
import module
from module import SpringDampingSystem
from module import SystemState
from module import ElecFeedback
import util

# matplotlib.use('TkAgg')

def external_accel(t):
    # return 9.8 * 0.01 * np.sin(2 * np.pi * 125 * t)
    return 9.8 * 1e-3


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Run simulation of a spring-damping system with pid feedback.')

    parser.add_argument(
        '--param',
        type=str,
        help='An optional simulation parameters file in json format',
        default='parameters/xiong-2018.json')
    parser.add_argument('--out',
                        type=str,
                        help='Data output directory.',
                        default=f'data/{util.formatted_date_time}-pid-closed-loop-test')
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


class PID(module.ModuleBase):
    
    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        config: dict,
        fs: float = 128 * 1e3,
        runtime: float = 1.,
    ) -> None:
        super().__init__(env=env, runtime=runtime, dt=1/fs)
        self.system_state = system_state
        self.fs = fs

        self.a = [config[f'a{i+1}'] for i in range(5)]
        self.b = [config[f'b{i+1}'] for i in range(2)]

        self.inter0 = 0.
        self.inter1 = 0.
        self.inter2 = 0.

        self.error = 0.

        self.simulation_data = {'time': [], 'output': []}

    def update(self, current_value):
        error_pre = self.error
        self.error = current_value
        diff = self.error - error_pre
        inter2_pre = self.inter2
        self.inter2 += self.inter1 - self.b[0] * self.inter2
        self.inter1 += self.inter0 - self.b[1] * inter2_pre
        self.inter0 += error_pre

        self.out = \
            self.a[0] * self.error + self.a[1] * diff    \
            + self.a[2] * self.inter0 + self.a[3] * self.inter1 \
            + self.a[3] * self.inter2

    def save_state(self):
        self.simulation_data['time'].append(self.env.now)
        self.simulation_data['output'].append(self.out)

    def run(self):
        while self.env.now < self.runtime:
            self.save_state()
            self.system_state.pid_cmd = self.out
            self.update(self.system_state.mass_block_state[0])
            yield self.env.timeout(self.dt)

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, 'time'),
                np.array(self.simulation_data['time']))
        np.save(os.path.join(directory, 'pid'),
                np.array(self.simulation_data['output']))

class PIDSystem(module.ModuleBase):
    def __init__(
        self, 
        env: simpy.Environment,
        config: dict,
        extern_accel=external_accel,
    ):
        self.fs = config['samp_rate']
        super().__init__(env=env, runtime=config['runtime'], dt=1/self.fs)

        self.system_state = SystemState()
        self.extern_accel = extern_accel

        self.system_state.mass_block_state[0] = self.extern_accel(0)

        self.pid = PID(
            env=self.env,
            system_state=self.system_state,
            config=config,
            fs=self.fs,
            runtime=self.runtime,
        )

    def run(self):
        while self.env.now < self.runtime:
            self.system_state.mass_block_state[0] = self.extern_accel(self.env.now)
            yield self.env.timeout(self.dt)
            


if __name__ == "__main__":
    args = argue_parser()
    verbose = args.verbose

    if verbose:
        print(f'Using parameters from file `{args.param}`.')
    with open(args.param, 'r', encoding='utf-8', errors='replace') as f:
        param = json.load(f)
        f.close()

    runtime = param['runtime']

    if verbose:
        print(json.dumps(param, indent=2))

    env = simpy.Environment(0)

    

    time_slice = 1000
    with tqdm(total=time_slice, desc='Running') as pbar:
        while env.now < runtime:
            env.run(until=env.now + runtime / time_slice)
            pbar.update(1)


    # Create a data object for analysis
    pass

    # Plot result
    if args.show:
        pass

    def save():
        """Save result
        """
        util.save_dict(os.path.join(args.out, 'param.json'), param)
        pass

    if args.save:
        save()
