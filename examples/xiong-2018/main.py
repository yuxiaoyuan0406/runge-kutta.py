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
from module import ElecFeedback
import util

# matplotlib.use('TkAgg')

def external_accel(t):
    return 9.8 * 0.01 * np.sin(2 * np.pi * 125 * t)


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


class PID:
    
    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        config: dict,
        fs: float = 128 * 1e3,
        runtime: float = 1.,
    ) -> None:
        self.env = env
        self.system_state = system_state
        self.fs = fs
        self.runtime = runtime

        self.a = [config[f'a{i+1}'] for i in range(5)]
        self.b = [config[f'b{i+1}'] for i in range(2)]

        self.inter0 = 0.
        self.inter1 = 0.
        self.inter2 = 0.

        self.error = 0.

        self.simulation_data = {'time': [], 'output': []}

        self.out = self.quantizer(0)

        self.env.process(self.run())

    def quantizer(self, val):
        return int(np.sign(val))

    def update(self, current_value):
        error_pre = self.error
        self.error = current_value
        diff = self.error - error_pre
        inter2_pre = self.inter2
        self.inter2 += self.inter1 - self.b[0] * self.inter2
        self.inter1 += self.inter0 - self.b[1] * inter2_pre
        self.inter0 += error_pre

        self.out = self.quantizer(
            self.a[0] * self.error + self.a[1] * diff
            + self.a[2] * self.inter0 + self.a[3] * self.inter1
            + self.a[3] * self.inter2
        )

    def save_state(self):
        self.simulation_data['time'].append(self.env.now)
        self.simulation_data['output'].append(self.out)

    def run(self):
        while self.env.now < self.runtime:
            self.save_state()
            self.system_state.pid_cmd = self.out
            self.update(self.system_state.mass_block_state[0])
            yield self.env.timeout(1 / self.fs)

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, 'time'),
                np.array(self.simulation_data['time']))
        np.save(os.path.join(directory, 'pid'),
                np.array(self.simulation_data['output']))

class System:
    """
    The whole MEMs system.
    """
    def __init__(
        self, 
        env: simpy.Environment,
        config: dict,
        extern_accel=external_accel,
    ):
        self.runtime = config['runtime']
        self.mechanic_dt = config['mechanic_dt']
        self.fs = config['samp_rate']

        self.env = env
        self.system_state = SystemState()

        self.extern_accel = extern_accel

        self.initial_state = np.array(config.get('initial_state', [0, 0]),
                                      dtype=np.float64)

        self.spring_system = SpringDampingSystem(
            env=self.env,
            mass=config['mass'],
            spring_coef=config['spring_coef'],
            damping_coef=config['damping_coef'],
            initial_state=self.initial_state,
            system_state=self.system_state,
            runtime=self.runtime,
            dt=self.mechanic_dt,
            input_accel=self.calclute_accel,
        )

        self.pid = PID(
            env=self.env,
            system_state=self.system_state,
            config=config,
            fs=self.fs,
            runtime=self.runtime,
        )

        self.elec_feedback = ElecFeedback(
            env=self.env,
            area=config['area'],
            gap=config['gap'],
            v_ref=config['v_ref'],
            runtime=self.runtime,
            fs=self.fs,
            system_state=self.system_state,
        )

    def calclute_accel(self, t):
        '''
        The acceleration delivered to the mass block is external acceleration 
        plus the electrical feedback force.
        '''
        return self.extern_accel(t) + self.elec_feedback.force() / self.spring_system.m

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

    simu_sys = System(
        env=env,
        config=param,
        )

    time_slice = 1000
    with tqdm(total=time_slice, desc='Running') as pbar:
        while env.now < runtime:
            env.run(until=env.now + runtime / time_slice)
            pbar.update(1)


    # Create a data object for analysis
    disp = util.Signal(np.array(simu_sys.spring_system.simulation_data['position']),
                       t=np.array(simu_sys.spring_system.simulation_data['time']),
                       label='Position')
    velo = util.Signal(np.array(simu_sys.spring_system.simulation_data['velocity']),
                       t=np.array(simu_sys.spring_system.simulation_data['time']),
                       label='Velocity')
    pid  = util.Signal(np.array(simu_sys.pid.simulation_data['output']),
                       t=np.array(simu_sys.pid.simulation_data['time']),
                       label='Velocity')


    # Plot result
    if args.show:
        util.Signal.plot_all([disp, velo], block=False)
        util.Signal.plot_all([pid])

    def save():
        """Save result
        """
        util.save_dict(os.path.join(args.out, 'param.json'), param)
        simu_sys.spring_system.save(os.path.join(args.out, 'mass_block'))
        simu_sys.pid.save(os.path.join(args.out, 'pid'))

    if args.save:
        save()
