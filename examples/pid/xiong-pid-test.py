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


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Run simulation of a spring-damping system with pid feedback.')

    parser.add_argument('--out',
                        type=str,
                        help='Data output directory.',
                        default=f'data/{util.formatted_date_time}-only-pid-test')
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
    parser.add_argument('--time',
                        type=float,
                        help='Simulation run time.',
                        default=1.)

    return parser.parse_args()


class PID(module.ModuleBase):
    PID_DELAY = 1e-9
    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        fs: float = 128 * 1e3,
        runtime: float = 1.,
    ) -> None:
        super().__init__(env=env, runtime=runtime+PID.PID_DELAY, dt=1/fs)
        self.system_state = system_state
        self.fs = fs

        self.a = [9., 28., 1.667, 0.1667, 0.0093]
        # self.a = [9., 28., 1.667, 0., 0.]
        self.b = [5.051e-5, 0.005]

        self.inter0 = 0.
        self.inter1 = 0.
        self.inter2 = 0.

        self.error = 0.
        self.out = 0.

        self.simulation_data = {'time': [], 'output': []}

    def update(self, current_value):
        error_pre = self.error
        self.error = current_value
        diff = self.error - error_pre
        inter2_pre = self.inter2
        inter1_pre = self.inter1
        inter0_pre = self.inter0
        self.inter2 += inter1_pre - self.b[0] * inter2_pre
        self.inter1 += inter0_pre - self.b[1] * inter2_pre
        self.inter0 += error_pre

        self.out = \
            self.a[0] * self.error + self.a[1] * diff    \
            + self.a[2] * self.inter0 + self.a[3] * self.inter1 \
            + self.a[4] * self.inter2

    def save_state(self):
        self.simulation_data['time'].append(self.env.now - PID.PID_DELAY)
        self.simulation_data['output'].append(self.out)

    def run(self):
        yield self.env.timeout(PID.PID_DELAY)
        while self.env.now < self.runtime:
            # self.system_state.pid_cmd = self.out
            self.update(self.system_state.mass_block_state[0])
            self.save_state()
            yield self.env.timeout(self.dt)

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, 'time'),
                np.array(self.simulation_data['time']))
        np.save(os.path.join(directory, 'pid'),
                np.array(self.simulation_data['output']))

class TopSystem(module.ModuleBase):
    def __init__(self, env: simpy.Environment, runtime: float, fs: float) -> None:
        super().__init__(env=env, runtime=runtime+1/fs, dt=1/fs)
        self.fs = fs
        self.system_state = SystemState()
        self.pid = PID(
            env=env, 
            system_state=self.system_state, 
            fs=self.fs, 
            runtime=self.runtime)
    
    def __unit_pulse(self, t):
        if 0. <= t and t < self.dt:
            return 1 / self.dt
        else:
            return 0.
    
    def __unit_step(self, t):
        # if t <= 0:
        #     return 0.
        # else:
            return 1.
    
    def __default_sin(self, t):
        return np.sin(2*np.pi*1e3*t)
    
    def run(self):
        '''
        '''
        while self.env.now < self.runtime:
            self.system_state.mass_block_state[0] = self.__unit_step(self.env.now)
            self.simulation_data['time'].append(self.env.now)
            self.simulation_data['output'].append(self.system_state.mass_block_state[0])
            yield self.env.timeout(self.dt)

    def save(self, directory):
        '''
        Save the simulation data to a file.
        '''
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.pid.save(directory=os.path.join(directory, 'pid'))

        directory = os.path.join(directory, 'error')
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, 'error'),
                np.array(self.simulation_data['output']))
        np.save(os.path.join(directory, 'time'),
                np.array(self.simulation_data['time']))


    @classmethod
    def load_from_file(cls, directory):
        '''
        Load the simulation data from a file.
        This method should be overridden by subclasses.
        '''
        pass


if __name__ == "__main__":
    args = argue_parser()
    verbose = args.verbose

    runtime = args.time
    fs = 128e3

    env = simpy.Environment(0)
    top = TopSystem(
        env=env,
        runtime=runtime,
        fs=fs)
    
    time_slice = 1000
    with tqdm(total=time_slice, desc='Running') as pbar:
        while env.now <= runtime + 1e-9:
            env.run(until=env.now + runtime / time_slice)
            pbar.update(1)

    # Create a data object for analysis
    pid_out = util.Signal(np.array(top.pid.simulation_data['output']),
                          t=np.array(top.pid.simulation_data['time']),
                          label='PID out')

    # Plot result
    if args.show:
        fig_time, ax_time = util.default_time_plot_fig()
        pid_out.plot_time_domain(ax=ax_time, show=True, block=True)

    def save():
        """Save result
        """
        top.save(args.out)

    if args.save:
        save()
