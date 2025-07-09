'''
Author: Yu Xiaoyuan
'''
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

def external_force(t):
    return 9.8 * 0.01 * np.sin(2 * np.pi * 125 * t)

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

class System:
    """
    The whole MEMs system.
    """
    def __init__(
        self, 
        env: simpy.Environment,
        config: dict,
        extern_accel=external_force,
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
    pass
