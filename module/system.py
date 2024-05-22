'''
System wide class.
'''
from dataclasses import dataclass
import numpy as np
import simpy


@dataclass
class SystemState:
    '''
    Class to store some shared system state between modules.
    '''
    mass_block_state: np.ndarray = np.array([0, 0], np.float64)
    pid_cmd: int = 1
    elec_feedback_enable = True

    # def __init__(
    #         self,
    #         mass_block_state: np.ndarray = np.array([0, 0], np.float64),
    # ) -> None:

    #     self.mass_block_state = mass_block_state
    #     self.pid_cmd = 1
    #     self.elec_feedback_enable = True


def external_force(t: float):
    '''
    External Force, nothing unusual.
    '''
    return 0.00004 * np.sin(2 * np.pi * 5e1 * t)


class System:
    '''
    The whole MEMs system.
    '''

    def __init__(
        self,
        env: simpy.Environment,
        config: dict,
        extern_f=external_force,
    ) -> None:
        self.runtime = config['runtime']
        self.mechanic_dt = config['mechanic_dt']
        self.fs = config['samp_rate']

        self.env = env
        self.system_state = SystemState()

        self.extern_f = extern_f

        self.initial_state = np.array(config.get('initial_state', [0, 0]),
                                      dtype=np.float64)

        from .spring_damping import SpringDampingSystem
        self.spring_system = SpringDampingSystem(
            env=self.env,
            mass=config['mass'],
            spring_coef=config['spring_coef'],
            damping_coef=config['damping_coef'],
            initial_state=self.initial_state,
            system_state=self.system_state,
            runtime=self.runtime,
            dt=self.mechanic_dt,
            input_force=self.calclute_force,
        )

        from .pid import PID
        self.pid = PID(
            env=self.env,
            kp=config['kp'],
            ki1=config['ki1'],
            ki2=config['ki2'],
            kd=config['kd'],
            system_state=self.system_state,
            fs=self.fs,
            runtime=self.runtime,
            target=config['pid_target'],
        )

        from .elec_feedback import ElecFeedback
        self.elec_feedback = ElecFeedback(
            env=self.env,
            area=config['area'],
            gap=config['gap'],
            v_ref=config['v_ref'],
            runtime=self.runtime,
            fs=self.fs,
            system_state=self.system_state,
        )

    def calclute_force(self, t):
        '''
        The force delivered to the mass block is external force 
        plus the electrical feedback force.
        '''
        return self.extern_f(t) + self.elec_feedback.force()
