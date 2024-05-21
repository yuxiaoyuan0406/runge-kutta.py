'''
'''
import numpy as np
import simpy

class SystemState:
    def __init__(
        self,
        mass_block_state: np.ndarray = np.array([0,0], np.float64),
    ) -> None:

        self.mass_block_state = mass_block_state
        self.pid_cmd = 1
        self.elec_feedback_enable = True

def external_force(t: float):
    '''
    External Force, nothing unusual.
    '''
    return 0.00004*np.sin(2 * np.pi * 5e1 * t)

class System:
    def __init__(
        self,
        env,
        extern_f = external_force,
    ) -> None:
        self.runtime = 1.
        self.mechanic_dt = 1e-6
        self.fs = 128 * 1e3

        self.env = env
        self.system_state = SystemState()

        self.extern_f = extern_f

        self.initial_state = np.array([0., 0.], dtype=np.float64)

        from .spring_damping import SpringDampingSystem
        self.spring_system = SpringDampingSystem(
            env = self.env,
            mass = 7.45e-7,
            spring_coef = 5.623,
            damping_coef = 4.95e-6,
            initial_state = self.initial_state,
            system_state = self.system_state,
            runtime = self.runtime,
            dt = self.mechanic_dt,
            input_force = self.calclute_force,
        )

        from .pid import PID
        self.pid = PID(
            env = self.env,
            kp = -5,
            ki1 = -0.516,
            ki2 = -0.5,
            kd = -1,
            system_state = self.system_state,
            fs = self.fs,
            runtime = self.runtime,
            target = 0,
        )

        from .elec_feedback import ElecFeedback
        self.elec_feedback = ElecFeedback(
            env = self.env,
            area = 1.7388e-6,
            gap = 3e-6,
            v_ref = 2.5,
            runtime = self.runtime,
            fs = self.fs,
            system_state = self.system_state,
        )

    def calclute_force(self, t):
        return self.extern_f(t) + self.elec_feedback.force()
