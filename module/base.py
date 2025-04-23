# import numpy as np
import simpy

class ModuleBase:
    '''
    Base class for all simpy-based modules.
    Provides common functionality for initialization and simulation.
    '''

    def __init__(self, env: simpy.Environment, runtime: float, dt: float) -> None:
        self.env = env
        self.simulation_data = {'time': [], 'output': []}
        self.runtime = runtime
        self.dt = dt
        self.env.process(self.run())

    def run(self):
        '''
        Step the simulation by dt seconds.
        This method should be overridden by subclasses.
        '''
        raise NotImplementedError("Subclasses must implement the `run` method.")

    def save(self, directory):
        '''
        Save the simulation data to a file.
        This method should be overridden by subclasses.
        '''
        raise NotImplementedError("Subclasses must implement the `save` method.")

    @classmethod
    def load_from_file(cls, directory):
        '''
        Load the simulation data from a file.
        This method should be overridden by subclasses.
        '''
        raise NotImplementedError("Subclasses must implement the `load_from_file` method.")
