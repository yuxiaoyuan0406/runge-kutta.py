'''
Noise object
'''
import numpy as np

class Noise:
    def __init__(
        self,
        noise_power,
        sample_time,
        mean,
        shape = (1,),
        seed = 9784,
    ):
        """Noise object.

        Usage:
        ```
        noise = Noise(noise_power=2, sample_time=1e-3, mean=0)
        samples = np.array([noise.next() for _ in range(num_samples)]).flatten()
        ```

        Args:
            noise_power (float): The ASD of noise. (NOT PSD!!!)
            sample_time (float): The sampling time. 
            mean (float): The mean of Gaussian distribution.
            shape (tuple, optional): The shape of output. Defaults to (1,).
            seed (int, optional): The seed of random generator.
        """
        self.asd = noise_power
        self.dt = sample_time
        self.variance = self.asd ** 2 / self.dt
        self.shape = shape
        self.seed = seed
        self.mean = mean
        self._rng = np.random.default_rng(self.seed)
        self._generator = self._noise_gen()
        self.__counter = 0

    def _noise_gen(self):
        while True:
            yield self._rng.normal(loc=self.mean, scale=np.sqrt(self.variance))

    def next(self):
        """
        Returns the next value from the internal generator.
        Returns:
            The next value produced by the internal generator.
        Raises:
            Should not raise any exceptions.
        """
        self.__counter += 1
        return next(self._generator)
