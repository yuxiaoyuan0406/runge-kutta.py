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

import logging

sys.path.append('.')
from module import SpringDampingSystem
from module import SystemState
from module import Noise
import util
import datetime

logger = util.default_logger(__name__, level=logging.INFO)

if __name__ == "__main__":
    logger.info('Start noise test.')
    for i in range(100):
        seed = int(datetime.datetime.now().timestamp() * 1e6)
        noise = Noise(noise_power=1e-6*9.81, sample_time=1e-6, mean=0., seed=seed)
        sampled_noise = np.array([noise.next() for _ in range(1000000)])
        print(f'power: {i+1}, mean: {np.mean(sampled_noise)}')
    logger.info('Noise test completed.')
