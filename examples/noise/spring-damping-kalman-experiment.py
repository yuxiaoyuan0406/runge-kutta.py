'''
Author: Yu Xiaoyuan
'''
import argparse
import sys
import subprocess

sys.path.append('.')

import util

# matplotlib.use('TkAgg')

def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Run experiment of a spring-damping system, noise and kalman filter.')

    parser.add_argument(
        '--param',
        type=str,
        help='An optional simulation parameters file in json format',
        default='parameters/default-noise.json')
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name.',
        default=''
    )

    return parser.parse_args()

logger = util.default_logger(__name__, level=util.logging.INFO)

if __name__ == "__main__":
    args = argue_parser()

    if args.name == '':
        experiment_name = f'{util.formatted_date_time}-noise-kalman'
    else:
        experiment_name = args.name
    resultDir = f'data/{experiment_name}'

    simulation_script_path = 'examples/noise/spring-damping-noise.py'
    filter_script_path = 'examples/noise/spring-damping-kalman.py'

    logger.info('Running simulation with noise...')
    subprocess.run(['python', simulation_script_path, '--param', args.param, '--out', resultDir, '--save'], check=True)
    logger.info('Simulation completed.')
    logger.info('Running Kalman filter on the simulation data...')
    subprocess.run(['python', filter_script_path, '--data', resultDir], check=True)
    logger.info('Kalman filter completed.')
