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
        description='Run experiment of a spring-damping system, with simulation and curve fitting.')

    parser.add_argument(
        '--param',
        type=str,
        help='An optional simulation parameters file in json format',
        default='parameters/default.json')
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name.',
        default=''
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = argue_parser()

    if args.name == '':
        experiment_name = f'{util.formatted_date_time}-spr-dmp'
    else:
        experiment_name = args.name
    resultDir = f'data/{experiment_name}'

    simulation_script_path = 'examples/spring-damping/spring-damping-simulation.py'
    fitting_script_path = 'examples/spring-damping/spring-damping-fitting.py'

    print('--- Running simulation ---')
    subprocess.run(['python', simulation_script_path, '--param', args.param, '--out', resultDir, '--save'], check=True)
    print('--- Fitting result ---')
    subprocess.run(['python', fitting_script_path, '--data', resultDir], check=True)
