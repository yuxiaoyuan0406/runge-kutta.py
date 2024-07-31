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

    return parser.parse_args()

if __name__ == "__main__":
    args = argue_parser()

    timestamp = util.formatted_date_time
    resultFilename = f'data/{timestamp}/simulation-result.json'

    simulation_script_path = 'examples/spring-damping/spring-damping-simulation.py'
    fitting_script_path = 'examples/spring-damping/spring-damping-fitting.py'

    print('--- Running simulation ---')
    subprocess.run(['python', simulation_script_path, '--param', args.param, '--out', resultFilename, '--save'], check=True)
    print('--- Fitting result ---')
    subprocess.run(['python', fitting_script_path, '--file', resultFilename], check=True)
