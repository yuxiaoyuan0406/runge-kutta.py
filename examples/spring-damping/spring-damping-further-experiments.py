import os
import sys

sys.path.append('.')
import util
from module import SpringDampingSystem

if __name__ == '__main__':
    Q = ['0.1', '0.5', '5.0']
    result_path_list = [os.path.join('data/20240826-015757-spr-dmp-multi-Q', f'Q_{q}') for q in Q]
    disp_list = []
    for i in range(len(result_path_list)):
        result_path = result_path_list[i]
        disp,_ = SpringDampingSystem.load_from_file(os.path.join(result_path, 'mass_block'))
        disp.label = f'Q={Q[i]}'
        disp_list.append(disp)
        
    util.Signal.plot_all(disp_list)
    