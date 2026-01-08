import sys
sys.path.append('.')

import util
from util.parameters import *

if __name__ == "__main__":
    file_path = 'util/tests/test-param.json'

    param = SpringDampingParameters(file_path)
    print(param)
