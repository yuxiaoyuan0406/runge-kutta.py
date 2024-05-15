import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from numba import jit

import freq
import quantizer

# @jit(nopython=True)
def f(x: np.ndarray):
    w = [0., x[0]]
    y = [1.]
    e = [1.]

    if w[1] >= 0:
        y.append(1.)
    else:
        y.append(-1.)

    e.append(y[1] - w[1])

    for i in tqdm(range(2, len(x)), desc='Simulation'):
        w.append(x[i-1] - 2 * e[i-1] + e[i-2])
        if w[i] >= 0:
            y.append(1.)
        else:
            y.append(-1.)
        e.append(y[i] - w[i])

    # print('i\tx\tw\ty\te\t')
    # for i in range(len(x)):
    #     print(f'{i}\t{x[i]}\t{w[i]}\t{y[i]}\t{e[i]}')
        
    return w,y,e

if __name__ == '__main__':
    t,dt = np.linspace(0,2,256000, retstep=True)
    # x = np.zeros(t.shape)
    # x[0] = 1
    x = np.sin(2*np.pi*1e2*t) * 2
    x = np.clip(x, -0.9, 0.9)
    w,y,e = f(x)
    plt.plot(t, w, label='w')
    plt.plot(t, y, label='y')
    plt.plot(t, e, label='e')
    plt.plot(t, x, label='x')

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show(block=False)

    freq.freq_and_plot(np.array(y), dt, log=True, max_freq=1e4)

    input('Press Enter to exit...')
    plt.close('all')
