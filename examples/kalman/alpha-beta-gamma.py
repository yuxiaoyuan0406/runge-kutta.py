import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker

import sys
import os

sys.path.append('.')
import module
import util


if __name__ == "__main__":
    mass_ground_truth = 1000.00

    iter_count = 26

    # _rng = np.random.default_rng(util.now_as_seed())
    _rng = np.random.default_rng(19970804)
    noise = np.array([_rng.normal(loc=0, scale=10) for i in range(iter_count)])

    def measure(n):
        return mass_ground_truth + noise[n]

    iter_list = [0]
    true_mass = [mass_ground_truth]
    meas_mass = [measure(0)]
    esti_mass = [measure(0)]

    for i in range(1, iter_count):
        iter_list.append(i)
        true_mass.append(mass_ground_truth)
        meas_mass.append(measure(i))
        esti_mass.append(esti_mass[i-1] + 1/i * (meas_mass[i] - esti_mass[i-1]))

    fig0, ax = plt.subplots(figsize=(8,6))
    ax.grid(True)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Mass [g]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    fig0.suptitle('Measurements vs. Ground Truth vs. Estimates', fontsize=util.FIG_TITLE_FONT_SIZE)

    ax.plot(iter_list, meas_mass, label='Measurements', color='blue', marker='s')
    ax.plot(iter_list, esti_mass, label='Estimates', color='red', marker='o')
    ax.plot(iter_list, true_mass, label='Ground Truth', color='green')

    ax.legend(loc='upper right')


    esti_mass = np.array(esti_mass)
    esti_loss = esti_mass - mass_ground_truth
    fig1, ax = plt.subplots(figsize=(8,6))
    ax.grid(True)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('loss [g]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    fig1.suptitle('Estimate loss', fontsize=util.FIG_TITLE_FONT_SIZE)

    ax.plot(iter_list, esti_loss, label='Estimate loss', marker='o')

    ax.legend(loc='upper right')

    plt.show()

    fig_save_dir = './data/alpha-beta-gamma/'
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    fig0.savefig(os.path.join(fig_save_dir, 'alpha-beta-gamma-1.png'),
                 bbox_inches='tight', dpi=300)
    fig1.savefig(os.path.join(fig_save_dir, 'alpha-beta-gamma-2.png'),
                 bbox_inches='tight', dpi=300)
