'''
Author: Xiaoyuan Yu
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class NamedArray(np.ndarray):
    def __new__(cls, input_array, name='', color=''):
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.color = color
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', "")
        self.color = getattr(obj, 'color', "")

def root_mean_square_error(a1: np.ndarray, a2: np.ndarray):
    '''
    Calculate the root mean square error of two np.ndarray
    '''
    return np.sqrt(np.mean(a1 - a2)**2)

def plot_and_difference(a1: NamedArray, a2: NamedArray):
    '''
    Plot both array and their difference.
    '''
    assert len(a1) == len(a2)

    plt.figure(num=f'{a1.name} and {a2.name}')
    plt.plot(a1, label=a1.name, color=a1.color)
    plt.plot(a2, label=a2.name, color=a2.color)
    plt.title(f'{a1.name} and {a2.name}')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show(block=False)

    diff = a1 - a2

    plt.figure(num=f'Difference between {a1.name} and {a2.name}')
    plt.plot(diff, color='black')
    plt.title(f'Difference between {a1.name} and {a2.name}')
    # plt.legend()
    plt.show(block=False)
    print(f'RMSE between {a1.name} and {a2.name} = \n\t{root_mean_square_error(a1, a2)}')


if __name__ == "__main__":
    c_out   = NamedArray(
        np.loadtxt('disp_c.dat', dtype=np.float64).transpose()[2],
        name="C data", color='red')
    py_out  = NamedArray(
        np.loadtxt('disp_py.dat', dtype=np.float64).transpose()[0],
        name="Python data", color='green')
    for_out = NamedArray(
        np.loadtxt('disp_for.dat' , dtype=np.float64).transpose()[1],
        name="Fortran data", color='blue')

    # print('C-Py   RMSE = {}'.format(root_mean_square_error(c_out, py_out)))
    # print('C-For  RMSE = {}'.format(root_mean_square_error(c_out, for_out)))
    # print('For-Py RMSE = {}'.format(root_mean_square_error(for_out, py_out)))
    plot_and_difference(c_out, py_out)
    plot_and_difference(c_out, for_out)
    plot_and_difference(for_out, py_out)

    input('Press Enter to exit...')
    plt.close('all')
