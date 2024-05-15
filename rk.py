import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

from scipy.integrate import odeint

mass = 7.45e-7
spring_coef = 5.623
dumping_coef = 4.95e-6

normal_spr_coef = spring_coef/mass
normal_dmp_coef = dumping_coef/mass

t = np.linspace(0.,2., num=2000000)

# @jit(nopython=True)
# def a(x):
#     # return 0.00004 * np.sin(2*np.pi * 50 * t)
#     return 40 * np.sin(2*np.pi * 50 * x)

a_in = 40 * np.sin(2*np.pi * 50 * t)

@jit(nopython=True)
def a(x):
    i = int(x*1e7)
    if i%10 == 0:
        return a_in[i//10]
    else:
        return .5*(a_in[i//10] + a_in[i//10+1])


@jit(nopython=True)
def f1(y,t):
    z1, z2 = y
    dydt = np.array([
        z2,
        -normal_spr_coef*z1 -normal_dmp_coef*z2 + a(t)
    ])
    return dydt

@jit(nopython=True)
def runge_kutta(func, t_0, y_0, h):
    '''
    -
    '''
    k_0 = func(y=y_0, t=t_0)
    k_1 = func(y=y_0 + h/2 * k_0, t=t_0 + h/2)
    k_2 = func(y=y_0 + h/2 * k_1, t=t_0 + h/2)
    k_3 = func(y=y_0 + h * k_2, t=t_0 + h)

    k = 1./6. * (k_0 + 2*k_1 + 2*k_2 + k_3)

    t_1 = t_0 + h
    y_1 = y_0 + h * k

    return t_1, y_1

t_0 = 0.
y_0 = np.array([0.,0.])

h = 1e-6

# t = [t_0]
y = [y_0]

_t = t_0
_y = y_0
for i in tqdm(range(2000000-1), desc="Runge-Kutta"):
    _t, _y = runge_kutta(f1, _t, _y, h)
    # t = np.append(t, [_t], 0)
    # y = np.append(y, [_y], 0)
    # t.append(_t)
    y.append(_y)

# t = np.array(t)
y = np.array(y)

# y0 = [0.,0.]
# t = np.linspace(0.,2., num=2000000)

# np.savetxt('standard_input.dat', a(t).transpose(), delimiter=' ')
np.savetxt('disp_py.dat', y, delimiter='  ')

# sol = odeint(f, y0, t, args=(normal_spr_coef, normal_dmp_coef))

# plt.plot(t, y.transpose()[0], 'blue', label='x(t)')
# plt.xlabel('t')
# plt.grid()
# plt.show()


