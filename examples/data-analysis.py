# %%
'''
Author: Yu Xiaoyuan
'''
import json
import sys

sys.path.append('.')
import simpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import util

# matplotlib.use('TkAgg')

# %%
with open('data/20240723-153402-data.json', 'r') as f:
    data = json.load(f)
    f.close()
param = data['parameters']
json.dump(param, sys.stdout, indent=2)
# print(data['parameters'])

# %% [markdown]
# ## Data analysis for mass block

# %%
m = param['mass']
b = param['damping_coef']
k = param['spring_coef']
bm = b / m
km = k / m

w_n = np.sqrt(km)
f_n = w_n / (2 * np.pi)
zeta = bm / (2 * w_n)

print(f'Natural frequency: {f_n} Hz')
print(f'Damping ratio: {zeta}')

# %% [markdown]
# Break dictionary into lists.

# %%
mass_block_data = data['mass_block_state']
t = np.array(mass_block_data['time'])
dt = t[1] - t[0]
disp = np.array(mass_block_data['position'])
velo = np.array(mass_block_data['velocity'])

# %%
t_ax, power, phase = None, None, None

# %% [markdown]
# Plot data.
del data

# %%
_, t_ax = util.plot(t, disp, label='displacement', ax=t_ax)
power, phase = util.freq_and_plot(disp,
                                  dt,
                                  'displacement',
                                  log=True,
                                  power_ax=power,
                                  phase_ax=phase)

# %% [markdown]
# Create transfer function.


# %%
@util.vectorize
def unit_pulse(x: float, offset: float = 0.) -> float:
    '''
    Unit pulse function.
    '''
    if x == offset:
        return 1.
    return 0.


x = unit_pulse(t)
f, df, input_freq = util.t_to_f(x, dt, retstep=True)

# %%
w = 2 * np.pi * f
jw = (1j) * w
Hjw = 1 / (jw**2 + bm * jw + km)# / np.power(10, 15.6 / 20)

# %%
_, ht = util.f_to_t(Hjw, df, retstep=False)

# %%
power.plot(f, 20 * np.log10(np.abs(Hjw)), label='transfer function')
phase.plot(f, np.unwrap(np.angle(Hjw)), label='transfer function')
t_ax.plot(t, np.real(ht), label='transfer function')

# %%
t_ax.legend(loc='upper right')
power.legend(loc='upper right')
phase.legend(loc='upper right')
t_ax.grid()
power.grid()
phase.grid()
plt.show()
