'''
Author: Xiaoyuan Yu
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

k = [
    1.55,
    0.038,
    -0.5,
    -0.01139 * 0.06,
    0.516,
    -2.
]

max_freq = 128 * 1e3
dt = 1 / (2*max_freq)

def h(f):
    # return k[5] - k[0] * (k[4] * (1 - np.exp(- (1j) * 2 * np.pi * f * dt)) - k[1]*k[2]) / ((1-np.exp(- (1j) * 2 * np.pi * f * dt))**2 - k[1] * k[3])
    return -k[5] - k[0] * (k[4] * (1 - np.exp(- (1j) * 2 * np.pi * f * dt))*np.exp(- (1j) * 2 * np.pi * f * dt) + k[1]*k[2]*np.exp(- (1j) * 2 * np.pi * f * dt)*np.exp(- (1j) * 2 * np.pi * f * dt)) / ((1-np.exp(- (1j) * 2 * np.pi * f * dt))**2 - k[1] * k[3]*np.exp(- (1j) * 2 * np.pi * f * dt)*np.exp(- (1j) * 2 * np.pi * f * dt))

def mec_x(f):
    return 1/(((1j)*2*np.pi*f)**2 + 2*0.00121*2*np.pi*208*((1j)*2*np.pi*f) + (2*np.pi*208)**2)

freq = np.linspace(1e-1, max_freq, 200000)
x = mec_x(freq)
h_f = h(freq)
y_f = h_f * x * 1.515e7 * 3 * 28.688 * 0.5
# err_f = 1/(1+1.354e9*y_f)
# sig_f = 4.086e7*y_f/(1+1.354e9*y_f)
err_f = 1/(1+y_f)
sig_f = y_f/(1+y_f)

# amp_err = 20 * np.log10(np.abs(err_f))
amp_err = np.abs(err_f)
ang_err = np.angle(err_f)

amp_sig = 20 * np.log10(np.abs(sig_f))
ang_sig = np.angle(sig_f)

# amp = 20 * np.log10(np.abs(h_f))
# ang = np.angle(h_f)
# ang = np.unwrap(ang)

_, (power, phase) = plt.subplots(2,1, sharex=True)
power.plot(freq, amp_err, label='Noise')
power.plot(freq, amp_sig, label='Signal')
power.legend(loc='upper right')
power.grid(True)
power.set_xscale('log')

phase.plot(freq, ang_err, label='Noise')
phase.plot(freq, ang_sig, label='Signal')
power.legend(loc='upper right')
phase.grid(True)
phase.set_xscale('log')
phase.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
phase.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val / np.pi:.2f}π'))

plt.tight_layout()
plt.show(block=False)

amp_h = 20 * np.log10(np.abs(h_f))
ang_h = np.angle(h_f)
_, (power, phase) = plt.subplots(2,1, sharex=True)
power.plot(freq, amp_h)
power.grid(True)
power.set_xscale('log')

phase.plot(freq, ang_h)
phase.grid(True)
phase.set_xscale('log')
phase.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
phase.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val / np.pi:.2f}π'))

plt.tight_layout()
plt.show(block=False)

amp_y = 20 * np.log10(np.abs(y_f))
ang_y = np.angle(y_f)

_, (power, phase) = plt.subplots(2,1, sharex=True)
power.plot(freq, amp_y)
power.grid(True)
power.set_xscale('log')

phase.plot(freq, ang_y)
phase.grid(True)
phase.set_xscale('log')
phase.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
phase.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val / np.pi:.2f}π'))

plt.tight_layout()
plt.show(block=False)



input('Press Enter to exit...')
plt.close('all')

