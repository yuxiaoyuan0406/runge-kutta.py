import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from numba import jit

# @jit(nopython=True)
def quantize(x, n_bits):
    n_levels = 2 ** n_bits
    x_min, x_max = np.min(x), np.max(x)
    
    if x_max - x_min != 0:
        x_normalized = (x - x_min) / (x_max - x_min)
    else:
        x_normalized = x
    
    step_size = 1.0 / (n_levels - 1)
    quantized = np.round(x_normalized / step_size) * step_size

    if x_max - x_min != 0:
        x_quantized = quantized * (x_max - x_min) + x_min
    else:
        x_quantized = quantized

    return x_quantized

def signal_func(t):
    return (np.sin(2 * np.pi * t) + np.sin(2 * np.pi * 2 * t)) / 2

normalize = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

if __name__ == "__main__":
    t, dt = np.linspace(-5, 5, 10000, retstep=True)
    t_high_samp_rate, dt_high_samp_rate = np.linspace(-5,5,20000, retstep=True)
    signal = signal_func(t)
    signal_high_samp_rate = signal_func(t_high_samp_rate)
    
    n_bits = 3
    quantized_signal = quantize(signal, n_bits)
    quantized_signal_high_samp_rate = quantize(signal_high_samp_rate, n_bits)
    
    figure_origin = plt.figure()
    
    plt.plot(t, signal, label='Original Signal')
    plt.plot(t, quantized_signal, label='Quantized Signal')
    plt.plot(t_high_samp_rate, quantized_signal_high_samp_rate, label='Higher sampling rate')

    plt.axis(True)
    plt.legend(loc='upper right')
    
    # plt.show(block=False)

    f_signal = np.abs(np.fft.fftshift(np.fft.fft(signal)))
    f_quantized = np.abs(np.fft.fftshift(np.fft.fft(quantized_signal)))
    f_quantized = normalize(f_quantized)
    f_quantized_high_samp_rate = np.abs(np.fft.fftshift(np.fft.fft(quantized_signal_high_samp_rate)))
    f_quantized_high_samp_rate = normalize(f_quantized_high_samp_rate)


    f = np.linspace(-.5/dt, .5/dt, len(t))
    f_high_samp_rate = np.linspace(-.5/dt_high_samp_rate, .5/dt_high_samp_rate, len(t_high_samp_rate))
    _ = f_high_samp_rate <= np.max(f)
    f_high_samp_rate = f_high_samp_rate[_]
    f_quantized_high_samp_rate = f_quantized_high_samp_rate[_]
    _ = f_high_samp_rate >= np.min(f)
    f_high_samp_rate = f_high_samp_rate[_]
    f_quantized_high_samp_rate = f_quantized_high_samp_rate[_]
    

    figure_freq = plt.figure()
    # plt.plot(f, f_signal, label='Original Signal')
    plt.plot(f, f_quantized, label=f'Quantized {n_bits} bit')
    plt.plot(f_high_samp_rate, f_quantized_high_samp_rate, label='Higher sampling rate')

    # n_bits = 6
    # quantized_signal = quantize(signal, n_bits)
    # f_quantized = np.abs(np.fft.fftshift(np.fft.fft(quantized_signal)))

    # plt.plot(f, f_quantized, label=f'Quantized {n_bits} bit')

    plt.axis(True)
    plt.legend(loc='upper right')

    
 
    plt.show(block=False)

    input('Press Enter to exit...')
    # plt.close('all')

