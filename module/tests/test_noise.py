import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 保证可以导入上级目录下的 noise.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from noise import Noise

def test_noise_psd():
    noise_power = 2.0      # 理论功率谱密度
    sample_time = 0.01     # 采样间隔
    mean = 0.0
    shape = (1,)
    seed = 42
    num_samples = 10000

    noise = Noise(noise_power, sample_time, mean, shape, seed)
    samples = np.array([noise.next() for _ in range(num_samples)]).flatten()

    # 计算功率谱密度
    freqs = np.fft.rfftfreq(num_samples, d=sample_time)

    # 正确的单位归一化：Periodogram 的公式为 S(f) = (Δt / N) * |X(f)|^2
    # 其中 Δt = sample_time，N = num_samples
    _ = np.fft.rfft(samples)
    psd = (sample_time / num_samples) * np.abs(_)**2


    print(f"理论ASD: {noise_power}")
    print(f"估算ASD均值: {np.sqrt(np.mean(psd)):.4f}")

    # 可视化
    plt.semilogy(freqs, psd)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.title('Noise Power Spectral Density')
    plt.show()

def test_noise_psd_2():
    noise_power = 2.0      # 理论功率谱密度
    sample_time = 0.01     # 采样间隔
    mean = np.array([0.,0.])
    shape = np.shape(mean)
    seed = 42
    num_samples = 10000

    noise = Noise(noise_power, sample_time, mean, shape, seed)
    # samples = np.array([noise.next() for _ in range(num_samples)]).flatten()
    samples = np.array([noise.next() for _ in range(num_samples)])

    # 计算功率谱密度
    freqs = np.fft.rfftfreq(num_samples, d=sample_time)

    # 正确的单位归一化：Periodogram 的公式为 S(f) = (Δt / N) * |X(f)|^2
    # 其中 Δt = sample_time，N = num_samples
    _ = np.fft.rfft(samples)
    psd = (sample_time / num_samples) * np.abs(_)**2


    print(f"理论ASD: {noise_power}")
    print(f"估算ASD均值: {np.sqrt(np.mean(psd)):.4f}")

    # 可视化
    plt.semilogy(freqs, psd)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.title('Noise Power Spectral Density')
    plt.show()

if __name__ == "__main__":
    test_noise_psd()
    test_noise_psd_2()
