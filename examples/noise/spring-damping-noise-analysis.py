'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from matplotlib import pyplot as plt
import os
import scipy

sys.path.append('.')
import util
from module import Noise

# matplotlib.use('TkAgg')


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(description='Plot simulation data. ')

    parser.add_argument('--data', type=str, help='Data directory to analysis.')

    return parser.parse_args()

def reconstruct_velocity(x: np.ndarray, dt: float, window_len: int=101, polyorder: int=3) -> np.ndarray:
    """
    根据位移序列 x（1-D numpy 数组）和采样间隔 dt，
    先平滑再微分，返回速度序列 v。
    
    参数
    ----
    x : np.ndarray
        位移序列，形状 (N,)
    dt : float
        采样间隔，单位与位移一致
    window_len : int, 可选
        Savitzky-Golay 滤波窗口长度（必须为奇数且 > polyorder）
    polyorder : int, 可选
        Savitzky-Golay 多项式阶数（应 < window_len）
    
    返回
    ----
    v : np.ndarray
        速度序列，形状 (N,)
    """
    # 1) 平滑位移 —— 减少微分放大的高频噪声
    # x_smooth = scipy.signal.savgol_filter(x, window_length=window_len,
    #                          polyorder=polyorder, mode='interp')
    x_smooth = x

    # 2) 中央差分求速度
    v = np.zeros_like(x_smooth)
    # 内点： (x_{k+1} - x_{k-1}) / 2dt
    v[1:-1] = (x_smooth[2:] - x_smooth[:-2]) / (2.0 * dt)
    # 两端用一阶差分
    v[0]  = (x_smooth[1]  - x_smooth[0]) / dt
    v[-1] = (x_smooth[-1] - x_smooth[-2]) / dt

    return v

if __name__ == '__main__':
    args = argue_parser()

    with open(os.path.join(args.data, 'param.json'), 'r',
              encoding='utf-8') as f:
        param = json.load(f)
        f.close()
    print(json.dumps(param, indent=2, sort_keys=True))

    m = param['mass']
    b = param['damping_coef']
    k = param['spring_coef']
    dt = param['mechanic_dt']
    bm = b / m
    km = k / m


    # for $ dx/dt = F * x + B * a $
    # where $ x $ is the state
    # $ x = [position, velocity]^T $
    # no control input in this case
    F = np.array([[0, 1], [-km, -bm]], dtype=np.float64)
    B = np.array([[0, 1]], dtype=np.float64).T

    # for $ x_k = \Phi * x_{k-1} + \Gamma * a_{k-1} $
    # $ \Phi = exp(F * dt) $
    Phi = scipy.linalg.expm(F * dt)
    # $ \Gamma = \int_0^dt exp(F * \tau) d\tau * B $
    # $ \Gamma = F^{-1} * (exp(F * dt) - I) * B $
    # where $ I $ is the identity matrix
    # $ F^{-1} $ is the inverse of $ F $
    # $ \Gamma $ is the influence of the Input acceleration on the state
    Gamma = scipy.linalg.solve(
        F, (Phi - np.eye(F.shape[0])) @ B)
    # assume ZOH, so the input acceleration is constant during the time step
    # x_k = \Phi_x * x_{k-1}
    # where $ x_k = [position, velocity, input acceleration]^T $
    # $ \Phi_x = [[\Phi, \Gamma], [0, 0, 1]] $
    Phi_x = np.block([[Phi, Gamma], [0, 0, 1]])


    zero_input_data = os.path.join(args.data, 'zero_input')
    t = np.load(os.path.join(zero_input_data, 'time.npy'))
    disp_pure_noise = np.load(os.path.join(zero_input_data, 'position.npy'))
    # noise_data = np.vstack((disp, velo)).T

    ideal_unit_pulse = os.path.join(args.data, 'mass_block')
    ideal_unit_pulse_data = np.load(os.path.join(ideal_unit_pulse, 'position.npy'))

    input_with_noise = os.path.join(args.data, 'nonzero_input')
    input_with_noise_data = np.load(os.path.join(input_with_noise, 'position.npy'))

    # Calculate system gain (x/a) according to the ideal unit pulse response.
    trans = util.Signal(ideal_unit_pulse_data, t=t, label='Ideal unit pulse')
    mask = np.abs(trans.f) <= 100.0
    linear_part = trans.X[mask]
    system_gain = np.mean(np.abs(linear_part))

    noise_x = disp_pure_noise
    noise_v = reconstruct_velocity(noise_x, dt)
    noise_a = reconstruct_velocity(noise_v, dt)

    noise_a_in = noise_a + km * noise_x + bm * noise_v

    noise_state = np.vstack((noise_x, noise_v, noise_a_in)).T

     # ---- 2. 计算每一步的过程噪声向量 w_{k-1} ----
    # w_{k-1} = x_k - Phi_x * x_{k-1}
    w = noise_state[1:] - noise_state[:-1] @ Phi_x.T       # 形状 (N-1, 3)

    # ---- 3. 样本协方差 ----
    #   用无偏估计 (ddof=1)；np.cov 默认会再除以 (N-1)
    Q_hat = np.cov(w, rowvar=False, ddof=1)

    print('Q:')
    print(Q_hat)

    # my_noise = Noise(1e-6 * 9.81, sample_time=dt, mean=0, seed=int(util.util.now.timestamp() * 1e6))
    # noise_a_in = np.array([my_noise.next() for _ in noise_x])

    fs = 1.0 / dt
    f, Pxx = scipy.signal.welch(
        noise_a_in / 9.81, fs=fs, window='hann', nperseg=8192, noverlap=6144,
        detrend='constant', scaling='density'
    )
    # Pxx_one = Pxx[1:] * 2
    asd = np.sqrt(Pxx)

    plt.loglog(f, asd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ASD (g/sqrt(Hz))')
    plt.title('ASD of Noise')
    plt.grid(True)
    plt.show()

    mask = (f >= -100) & (f <= 100)
    asd_mean = np.mean(asd[mask])
    print(f"Mean ASD in [-100, 100] Hz: {asd_mean:.4e} g/sqrt(Hz)")

    # windows methods
    # nperseg = len(disp) // 4
    # noverlap = nperseg // 2
    # window = 'hann'
    # freqs, Pxx = scipy.signal.welch(disp,
    #                                 fs=1 / dt,
    #                                 nperseg=nperseg,
    #                                 noverlap=noverlap,
    #                                 window=window,
    #                                 nfft=None,
    #                                 detrend='constant',
    #                                 return_onesided=True,
    #                                 scaling='density')

    # mask =  (freqs >= 0) & (freqs <= 300)
    # f_band = freqs[mask]
    # Pxx_band = Pxx[mask]

    # disp = util.Signal(disp,
    #                    t=t,
    #                    color='blue',
    #                    linestyle='-',
    #                    label='Simulation displacement')

    # nois = plt.subplot()
    # nois.loglog(freqs, Pxx, color='red', linestyle='-', label='Noise PSD')
    # nois.set_xlabel('Frequency (Hz)')
    # nois.set_ylabel('Power Spectral Density (PSD)')
    # nois.set_title('Power Spectral Density of Noise')
    # nois.grid(True)

    # d_f = freqs[1] - freqs[0]
    # rms_noise = np.sqrt(np.sum(Pxx_band) * d_f)
    # asd = np.sqrt(rms_noise)
    # print(f"RMS noise: {rms_noise:.4e} m^2/Hz")
    # print(f"ASD: {asd:.4e} m/sqHz")
    # asd = asd * km / 9.81
    # print(f"ASD(acce): {asd:.4e} g/sqHz")
    # # print(f"样本方差: {np.var(disp.X):.4e}, 积分 PSD 得到的方差: {total_variance:.4e}")

    # _ = np.fft.rfft(disp.X)
    # psd = (dt / len(disp.X)) * np.abs(_) ** 2
    # asd = np.sqrt(np.mean(psd))
    # print(f"FFT ASD: {asd:.4e} m/sqHz")
    # asd = asd * km / 9.81
    # print(f"FFT ASD(acce): {asd:.4e} g/sqHz")

    # util.Signal.plot_all([
    #     disp,
    # ], title='Simulation data', block=True)
