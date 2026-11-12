'''
Author:     Kedu Han
            Zeyu Wang
            Xiaoyuan Yu
Funtion:    用于闭环的ASD评估。
'''



"""

## Original Matlab script here.

## File: bitstreamPSDfn_analogcl.m
## Author: Zeyu Wang

*Note: 理论上该函数计算的是ASD，但上一任作者糊涂搁这写的PSD.*
*这里不做修改，但Python脚本中的注释和变量会进行对应的修正.*

function bitstreamPSDfn_analogcl(output_g, fl, fh)
% bitstreamPSDfn_closeloop
% 用于模拟闭环（连续值输出）的 PSD 评估
% 输入 output_g：workspace 导出的序列，单位已是 g
% fl, fh：用于计算噪底的频带边界(Hz)，例如 10, 300
%
% 说明：
% - 输入不是 ±1 比特流，而是连续幅值（g）
% - 不做归一化、不做单位换算、不去均值
% - 采样率固定为 128 kHz（与你当前 Simulink / 论文一致）

if nargin == 0
    error('requires input data array');
elseif nargin == 1
    fl = 20;
    fh = 200;
elseif nargin == 2
    fh = 200;
end

N  = length(output_g);
fs = 128000;                        % 固定采样率 128 kHz
Nfft = 0.512*2*fs;                  % Nfft = 1.024*fs (=131072)，保持与原作者一致

% 输入已经是 g，直接用于 PSD
x = output_g(:);

if N < Nfft
    [Psd, Freq] = periodogram(x, WinBH7(N), N, fs);
else
    [Psd, Freq] = pwelch(x, WinBH7(Nfft), Nfft/2, Nfft, fs);
end

% 噪底计算（保持原作者方式）
fband = find(Freq>=fl & Freq<=fh);
nf = 10*log10(mean(Psd(fband)));
disp(['Noise Floor (' num2str(fl) '-' num2str(fh) 'Hz): ' ...
      num2str(round(nf*10)/10)]);

% 画图
figure
semilogx(Freq, 10*log10(Psd));
set(gca, 'Xlim', [1, 1e5]);
set(gca, 'Ylim', [-200, 20]);
xlabel('Frequency (Hz)')
ylabel('1-sided Spectrum Level (dBg/\surdHz)')
grid
title(['Closed-loop output PSD, fs = ' num2str(fs) ' Hz']);
text(1.1, 10, ['Noise Floor (' num2str(fl) '-' num2str(fh) 'Hz): ' ...
    num2str(round(nf))], 'fontSize', 10, 'fontWeight', 'bold', 'interpreter', 'none');
text(1.1, 0, ['File'], 'fontSize', 10, 'fontWeight', 'bold', 'interpreter', 'none');

end

% ===== 7-term Blackman–Harris window（保持与原代码一致）=====
function W = WinBH7(N)

a0=0.61640321314050;
a1=0.98537119272586;
a2=0.49603771622007;
a3=0.14992232793243;
a4=0.02458719103474;
a5=0.0017660465148687;
a6=0.000031581188567106;

v = 2*pi/N;
ix = 0:N/2-1;
m  = (ix+0.5);

t1=cos(v*m);
t2=cos(2*v*m);
t3=cos(3*v*m);
t4=cos(4*v*m);
t5=cos(5*v*m);
t6=cos(6*v*m);

z = a0-a1*t1+a2*t2-a3*t3+a4*t4-a5*t5+a6*t6;

if rem(N,2)==0        % Even
    W=[z fliplr(z)];
else
    zflr=fliplr(z);
    W=[z zflr(2:N/2)];
end

W = W(:);
end

"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



def win_bh7(N: int) -> np.ndarray:
    """
    7-term Blackman–Harris window (DFT-even, same coefficients as MATLAB WinBH7).

    Returns a symmetric 1D array of exactly length N, for both even and odd N.

    Note
    ----
    The original MATLAB WinBH7 builds ``W = [z zflr(2:N/2)]`` from
    ``ix = 0:N/2-1``. For odd N that construction yields only ``N-2`` samples,
    so ``periodogram(x, WinBH7(N), N, fs)`` always errors out on odd-length
    records. This port fixes the odd case instead of reproducing the defect.
    For even N the output is bit-identical to the MATLAB version.
    """
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive.")

    a0 = 0.61640321314050
    a1 = 0.98537119272586
    a2 = 0.49603771622007
    a3 = 0.14992232793243
    a4 = 0.02458719103474
    a5 = 0.0017660465148687
    a6 = 0.000031581188567106

    v = 2 * np.pi / N
    # 半窗取 ceil(N/2) 个点：偶数 N -> N/2；奇数 N -> (N+1)/2（含唯一中心点）。
    # MATLAB 原文 `ix = 0:N/2-1` 在 N 为奇数时只得到 (N-1)/2 个点，
    # 再配上 `zflr(2:N/2)`（奇数时只取到 1 个点），拼出的窗长只有 N-2。
    half = (N + 1) // 2
    ix = np.arange(0, half, dtype=np.float64)
    m = ix + 0.5

    t1 = np.cos(v * m)
    t2 = np.cos(2 * v * m)
    t3 = np.cos(3 * v * m)
    t4 = np.cos(4 * v * m)
    t5 = np.cos(5 * v * m)
    t6 = np.cos(6 * v * m)

    z = a0 - a1 * t1 + a2 * t2 - a3 * t3 + a4 * t4 - a5 * t5 + a6 * t6

    if (N % 2) == 0:  # Even: W = [z fliplr(z)]
        W = np.concatenate([z, z[::-1]])
    else:
        # 奇数：z 已包含中心点，镜像时从倒数第二个元素开始取，避免中心点重复。
        # 等价于 MATLAB 想表达的 `W=[z zflr(2:N/2)]`，但长度恰好为 N。
        W = np.concatenate([z, z[-2::-1]])

    return W.astype(np.float64)


def bitstream_psd_fn_analogcl(output_g, fl=None, fh=None, *, show_plot=True):
    """
    Python equivalent of the provided MATLAB function bitstreamPSDfn_analogcl.

    Parameters
    ----------
    output_g : array-like
        Time series already in unit of g (continuous-valued output).
    fl, fh : float
        Noise-floor band edges in Hz. Defaults: fl=20, fh=200 (matching MATLAB).
    show_plot : bool
        If True, plot PSD in dB scale vs log-frequency (matching MATLAB).

    Returns
    -------
    freq : np.ndarray
        Frequency axis (Hz), one-sided.
    psd : np.ndarray
        One-sided PSD (g^2/Hz).
    nf_db : float
        Noise floor in dB, computed as 10*log10(mean(psd[fl..fh])).
    """
    if output_g is None:
        raise ValueError("requires input data array")

    # Match MATLAB nargin defaults
    if fl is None and fh is None:
        fl = 20
        fh = 200
    elif fl is not None and fh is None:
        fh = 200
    elif fl is None and fh is not None:
        # MATLAB 里不存在这种调用方式（因为 nargin=2 时是 fl, fh），这里给个合理兜底
        fl = 20

    x = np.asarray(output_g, dtype=np.float64).reshape(-1)
    N = x.size
    if N < 2:
        raise ValueError(f"at least 2 samples are required, but got N={N}.")

    fs = 128000.0
    Nfft = int(round(0.512 * 2 * fs))  # = 131072, keep author choice

    # PSD estimation (one-sided), density scaling
    if N < Nfft:
        w = win_bh7(N)
        freq, psd = signal.periodogram(
            x,
            fs=fs,
            window=w,
            nfft=N,               # MATLAB: periodogram(x, WinBH7(N), N, fs)
            detrend=False,        # "不去均值/不去趋势"
            return_onesided=True,
            scaling="density",
        )
    else:
        w = win_bh7(Nfft)
        freq, psd = signal.welch(
            x,
            fs=fs,
            window=w,
            nperseg=Nfft,         # MATLAB: WinBH7(Nfft)
            noverlap=Nfft // 2,   # MATLAB: Nfft/2
            nfft=Nfft,            # MATLAB: Nfft
            detrend=False,        # "不去均值/不去趋势"
            return_onesided=True,
            scaling="density",
        )

    # Noise floor (keep MATLAB method)
    band_mask = (freq >= float(fl)) & (freq <= float(fh))
    if not np.any(band_mask):
        nf_db = float("nan")
    else:
        nf_db = 10.0 * np.log10(np.mean(psd[band_mask]))

    print(f"Noise Floor ({fl}-{fh}Hz): {np.round(nf_db, 1)}")

    if show_plot:
        plt.figure()
        plt.semilogx(freq, 10.0 * np.log10(psd))
        plt.xlim([1, 1e5])
        plt.ylim([-200, 20])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(r"1-sided Spectrum Level (dBg/$\sqrt{Hz}$)")
        plt.grid(True, which="both")
        plt.title(f"Closed-loop output PSD, fs = {int(fs)} Hz")
        plt.text(
            1.1, 10,
            f"Noise Floor ({fl}-{fh}Hz): {int(np.round(nf_db))}",
            fontsize=10, fontweight="bold"
        )
        plt.text(
            1.1, 0,
            "File",
            fontsize=10, fontweight="bold"
        )
        plt.show()

    return freq, psd, nf_db


