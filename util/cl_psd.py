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
    7-term Blackman–Harris window (matching the given MATLAB WinBH7).
    Returns column-like 1D array of length N.
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
    # MATLAB: ix = 0:N/2-1; m = (ix+0.5)
    half = N // 2
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
        # MATLAB:
        # zflr=fliplr(z);
        # W=[z zflr(2:N/2)];
        # 注意 MATLAB 的索引是 1-based；zflr(2:...) 等价于 Python 的 z[::-1][1:...]
        zflr = z[::-1]
        # MATLAB 的 N/2 在 odd 时是 形如 5/2=2.5，但其冒号索引实际以整数长度为准。
        # 原代码的目标长度最终仍应为 N。
        # 这里按 MATLAB 的构造：取 z + (zflr 去掉第一个元素) 然后再裁到 N。
        W = np.concatenate([z, zflr[1:]])
        W = W[:N]

    return W.astype(np.float64)


