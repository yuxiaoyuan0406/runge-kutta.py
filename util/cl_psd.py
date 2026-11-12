'''
Author:     Kedu Han
            Zeyu Wang
            Xiaoyuan Yu
Function:   用于闭环输出的单边 ASD（g/√Hz）评估与噪底计算。
'''



"""

## Original Matlab script here.

## File: bitstreamPSDfn_analogcl.m
## Author: Zeyu Wang

*Note: 理论上该函数计算的是ASD，但上一任作者糊涂搁这写的PSD.*
*这里不做修改，但Python脚本中的注释和变量会进行对应的修正.*
*Python 移植版按 ASD 口径实现：返回 asd = sqrt(psd)，噪底单位 dB re 1 g/√Hz。*

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

from .plot import default_noise_plot_fig
from .signal import Signal



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


def bitstream_asd_fn_analogcl(
    output_g,
    fl=None,
    fh=None,
    *,
    fs=None,
    nfft=None,
    noverlap=None,
):
    """
    Python equivalent of the provided MATLAB function bitstreamPSDfn_analogcl
    (which is an ASD estimate despite its name), minus its plotting: render the
    result with :func:`plot_asd`.

    The spectrum is estimated as a one-sided PSD (Welch/periodogram, density
    scaling) and returned as an amplitude spectral density, ``asd = sqrt(psd)``,
    in g/sqrt(Hz).

    Parameters
    ----------
    output_g : array-like or util.Signal
        Time series already in unit of g (continuous-valued output). A
        ``util.Signal`` is accepted as well; its samples and sampling rate are
        then taken from the object.
    fl, fh : float
        Noise-floor band edges in Hz. Defaults: fl=20, fh=200 (matching MATLAB).
        `fl` must be finite and >= 0, `fh` must be finite and >= `fl`.
    fs : float, optional
        Sampling rate in Hz. Defaults to 128 kHz (the MATLAB constant), or to
        ``1/Signal.dt`` when a ``util.Signal`` is passed. Must be finite and > 0.
    nfft : int, optional
        Segment/FFT length. Defaults to ``int(round(1.024*fs))``, i.e. 131072 at
        128 kHz (the MATLAB choice). If the record is shorter than `nfft`, a
        single periodogram over the whole record is used instead, without zero
        padding (same as MATLAB's ``if N < Nfft`` branch).
    noverlap : int, optional
        Overlap between Welch segments. Defaults to ``nfft // 2``. Ignored when
        the record is shorter than `nfft`.

    Returns
    -------
    freq : np.ndarray
        Frequency axis (Hz), one-sided.
    asd : np.ndarray
        One-sided amplitude spectral density (g/sqrt(Hz)), ``asd = sqrt(psd)``.
    nf_dbg : float
        Noise floor in dB re 1 g/sqrt(Hz), computed as ``10*log10(mean(psd))``
        over the requested band, clipped to the frequency range actually covered
        by the spectrum. This is the *power* mean of the ASD (i.e.
        ``20*log10(rms(asd))``); ``20*log10(mean(asd))`` is a different, lower
        number whose gap depends on the number of independent bins in the band.
        An empty band raises instead of returning NaN; an identically zero
        spectrum (e.g. a stuck-at-zero output) yields ``-inf``.

    See Also
    --------
    plot_asd : render ``freq``/``asd`` as a dB curve on a log-frequency axis.

    Raises
    ------
    ValueError
        If the record is shorter than 2 samples, if `fs`/`nfft`/`noverlap` are
        outside their valid ranges, if `fl`/`fh` are not finite or ordered
        correctly, or if the requested band contains no spectral bin.
    """
    if output_g is None:
        raise ValueError("requires input data array")

    # 采样率来源优先级：显式 fs > Signal.dt > MATLAB 的 128 kHz。
    if isinstance(output_g, Signal):
        data = output_g.x
        if fs is None:
            fs = 1.0 / float(output_g.dt)
    else:
        data = output_g

    if fs is None:
        fs = 128000.0
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be a finite positive number, got fs={fs!r}")

    # Match MATLAB nargin defaults
    if fl is None and fh is None:
        fl = 20
        fh = 200
    elif fl is not None and fh is None:
        fh = 200
    elif fl is None and fh is not None:
        # MATLAB 里不存在这种调用方式（因为 nargin=2 时是 fl, fh），这里给个合理兜底
        fl = 20

    x = np.asarray(data, dtype=np.float64).reshape(-1)
    N = x.size
    if N < 2:
        raise ValueError(f"at least 2 samples are required, but got N={N}.")

    if nfft is None:
        nfft = int(round(0.512 * 2 * fs))  # MATLAB: Nfft = 1.024*fs (= 131072 @ 128 kHz)
    nfft = int(nfft)
    if nfft < 2:
        raise ValueError(f"nfft must be >= 2, got nfft={nfft}")

    if noverlap is None:
        noverlap = nfft // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nfft:
        raise ValueError(
            f"noverlap must satisfy 0 <= noverlap < nfft, got noverlap={noverlap}, nfft={nfft}"
        )

    # PSD estimation (one-sided), density scaling
    if N < nfft:
        # 记录不足一段：用全部数据做单段 periodogram，不补零（MATLAB 的 if N < Nfft 分支）。
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
        w = win_bh7(nfft)
        freq, psd = signal.welch(
            x,
            fs=fs,
            window=w,
            nperseg=nfft,         # MATLAB: WinBH7(Nfft)
            noverlap=noverlap,    # MATLAB: Nfft/2
            nfft=nfft,            # MATLAB: Nfft
            detrend=False,        # "不去均值/不去趋势"
            return_onesided=True,
            scaling="density",
        )

    # 对外口径是 ASD：asd = sqrt(psd)，单位 g/√Hz。
    asd = np.sqrt(psd)

    # 噪底频带校验。单边谱只覆盖 [0, fs/2]，所以：
    #  - 参数本身非法（非有限 / 负频 / 反序）直接报错；
    #  - 与谱范围只有部分交集时裁到交集，并如实打印实际使用的频带；
    #  - 交集为空时报错，绝不返回 NaN 让下游拿着 nan dB 继续跑。
    fl = float(fl)
    fh = float(fh)
    if not (np.isfinite(fl) and np.isfinite(fh)):
        raise ValueError(f"fl/fh must be finite, got fl={fl!r}, fh={fh!r}")
    if fl < 0:
        raise ValueError(f"fl must be >= 0 (one-sided spectrum), got fl={fl}")
    if fl > fh:
        raise ValueError(f"invalid band: fl={fl} > fh={fh}")

    f_lo, f_hi = float(freq[0]), float(freq[-1])
    lo, hi = max(fl, f_lo), min(fh, f_hi)
    band_mask = (freq >= lo) & (freq <= hi)
    if not np.any(band_mask):
        raise ValueError(
            f"no spectral bin in [{fl:g}, {fh:g}] Hz; the one-sided spectrum "
            f"covers [{f_lo:g}, {f_hi:g}] Hz with df={freq[1] - freq[0]:g} Hz"
        )
    # 功率平均（rms 意义下的等效平坦 ASD），不是 20*log10(mean(asd))。
    band_mean = float(np.mean(psd[band_mask]))
    if band_mean <= 0.0:
        # 全零输出（例如闭环卡死在 0）时 log10(0) 会给出 -inf 并刷 RuntimeWarning。
        nf_dbg = float("-inf")
    else:
        nf_dbg = 10.0 * np.log10(band_mean)

    print(f"Noise Floor ({lo:g}-{hi:g}Hz): {np.round(nf_dbg, 1)} dBg/√Hz")

    return freq, asd, nf_dbg


def plot_asd(
    freq,
    asd,
    nf_dbg=None,
    *,
    ax=None,
    label="",
    band=None,
    floor_db=None,
    title=None,
    show=False,
    block=False,
):
    """
    Plot a one-sided ASD as a level curve on a log-frequency axis.

    The drawn curve is ``20*log10(asd)``, i.e. dB relative to 1 g/sqrt(Hz).

    Parameters
    ----------
    freq, asd : array-like
        One-sided frequency axis (Hz) and ASD (g/sqrt(Hz)), as returned by
        :func:`bitstream_asd_fn_analogcl`.
    nf_dbg : float, optional
        Noise floor in dB re 1 g/sqrt(Hz). Drawn as a dashed horizontal line
        when finite.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; useful to overlay several runs. A new figure is
        created through :func:`util.plot.default_noise_plot_fig` when omitted.
    label : str
        Curve label (e.g. the data file or configuration name).
    band : (float, float), optional
        Noise-floor band, used only for the legend text.
    floor_db : float, optional
        Hard lower y limit. Left to matplotlib's autoscale when omitted, which
        keeps overlays of differently-scaled runs working; pass e.g. ``-120`` to
        zoom in on a quiet floor, or to clamp an all-zero (``-inf``) spectrum.
    title : str, optional
        Figure title. Defaults to one that names the inferred sampling rate.
    show, block : bool
        Show the figure after drawing, forwarding `block` to ``plt.show``.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes holding the curve.
    """
    freq = np.asarray(freq, dtype=np.float64).reshape(-1)
    asd = np.asarray(asd, dtype=np.float64).reshape(-1)
    if freq.shape != asd.shape:
        raise ValueError(
            f"freq and asd must have the same length, got {freq.shape} vs {asd.shape}"
        )
    if freq.size == 0:
        raise ValueError("freq/asd are empty.")

    if ax is None:
        if title is None:
            title = f"Closed-loop output ASD, fs = {2.0 * float(freq[-1]):g} Hz"
        fig, ax = default_noise_plot_fig(title=title)
    else:
        fig = ax.figure

    # 全零谱会 log10(0) 得到 -inf；这里是预期结果，画出来交给坐标轴裁剪即可。
    with np.errstate(divide="ignore"):
        level_db = 20.0 * np.log10(asd)
    ax.plot(freq, level_db, label=label or None)

    positive = freq[freq > 0.0]
    if positive.size:
        ax.set_xlim(left=float(positive.min()), right=float(freq[-1]))

    if floor_db is not None:
        ax.set_ylim(bottom=float(floor_db))
    if not np.any(np.isfinite(level_db)):
        # 全零谱：没有有限值可供 autoscale。
        bottom = float(floor_db) if floor_db is not None else -200.0
        ax.set_ylim(bottom=bottom, top=bottom + 10.0)

    ax.set_ylabel(r"1-sided ASD (dBg/$\sqrt{Hz}$)")

    if nf_dbg is not None and np.isfinite(nf_dbg):
        band_text = ""
        if band is not None:
            band_text = f" ({band[0]:g}-{band[1]:g} Hz)"
        ax.axhline(
            float(nf_dbg),
            linestyle="--",
            linewidth=1.0,
            color="C1",
            label=f"Noise floor{band_text}: {nf_dbg:.1f} dBg/√Hz",
        )

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")

    if show:
        plt.show(block=block)

    return fig, ax


