'''
Some useful utilities.
'''
# import yaml
import json
from datetime import datetime
import os
from functools import wraps
import numpy as np
import logging
from scipy import signal

now = datetime.now()
formatted_date_time = now.strftime('%Y%m%d-%H%M%S')


def save_dict(file_name: str, data: dict):
    '''
    Save the given data to a file.
    '''
    directory = os.path.dirname(file_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.close()


def vectorize(func):
    '''
    A vectorizing function to wrap any single value
    function(float to float for instance) to behave
    like a numpy function such as `np.sin`, which
    returns an array when given an array and returns
    a scalar value when given a scalar.
    '''

    @wraps(func)
    def wrapper(x, *args, **kwargs):
        if np.isscalar(x):
            return func(x, *args, **kwargs)
        else:
            vectorized_func = np.vectorize(func)
            return vectorized_func(x, *args, **kwargs)

    return wrapper

def default_logger(name: str = __name__, level: int = logging.INFO):
    '''
    Set up a default logger.
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    return logger

def now_as_seed()->int:
    """Get current time as a seed for random number generator.
    Returns:
        int: Current time in microseconds since epoch.
    """
    return int(datetime.now().timestamp() * 1e6)

def compute_asd_welch(
    x, dt, nperseg=None, noverlap=None,
    window='hann', detrend='constant', safe_min_points=256
):
    # 一维化
    x = np.asarray(x); x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError(f"x 应为一维，当前 shape={x.shape}")

    fs = 1.0 / float(dt)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"非法 dt={dt}，导致 fs={fs}")

    # 清洗 NaN/Inf
    mask = np.isfinite(x)
    if not mask.all():
        x = x[mask]
    N = x.size
    if N < safe_min_points:
        raise ValueError(f"有效数据点数过少：{N}，至少需要 {safe_min_points}")

    # 自适应 nperseg（目标 ~8~16 段）
    if (nperseg is None) or (nperseg > N):
        target = max(N // 12, 256)
        nperseg = 2 ** int(np.floor(np.log2(min(target, N))))
        nperseg = max(256, min(nperseg, N))

    # 合法化 noverlap
    if (noverlap is None) or (noverlap >= nperseg):
        noverlap = nperseg // 2
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)

    # Welch
    f, psd = signal.welch(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        detrend=detrend, return_onesided=True, scaling='density'
    )
    asd = np.sqrt(psd)

    # 自检（使用 np.trapezoid 消除弃用警告）
    x_check = x - (np.mean(x) if detrend == 'constant' else 0.0)
    var_time = np.var(x_check, ddof=0)
    var_spec = np.trapezoid(psd, f) if len(f) > 1 else np.nan

    info = {
        "fs": float(fs), "N": int(N), "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "segments_used": int(np.floor((N - noverlap) / (nperseg - noverlap))),
        "window": window, "detrend": detrend,
        "var_time": float(var_time), "var_spec": float(var_spec),
        "var_ratio_spec_over_time": float(var_spec / var_time) if var_time > 0 else np.nan,
        "df": float(f[1] - f[0]) if len(f) > 1 else np.nan
    }
    return f, asd, psd, info


def _band_average_asd_from_spectrum(
    f, asd, fmin, fmax, *, method="equivalent_flat", inclusive=True
):
    f = np.asarray(f).ravel(); asd = np.asarray(asd).ravel()
    if f.shape != asd.shape:
        raise ValueError(f"f 与 asd 形状不一致：{f.shape} vs {asd.shape}")
    if f.size < 2:
        raise ValueError("f 长度过短，无法形成频带")

    lo, hi = (fmin, fmax) if fmin <= fmax else (fmax, fmin)
    lo = max(lo, f.min()); hi = min(hi, f.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"非法频带：[{fmin}, {fmax}] 与数据范围 [{f.min()}, {f.max()}] 不相交")

    idx = (f >= lo) & (f <= hi) if inclusive else (f > lo) & (f < hi)
    if not np.any(idx):
        raise ValueError("选定频带内没有有效频点")

    f_sel = f[idx]; asd_sel = asd[idx]

    if method == "mean":
        return float(np.mean(asd_sel))
    elif method == "median":
        return float(np.median(asd_sel))
    elif method == "equivalent_flat":
        # 等效平坦 ASD = sqrt(平均 PSD) = sqrt( (∫PSD df) / 带宽 )
        if f_sel.size == 1:
            return float(asd_sel[0])  # 退化：单 bin
        psd_sel = asd_sel**2
        area = float(np.trapezoid(psd_sel, f_sel))  # ∫PSD df
        bandwidth = float(f_sel.max() - f_sel.min())
        return float(np.sqrt(area / bandwidth))
    else:
        raise ValueError("method 仅支持 'equivalent_flat' | 'mean' | 'median'")


def band_asd(
    x, dt, fmin, fmax, *,
    method="equivalent_flat",
    # 下面这些可选参数透传给 Welch，保持简洁默认即可
    nperseg=None, noverlap=None, window='hann', detrend='constant',
    safe_min_points=256
):
    """
    入口简化版：传入时域噪声序列 x、采样间隔 dt、频带 [fmin, fmax]，
    直接返回该频带内的“平均 ASD”。

    默认 method='equivalent_flat'（推荐），可改为 'mean' 或 'median'。
    """
    f, asd, _, _ = compute_asd_welch(
        x, dt, nperseg=nperseg, noverlap=noverlap,
        window=window, detrend=detrend, safe_min_points=safe_min_points
    )
    return _band_average_asd_from_spectrum(f, asd, fmin, fmax, method=method, inclusive=True)

if __name__ == '__main__':
    @vectorize
    def unit_pulse(x, offset: float = 0)-> float:
        if x == offset:
            return 1.
        return 0.

    t = np.linspace(0, 1, int(1e6), endpoint=False)
    print(unit_pulse(t).dtype)
    print(unit_pulse(t, 1e-6))
