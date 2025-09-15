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
    return int(datetime.now().timestamp() * 1e6)

def compute_asd_welch(
    x,
    dt,
    nperseg=None,
    noverlap=None,
    window='hann',
    detrend='constant',
    safe_min_points=256
):
    """
    计算 ASD（以及可选 PSD），基于 Welch 方法，自动处理一维化、NaN/Inf、参数取值等。
    
    参数
    ----
    x : array-like
        时域噪声序列，可为 (N,) 或 (N,1) 等形状。
    dt : float
        采样间隔（秒），fs = 1/dt。
    nperseg : int or None
        Welch 分段长度。None 则自动估计（目标 8~16 段）。
    noverlap : int or None
        相邻段重叠点数。None 则取 nperseg//2，并确保 < nperseg。
    window : str or array
        Welch 窗函数（默认 'hann'）。
    detrend : {'constant','linear',False}
        去趋势策略。默认去均值（'constant'）。
    safe_min_points : int
        最小需求点数（清洗后数据若不足此数，给出友好报错）。

    返回
    ----
    f : ndarray
        频率轴（Hz，单边）。
    asd : ndarray
        ASD（单位：x 的单位 / sqrt(Hz)）。
    psd : ndarray（当 return_psd=True）
        PSD（单位：x 的单位^2 / Hz）。
    info : dict
        元信息（fs、nperseg、noverlap、实用段数、时域方差 vs 谱积分等）。
    """
    # 1) 一维化 + 转 numpy
    x = np.asarray(x)
    x = np.squeeze(x)            # 去掉多余维度，(N,1)->(N,)
    if x.ndim != 1:
        raise ValueError(f"x 应为一维，当前 shape={x.shape}")

    # 2) 基础检查
    fs = 1.0 / float(dt)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"非法 dt={dt}，导致 fs={fs}")

    # 3) 清洗 NaN/Inf
    mask = np.isfinite(x)
    if not mask.all():
        x = x[mask]
    N = x.size
    if N < safe_min_points:
        raise ValueError(f"有效数据点数过少：{N}（去除 NaN/Inf 之后），至少需要 {safe_min_points} 点用于稳定谱估计")

    # 4) 自适应 nperseg（目标 ~8~16 段）
    if (nperseg is None) or (nperseg > N):
        target = max(N // 12, 256)           # 目标段长：约 12 段；至少 256
        nperseg = 2 ** int(np.floor(np.log2(min(target, N))))  # 取不超过 N 的 2^k
        nperseg = max(256, min(nperseg, N))

    # 5) 合法化 noverlap
    if (noverlap is None) or (noverlap >= nperseg):
        noverlap = nperseg // 2
    if noverlap >= nperseg:   # 极端情况下再兜底
        noverlap = max(0, nperseg - 1)

    # 6) Welch 计算
    f, psd = signal.welch(
        x, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap,
        detrend=detrend, return_onesided=True,
        scaling='density'   # PSD 单位：x^2 / Hz
    )
    asd = np.sqrt(psd)

    # 7) 归一性自检（可用于 sanity check）
    # 时域方差（与 detrend 一致，'constant' 相当于减均值）
    x_check = x - (np.mean(x) if detrend == 'constant' else 0.0)
    var_time = np.var(x_check, ddof=0)
    # 谱积分（近似 var_time）
    var_spec = np.trapz(psd, f)

    info = {
        "fs": fs,
        "N": N,
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "segments_used": int(np.floor((N - noverlap) / (nperseg - noverlap))),
        "window": window,
        "detrend": detrend,
        "var_time": float(var_time),
        "var_spec": float(var_spec),
        "var_ratio_spec_over_time": float(var_spec / var_time) if var_time > 0 else np.nan,
        "df": float(f[1] - f[0]) if len(f) > 1 else np.nan
    }

    return f, asd, psd, info

if __name__ == '__main__':
    @vectorize
    def unit_pulse(x, offset: float = 0)-> float:
        if x == offset:
            return 1.
        return 0.

    t = np.linspace(0, 1, int(1e6), endpoint=False)
    print(unit_pulse(t).dtype)
    print(unit_pulse(t, 1e-6))
