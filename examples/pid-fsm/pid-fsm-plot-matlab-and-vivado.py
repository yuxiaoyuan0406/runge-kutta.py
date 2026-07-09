# plot fsm simulation data from matlab and vivado
import argparse
import json
import sys

import simpy
import numpy as np
import matplotlib
import os
from typing import Callable
from tqdm import tqdm

sys.path.append('.')
import module
from module import SpringDampingSystem
from module import SystemState
from module import C2V
from module import ElecFeedback
import util

def zoh_resample(t_src, y_src, t_dst):
    """
    将 y_src(t_src) 按零阶保持方式重采样到 t_dst 上。

    对每个 t_dst[i]，寻找满足：
        t_src[k] <= t_dst[i] < t_src[k+1]
    的 k，然后返回 y_src[k]。
    """
    t_src = np.asarray(t_src)
    y_src = np.asarray(y_src)
    t_dst = np.asarray(t_dst)

    idx = np.searchsorted(t_src, t_dst, side="right") - 1

    # 只保留落在源数据时间范围内的目标点
    valid = (idx >= 0) & (idx < len(t_src))

    y_dst = np.empty(len(t_dst), dtype=y_src.dtype)
    y_dst[valid] = y_src[idx[valid]]

    return y_dst, valid

def main():

    # read from files
    data_root = '/home/dennis/Program/runge-kutta.py/data/20260514PID-fsm'
    ## matlab
    ### in
    data = np.genfromtxt(
        os.path.join(data_root, "matlab_ref_input.csv"),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )
    m_in_time = data['time_s']
    m_in_ref  = data['ref_in']
    # mask_4ms  = m_in_time < 4e-3
    # m_in_time = m_in_time[mask_4ms]
    # m_in_ref  = m_in_ref [mask_4ms]
    m_ref = util.Signal(
        m_in_ref, 
        t=m_in_time, 
        label='MATLAB ref_in'
    )
    ### out
    data = np.genfromtxt(
        os.path.join(data_root, "matlab_pid_output.csv"),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )
    m_out_time = data['time_s']
    m_out_u_out= data['u_out']
    # mask_4ms   = m_out_time < 4e-3
    # m_out_time = m_out_time[mask_4ms]
    # m_out_u_out= m_out_u_out [mask_4ms]
    m_out = util.Signal(
        m_out_u_out,
        t=m_out_time,
        label='MATLAB u_out'
    )
    ## vivado
    ### in
    data = np.genfromtxt(
        os.path.join(data_root, "tb_pid_fsm_sin_input.csv"),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )
    v_in_time = data['time_ns'].astype(np.float64)
    v_in_time /= 1e12
    v_in_ref  = data['ref_in']
    # mask_4ms  = v_in_time < 4e-3
    # v_in_time = v_in_time[mask_4ms]
    # v_in_ref  = v_in_ref [mask_4ms]
    v_ref = util.Signal(
        v_in_ref, 
        t=v_in_time, 
        label='vivado ref_in'
    )
    ### out
    data = np.genfromtxt(
        os.path.join(data_root, "tb_pid_fsm_sin_output.csv"),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )
    v_out_time = data['time_ns'].astype(np.float64)
    v_out_time /= 1e12
    v_out_u_out= data['u_out']
    # mask_4ms   = v_out_time < 4e-3
    # v_out_time = v_out_time [mask_4ms]
    # v_out_u_out= v_out_u_out[mask_4ms]
    v_out = util.Signal(
        v_out_u_out, 
        t=v_out_time, 
        label='vivado u_out'
    )
    
    m_ref_zoh, valid = zoh_resample(m_in_time, m_in_ref, v_in_time)
    m_ref_zoh = util.Signal(
        m_ref_zoh,
        t=v_in_time[valid], 
        label='MATLAB ref_in_zoh'
    )
    ref_diff = m_ref_zoh - v_ref

    m_out_zoh, valid = zoh_resample(m_out_time, m_out_u_out, v_out_time)
    m_out_zoh = util.Signal(
        m_out_zoh,
        t=v_out_time[valid], 
        label='MATLAB u_out_zoh'
    )
    out_diff = m_out_zoh - v_out


    
    
    fig, ax = util.plot.default_time_plot_fig('ref_in compare', ylabel='ref_in')
    # m_ref.plot_step(ax=ax, show=True)
    m_ref_zoh.plot_step(ax=ax, show=True)
    v_ref.plot_step(ax=ax, show=True)

    fig, ax = util.plot.default_time_plot_fig('u_out compare', ylabel='u_out')
    # m_out.plot_step(ax=ax, show=True)
    m_out_zoh.plot_step(ax=ax, show=True)
    v_out.plot_step(ax=ax, show=True)

    fig, ax = util.plot.default_time_plot_fig('ref_in diff')
    ref_diff.plot_time_domain(ax, show=True)
    fig, ax = util.plot.default_time_plot_fig('u_out diff')
    out_diff.plot_time_domain(ax, show=True)

    util.plot.plt.show()

if __name__ == "__main__":
    main()
