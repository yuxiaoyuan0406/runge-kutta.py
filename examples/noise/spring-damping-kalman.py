'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys
import logging

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import scipy
import scipy.linalg

sys.path.append('.')
import util
import kalman

# matplotlib.use('TkAgg')
logger = util.default_logger(__name__, level=logging.INFO)


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Apply Kalman on simulation data. ')

    parser.add_argument('--data', type=str, help='Data directory to analysis.')
    parser.add_argument(
        '--out',
        type=str,
        help=
        'Output directory to save the results, relative to the data directory.',
        default='')

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

    logger.info('Using data directory `%s`.', args.data)
    with open(os.path.join(args.data, 'param.json'), 'r',
              encoding='utf-8') as f:
        logger.info('Loading parameters from `%s`.', f.name)
        param = json.load(f)
        f.close()
    # logger.info('Parameters: %s', json.dumps(param, indent=2, sort_keys=True))

    if args.out == '':
        args.out = f'kalman'
    save_path = os.path.join(args.data, args.out)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    m = param['mass']
    b = param['damping_coef']
    k = param['spring_coef']
    bm = b / m
    km = k / m
    dt = param['mechanic_dt']

    mass_block_data = os.path.join(args.data, 'zero_input')
    logger.info('Loading zero input data from `%s`.', mass_block_data)
    t = np.load(os.path.join(mass_block_data, 'time.npy'))
    disp = np.load(os.path.join(mass_block_data, 'position.npy'))
    logger.info('Reconstructing velocity and acceleration from displacement data for Kalman filter constructor.')
    velo = reconstruct_velocity(disp, dt)

    acce = reconstruct_velocity(velo, dt)

    a_in = acce + km * disp + bm * velo

    # Reconstructed state vector
    state_recon = np.vstack((disp, velo, a_in)).T

    class KFilter(kalman.KalmanBase):
        """
        Kalman filter for the spring-damping system.
        """

        def __init__(self):
            H = np.array([[1, 0, 0]], dtype=np.float64)

            # for $ dx/dt = F * x + B * a $
            # where $ x $ is the state
            # $ x = [position, velocity]^T $
            # no control input in this case
            self.F = np.array([[0, 1], [-km, -bm]], dtype=np.float64)
            self.B = np.array([[0, 1]], dtype=np.float64).T

            # for $ x_k = \Phi * x_{k-1} + \Gamma * a_{k-1} $
            # $ \Phi = exp(F * dt) $
            self.Phi = scipy.linalg.expm(self.F * dt)
            # $ \Gamma = \int_0^dt exp(F * \tau) d\tau * B $
            # $ \Gamma = F^{-1} * (exp(F * dt) - I) * B $
            # where $ I $ is the identity matrix
            # $ F^{-1} $ is the inverse of $ F $
            # $ \Gamma $ is the influence of the Input acceleration on the state
            self.Gamma = scipy.linalg.solve(
                self.F, (self.Phi - np.eye(self.F.shape[0])) @ self.B)
            # assume ZOH, so the input acceleration is constant during the time step
            # x_k = \Phi_x * x_{k-1}
            # where $ x_k = [position, velocity, input acceleration]^T $
            # $ \Phi_x = [[\Phi, \Gamma], [0, 0, 1]] $
            self.Phi_x = np.block([[self.Phi, self.Gamma], [0, 0, 1]])

            w = state_recon[1:] - state_recon[:-1] @ self.Phi_x.T
            Q_hat = np.cov(w, rowvar=False, ddof=1)
            R_hat = np.cov(disp - disp.mean(), ddof=1)
            # R_hat = param['observ_noise_level']**2 / dt
            print(f'R = {R_hat:.3e}')
            super().__init__(x0=np.zeros((3, 1)), P0=np.eye(Q_hat.shape[0])*1e-7, H=H, Q=Q_hat, R=np.array([[10]], dtype=np.float64)*R_hat)

        def state_update(self, x: np.ndarray, u) -> np.ndarray:
            """
            Update the state with the system dynamics.
            """
            return self.Phi_x @ x

        def covariance_update(self, P: np.ndarray) -> np.ndarray:
            """
            Update the covariance matrix.
            """
            # $ P_k = \Phi * P_{k-1} * \Phi^T + Q $
            # where $ Q $ is the process noise covariance
            return self.Phi_x @ P @ self.Phi_x.T + self.Q

    kf = KFilter()

    mass_block_data = os.path.join(args.data, 'nonzero_input')
    disp = np.load(os.path.join(mass_block_data, 'position.npy'))

    logger.info('Applying Kalman filter on the displacement data.')
    filtered = kf.apply_filter(disp, np.zeros_like(disp))
    logger.info('Kalman filter applied.')

    disp = util.Signal(disp, t=t, label='Displacement')
    filtered = util.Signal(filtered, t=t, label='Filtered Displacement')
    diff = filtered - disp
    diff.label = 'Difference'

    util.Signal.plot_all([disp, filtered], block=False)
    util.Signal.plot_all([diff])

    np.save(os.path.join(save_path, 'disp.npy'), filtered.x)
    kf.save_kalman_gain_history(save_path)

    plt.show()
