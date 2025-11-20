'''
Simple plot.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
FIG_TITLE_FONT_SIZE = 16
FIG_SIZE = (8,6)

def default_time_plot_fig(title: str = 'Time Series Plot', xlabel: str = 'Time [s]', ylabel: str = ''):
    """
    Creates and returns a default matplotlib figure and axis for time series plots.

    The figure is sized to 16x9 inches, with grid enabled, legend positioned at the upper right,
    and the x-axis labeled as 'Time [s]'.

    Returns:
        tuple: A tuple containing the matplotlib Figure and Axes objects (fig_time, ax_time).
    """
    fig_time, ax_time = plt.subplots(figsize=FIG_SIZE)
    ax_time.grid(True)
    ax_time.legend(loc='upper right')
    ax_time.set_xlabel(xlabel)
    if ylabel != '':
        ax_time.set_ylabel(ylabel)
    fig_time.suptitle(title, fontsize=FIG_TITLE_FONT_SIZE)
    return fig_time, ax_time

def default_noise_plot_fig(title: str = 'Noise power'):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.grid(True)
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.set_xlabel('Frequency [Hz]')
    fig.suptitle(title, fontsize=FIG_TITLE_FONT_SIZE)
    return fig, ax

def default_freq_plot_fig(title: str = 'Frequency Response'):
    """
    Creates a default matplotlib figure for frequency response plots with two subplots: power and phase.

    Returns:
        tuple: A tuple containing the figure object and a tuple of axes (ax_power, ax_phase).
            - fig_freq (matplotlib.figure.Figure): The created figure.
            - (ax_power, ax_phase) (tuple of matplotlib.axes.Axes): The axes for power and phase plots.
    """
    fig_freq, (ax_power, ax_phase) = plt.subplots(2,
                                                  1,
                                                  figsize=FIG_SIZE,
                                                  sharex=True)
    ax_power.grid(True)
    ax_phase.grid(True)
    ax_power.set_xscale('log')
    ax_phase.set_xscale('log')

    # 为对数 x 轴添加次刻度（每个十进制区间 1-9）
    ax_power.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(1, 10), numticks=12))
    ax_phase.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(1, 10), numticks=12))
    ax_power.minorticks_on()
    ax_phase.minorticks_on()

    # 在横坐标上显示次网格线
    ax_power.grid(which='minor', axis='x', linestyle=':')
    ax_phase.grid(which='minor', axis='x', linestyle=':')

    # ax_power.set_ylabel('dB')
    ax_phase.set_xlabel('Frequency [Hz]')
    ax_power.legend(loc='upper right')
    ax_phase.legend(loc='upper right')
    fig_freq.suptitle(title, fontsize=FIG_TITLE_FONT_SIZE)
    return fig_freq, (ax_power, ax_phase)


def plot(
    x_data: np.ndarray,  # 自变量数据
    y_data: np.ndarray,  # 因变量数据
    label: str = 'Data',
    ax=None,  # 可选的Axes对象
    show=False,
    block=False,
):
    '''
    Simple plot.
    '''
    # 检查是否提供了Axes对象，如果没有则创建新的Figure和Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  # 获取Axes所属的Figure对象

    ax.plot(x_data, y_data, label=label)  # 使用提供的x_data和y_data进行作图
    # ax.set_title('Data Plot')
    # ax.set_xlabel('X-Axis')
    # ax.set_ylabel('Y-Axis')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if show:
        plt.show(block=block)

    return fig, ax  # 返回Figure和Axes对象
