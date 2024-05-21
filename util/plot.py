import numpy as np
import matplotlib.pyplot as plt

def plot(
    x_data: np.ndarray,  # 自变量数据
    y_data: np.ndarray,  # 因变量数据
    label: str = 'Data',
    ax=None, # 可选的Axes对象
    show=False,
    block=False,
):
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
