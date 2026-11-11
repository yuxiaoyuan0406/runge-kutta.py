# 数学推导与符号字典

本文件对应两份任务书的第 11—20 节：第一阶段（V1.1）的零阶保持（ZOH）五状态
模型，以及第二阶段（V2.1）的一阶保持（FOH）六状态模型。第 1—10 节是阶段一的
符号定义、连续→离散推导、相关噪声下的条件预测与数值证据；第 11 节起是二阶段
的增量推导、退化关系、可观测性结果和两类先验的对比数据。所有公式与任务书冻结
的形式一致，实现代码中的注释指向对应公式编号。

## 1. 符号字典

| 符号 | 维度 | 含义 | 单位 |
|---|---|---|---|
| $x, v$ | 标量 | 质量块位移、速度 | m, m/s |
| $a$ | 标量 | 外部加速度 | m/s² |
| $u$ | 标量 | PID 输出（观测，同时是真实反馈量） | 控制单位 |
| $e$ | 标量 | 数字误差 $e=K_s x+n$ | 数字误差单位 |
| $I, h$ | 标量 | PID 积分状态、历史误差 $h[k]=e[k-1]$ | 控制单位, 数字误差单位 |
| $X$ | 5×1 | $[x,v,a,I,h]^\top$ | 混合 |
| $\mu_k, P_k$ | 5×1, 5×5 | 以 $U[k-1]$ 为条件的先验 | 混合 |
| $\mu_k^+, P_k^+$ | 5×1, 5×5 | 以 $U[k]$ 为条件的后验 | 混合 |
| $A_m$ | 2×2 | $\begin{bmatrix}0&1\\-k/m&-c/m\end{bmatrix}$ | 1/s |
| $B_u, B_a$ | 2×1 | $[0,K_u/m]^\top$, $[0,1]^\top$ | 1/kg, 1 |
| $\Phi$ | 2×2 | $\exp(A_m T_s)$ | — |
| $\Gamma_u,\Gamma_a$ | 2×1 | $\int_0^{T_s}\exp(A_m\tau)B_{u,a}\,d\tau$ | — |
| $Q_m$ | 2×2 | 机械离散过程协方差 | m², m²/s … |
| $W_m$ | 2×2 | $B_a q_b B_a^\top$ | m²/s⁴ |
| $q_F, q_b$ | 标量 | 双边力 PSD、等效加速度 PSD | N²/Hz, (m/s²)²/Hz |
| $q_a$ | 标量 | 加速度每采样增量方差 | (m/s²)² |
| $r, R$ | 标量 | $\sigma_n^2$、观测方差 $\alpha^2 r$ | 控制单位² |
| $\alpha,\beta$ | 标量 | $K_p+K_d$、$-K_d$ | — |
| $K_s$ | 标量 | $k_{cv}k_{adc}$ | 数字/m |

## 2. 连续模型

$$m\ddot x+c\dot x+kx=F_{fb}+m\,a+F_n,\qquad F_{fb}=K_u u$$

注意加速度项为 $+m\,a$。取 $s_m=[x,v]^\top$：

$$\dot s_m=A_m s_m+B_u u+B_a a,\qquad W_m=B_a q_b B_a^\top,\quad q_b=q_F/m^2$$

$q_F$ 与 $q_b$ 是同一随机力的两种单位，不可相加。

## 3. 精确零阶保持离散化

一次 4×4 增广指数同时得到全部确定性矩阵，不需要 $A_m$ 可逆：

$$M=\begin{bmatrix}A_m&B_u&B_a\\0&0&0\\0&0&0\end{bmatrix},\qquad
e^{MT_s}=\begin{bmatrix}\Phi&\Gamma_u&\Gamma_a\\0&1&0\\0&0&1\end{bmatrix}$$

由 $B_u=(K_u/m)B_a$ 立即得到核对式 $\Gamma_u=(K_u/m)\Gamma_a$，
实现中在 `model._check_gamma_relation` 里强制检查。

## 4. 连续白噪声到离散协方差

$$\begin{bmatrix}\Phi&Z\\0&\Phi^{-\top}\end{bmatrix}=
\exp\left(\begin{bmatrix}A_m&W_m\\0&-A_m^\top\end{bmatrix}T_s\right),\qquad
Q_m=Z\Phi^\top$$

$\Phi^{-\top}$ 只是理论记号，程序中不出现求逆。PSD 约定在 `config._convert_noise`
集中处理：双边 PSD 直接作为强度；单边 PSD 除以 2；单边 ASD 先平方再除以 2。

## 5. 五状态联合模型

$$X[k+1]=A X[k]+G u[k]+L n[k]+\xi[k],\qquad
u[k]=H X[k]+\alpha n[k]$$

$$A=\begin{bmatrix}
\phi_{11}&\phi_{12}&g_{a1}&0&0\\
\phi_{21}&\phi_{22}&g_{a2}&0&0\\
0&0&1&0&0\\
K_iK_s&0&0&1&0\\
K_s&0&0&0&0
\end{bmatrix}$$

其中第 1、2 行为机械行，第 3 行为 $a$，第 4 行为积分 $I[k+1]=I[k]+K_i e[k]$，
第 5 行为 $h[k+1]=e[k]$。对应地

$$G=\begin{bmatrix}\Gamma_u\\0\\0\\0\end{bmatrix},\quad
L=\begin{bmatrix}0\\0\\0\\K_i\\1\end{bmatrix},\quad
H=\begin{bmatrix}\alpha K_s&0&0&1&\beta\end{bmatrix},\quad
Q_0=\mathrm{blockdiag}(Q_m,q_a,0,0)$$

## 6. 为什么不能套用标准 Kalman 更新

$u[k]$ 同时是观测和真实反馈量：同一个 $n[k]$ 既进入当前观测，又通过
$I[k+1]$、$h[k+1]$ 和下一周期的机械状态继续传播。把 $u[k]$ 当成"无噪声且与
观测误差独立的控制输入"，或者先做标准更新再假定 $n[k]$ 独立，都会得到错误的
条件均值与协方差。

闭环形式（仅用于建立联合分布）：

$$F=A+GH,\qquad D=L+\alpha G,\qquad
Q_{cl}=Q_0+rDD^\top,\qquad N_{cl}=\alpha r D$$

## 7. 实现采用的主路径

$$
\nu_k=u[k]-H\mu_k,\qquad S_k=HP_kH^\top+R \tag{14-1}
$$

$$
K_k=P_kH^\top/S_k,\qquad \mu_k^+=\mu_k+K_k\nu_k \tag{14-2}
$$

$$
P_k^+=(E-K_kH)P_k(E-K_kH)^\top+K_kRK_k^\top \tag{14-3}
$$

$$
C_k=AP_kH^\top+\alpha rL,\qquad J_k=C_k/S_k \tag{14-6}
$$

$$
\mu_{k+1}=A\mu_k+Gu[k]+J_k\nu_k \tag{14-7}
$$

$$
B_k=A-J_kH,\quad \ell_k=L-\alpha J_k,\qquad
P_{k+1}=B_kP_kB_k^\top+r\ell_k\ell_k^\top+Q_0 \tag{14-9}
$$

(14-9) 是正项形式，避免 (14-8) 的大数相减。注意 $\mu_{k+1}$ 与 $P_{k+1}$
都由**先验** $\mu_k,P_k$ 计算，且机械预测不额外加入 $\Gamma_uR\Gamma_u^\top$：
$u[k]$ 在给定 $U[k]$ 后已是确定量，读出噪声的信息通过条件化进入后验。

### 审查用等价形式

联合条件形式：

$$
\mu_{k+1}=F\mu_k+C_{cl,k}\nu_k/S_k,\qquad
P_{k+1}=V_k-C_{cl,k}C_{cl,k}^\top/S_k \tag{14-4,14-5}
$$

其中 $V_k=FP_kF^\top+Q_{cl}$、$C_{cl,k}=FP_kH^\top+N_{cl}$。

后验噪声形式（$\eta_k=\alpha r\nu_k/S_k$，$\rho_k=r-\alpha^2r^2/S_k$，
$T_k=-K_k\alpha r$）：

$$
\mu_{k+1}=A\mu_k^+Gu[k]+L\eta_k \tag{15-2}
$$

$$
P_{k+1}=AP_k^+A^\top+Q_0+L\rho_kL^\top+AT_kL^\top+LT_k^\top A^\top \tag{15-3}
$$

`kalman.reference_predict_joint` 与 `kalman.reference_predict_posterior`
实现了这两条审查式，`verification.check_prediction_equivalence` 在 100 组随机
合法先验上比较三者，最大相对偏差 $2.5\times10^{-16}$。

## 8. 初值与索引

$k=0$ 时存储 $[\hat x[0],\hat v[0],\hat a[0],\hat I[0],e[-1]]^\top$ 的先验；
第一次 `step` 接收 $u[0]$ 后先输出 $X[0|0]$，再存 $X[1|0]$。若只知道 $I[-1]$，
用 $T_{init}=E_5+K_i e_4e_5^\top$ 变换：$X_0=T_{init}Z$，$P_0=T_{init}\mathrm{Cov}(Z)T_{init}^\top$，
这会同时产生 $I[0]$ 与 $h[0]$ 的交叉协方差。实现见 `config.build_config`
的 `init_mode='k_minus_1'` 分支。

## 9. 可观测性与可检测性

对闭环对 $(F,H)$ 构造 $O=[H;HF;\dots;HF^4]$，按先验标准差缩放坐标
$T=\mathrm{diag}(\sqrt{\mathrm{diag}(P_0)})$，在 $z=T^{-1}X$ 下做 SVD 判秩。
可检测性用 PBH：对所有 $|\lambda|\ge1$ 的 $F$ 特征值，
$\mathrm{rank}[\lambda E-F;H]=5$。同时报告 $\ker O$ 是否含有加速度方向
（$c_a=[0,0,1,0,0]$ 的投影范数）。

解析退化点（任务书 17.2）在 `model.observability_report` 中作为 `notes` 输出：
$K_s=0$、$K_i=0$ 的静态不可观测方向、$K_d=0$ 时 $h$ 的特征值 0 模态。

## 10. 数值证据

`main.py check` 在 `configs/algebraic-check.json`（**合成系数**）上的一次运行：

| 检查 | 结果 |
|---|---|
| 增广指数 vs 独立数值积分 | 相对偏差 < 1e-12 |
| 自由质点极限 (19-1) | 通过，$A_m$ 奇异亦无需求逆 |
| Van Loan vs Lyapunov 数值积分 | 相对偏差 < 1e-9，$Q_m$ 半正定 |
| 三条预测式等价 | 最大相对偏差 2.5e-16 |
| 蒙特卡洛条件一致性（2e5 样本） | 残差协方差相对误差 2.0e-3，与创新交叉协方差 1.2e-3，创新标准差比 1.0015 |

蒙特卡洛检验按任务书要求由**同一次** $n[k]$ 生成观测与机械状态，而不是各生成
一次；否则相关噪声被掩盖，检验失去意义。

以上只证明实现的代数正确性，不构成任何器件性能结论。

---

# 第二阶段：FOH（一阶保持）加速度先验

## 11. FOH 机械离散化与 Jerk 输入列

在 $[t_k,t_{k+1})$ 内 Jerk 恒为 $j[k]$，外部加速度随时间线性变化
$a(\tau)=a[k]+j[k]\tau$，$\tau\in[0,T_s)$。PID 控制量仍按零阶保持，$u(\tau)=u[k]$。
机械方程不变：

$$m\ddot x+c\dot x+kx=K_u u+m\,a+F_n$$

把区间内的确定性传播写成一个 5×5 常值矩阵指数（V2.1 式 F12），增广向量为
$[s_m,a,j,u]^\top$，分块大小依次为 2、1、1、1：

$$
M=\begin{bmatrix}
A_m & B_a & 0 & B_u\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0
\end{bmatrix},
\qquad
e^{MT_s}=
\begin{bmatrix}
\Phi & \Gamma_a & \Gamma_j & \Gamma_u\\
0 & 1 & T_s & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
$$

因此

$$
s_m[k+1]=\Phi s_m[k]+\Gamma_a a[k]+\Gamma_j j[k]+\Gamma_u u[k]+w_m[k],
\qquad
\Gamma_j=\int_0^{T_s}e^{A_m(T_s-\tau)}B_a\,\tau\,d\tau
$$

**核函数是 $T_s-\tau$，不是 $\tau$。** 由 $ds/d\tau=A_m s+B_a a(\tau)$ 与
$a(\tau)=a_0+j_0\tau$ 直接积分得到；自由质点 $A_m=\begin{bmatrix}0&1\\0&0\end{bmatrix}$
时 $\Gamma_j=[T_s^3/6,\ T_s^2/2]^\top$，而错误核会给出 $T_s^3/3$。等价恒等式

$$\Gamma_j=T_s\Gamma_a-\int_0^{T_s}\tau\,e^{A_m\tau}B_a\,d\tau$$

作为第二条独立核对途径（`verification.check_foh_discretization`）。实现只做一次
`scipy.linalg.expm`，不要求 $A_m$ 可逆。

机械噪声的进入方式没有改变，因此 $Q_m$ 与阶段一完全相同（V2.1 §12.2）：

$$W_m=B_aq_bB_a^\top,\qquad
\exp\left(\begin{bmatrix}A_m&W_m\\0&-A_m^\top\end{bmatrix}T_s\right)
=\begin{bmatrix}\Phi&Z\\0&\Phi^{-\top}\end{bmatrix},\qquad Q_m=Z\Phi^\top$$

## 12. 六状态联合模型

状态顺序按 V2.1 §18 取 $X=[x,v,a,j,I,h]^\top$，$h[k]=e[k-1]$：

$$
A=\begin{bmatrix}
\phi_{11}&\phi_{12}&g_{a1}&g_{j1}&0&0\\
\phi_{21}&\phi_{22}&g_{a2}&g_{j2}&0&0\\
0&0&1&T_s&0&0\\
0&0&0&1&0&0\\
K_iK_s&0&0&0&1&0\\
K_s&0&0&0&0&0
\end{bmatrix}
$$

$$
G=\begin{bmatrix}\Gamma_u\\0\\0\\0\\0\\0\end{bmatrix},
\quad
L=\begin{bmatrix}0\\0\\0\\0\\K_i\\1\end{bmatrix},
\quad
H=\begin{bmatrix}\alpha K_s&0&0&0&1&\beta\end{bmatrix},
\quad
Q_0=\begin{bmatrix}Q_m&0\\0&0\end{bmatrix}\oplus q_j e_je_j^\top
$$

其中 $e_j$ 是第四个单位向量，$Q_0$ 只在 $(4,4)$（从 1 开始计数）含 $q_j$，加速度
对角元为 0。**FOH 不读取 $q_a$**：$q_a$ 是 ZOH 的加速度随机游走方差，$q_j$ 是
Jerk 的边界增量方差，二者单位不同，数值相等也不代表先验强度相同。

$A$ 的第 3、4 行体现 V2.1 §11.1 的边界顺序：区间末先用**旧** Jerk 外推加速度
$a[k+1]=a[k]+T_sj[k]$，再对 Jerk 施加增量 $j[k+1]=j[k]+w_j$，$w_j$ 只通过 $Q_0$
进入，不回灌刚结束区间的机械运动。因此 $q_j$ 当步只增加 Jerk 的方差，加速度的
$T_s^2q_j$ 要到下一步才出现——`verification.check_foh_noise_timing` 分别从滤波
一步差分与 $Ae_j=[\Gamma_j^\top,T_s,1,0,0]^\top$ 两条途径核对。

## 13. 条件递推

递推公式与阶段一逐条相同，只是维数变成 6（V2.1 §18）：所有右端都使用**同一份**
更新前先验 $(\mu_k,P_k)$，

$$
\nu_k=u[k]-H\mu_k,\quad S_k=HP_kH^\top+R,\quad
K_k=P_kH^\top/S_k,\quad
\mu_k^+=\mu_k+K_k\nu_k
$$

$$
C_k=AP_kH^\top+\alpha rL,\quad J_k=C_k/S_k,\quad
\mu_{k+1}=A\mu_k+Gu[k]+J_k\nu_k
$$

$$
B_k=A-J_kH,\quad \ell_k=L-\alpha J_k,\quad
P_{k+1}=B_kP_kB_k^\top+r\ell_k\ell_k^\top+Q_0
$$

$$
P_k^+=(E_6-K_kH)P_k(E_6-K_kH)^\top+K_kRK_k^\top
$$

`kalman.reference_predict_joint` / `reference_predict_posterior` 保留联合式
(F30) 与后验噪声式 (F39)(F40) 作为审查式；`check_prediction_equivalence` 在 100
组随机合法六维先验上比较三条路径（最大相对偏差 2.6e-16）。当前读出噪声 $n[k]$
仍与 PID 记忆状态联合条件化：$u[k]$ 同时是观测和真实反馈量，不能当作与噪声
独立的普通控制输入；机械预测也不额外加入 $\Gamma_uR\Gamma_u^\top$。

## 14. 初值、退化和 ZOH 兼容边界

六维先验按 V2.1 §15.1 装配。沿用五维 $P_0$ 时取选择矩阵 $S$（五个单位列放在
第 1、2、3、5、6 行）：

$$P_{0,\mathrm{FOH}}=SP_0S^\top+\mathrm{diag}(0,0,0,v_j,0,0),\qquad
X_{0,\mathrm{FOH}}=SX_{0,\mathrm{ZOH}}+\text{jerk0\_hat}\cdot e_j$$

$v_j>0$ 且允许 Jerk 交叉协方差时对应 `jerk_init_mode="unknown"`；$v_j=0$ 且整行
整列为零对应 `known`。配置校验会拒绝自相矛盾的组合（known 却带方差、unknown
却零方差、装配缺少 `jerk_var0`），不允许用任意大数代替先验设计。

退化核对：当 $j_0=0$、$q_j=0$ 时 Jerk 恒为零，去掉第四状态后 FOH 在共同状态上
退化为 $q_a=0$ 的 ZOH。`check_foh_zoh_degeneracy` 用同一段 PID 序列驱动两个
估计器 120 步，共同状态的均值相对偏差 3.4e-15、协方差相对偏差 1.2e-16、创新
绝对偏差 6.6e-19、Jerk 恰为 0。该特例只用于代数一致性核对，不能替代回归测试：
第一阶段真正的验收对象是独立的 ZOH 路径。

**ZOH 回归**：`tests/fixtures/zoh-reference.npz` 由阶段一程序在 FOH 改造之前
录制，覆盖四组记录的逐点状态、协方差、创新、增益与报告指标。改造后重新运行，
四组记录的数组最大相对偏差与指标最大相对偏差均为 **0.0**（声明容差 1e-12），
即 ZOH 的算术路径逐位可复现。

## 15. 可观测性、可检测性与弱可观测

对闭环对 $(F,H)$，$F=A+GH$，构造 $O=[H;HF;\dots;HF^5]$。按 §17.4 选择物理
尺度 $T=\mathrm{diag}(\tau_x,\tau_v,\tau_a,\tau_j,\tau_I,\tau_h)$，$z=T^{-1}X$
下判秩；Jerk 尺度必须单独选取，不能机械地等于加速度尺度。设备参数
（`configs/xiong-2018-foh.json`）的结果：

| 量 | 值 |
|---|---|
| 缩放后奇异值 | 0.2606, 0.1727, 0.1101, 8.510e-3, 3.367e-5, 2.259e-7 |
| 秩（相对阈值 1e-10） | 6 |
| 条件数 | 1.15e6 |
| PBH（$\lvert\lambda\rvert\ge1$） | 全部满秩，可检测 |
| $\ker O$ 与 $c_a$、$c_j$ 的投影 | 0、0 |

**满秩不等于强可观测**：最小相对奇异值约 $8.7\times10^{-7}$，条件数 $1.15\times
10^6$，因为 $T_s=7.8\,\mu s$ 时 Jerk 只能通过 $\Gamma_j\approx T_s^3/6$ 影响位移。
因此结果按“数值上可观测但偏弱”解释：Jerk 主要依靠先验与加速度的累计效应约束，
其逐点估计不应被当作独立传感器量。若配置未给出有物理意义的 `state_scales`
（例如 Jerk 先验方差为零时默认取单位尺度），报告会把该情形标为数值弱可观测，
而不会把秩直接当作可观测的证明。

解析退化沿用 §17.2：$K_i=0$ 时六维静态不可观测方向为
$[1,0,k/m,0,-K_pK_s,K_s]^\top$（Jerk 分量为 0）；$K_d=0$ 时 $h$ 仍构成特征值 0
的不可观测模态。`check_foh_static_unobservable_direction` 在参数满足条件时
核对 $Hv=0$、$Fv=v$。

## 16. 六维实现的数值证据

`main.py check --config configs/xiong-2018-foh.json` 的一次运行（设备参数，
非合成系数）：

| 检查 | 结果 |
|---|---|
| $\Gamma_j$ 增广指数 vs $(T_s-\tau)$ 积分 | 相对偏差 4.2e-16 |
| $\Gamma_j$ vs $T_s\Gamma_a-\int\tau e^{A_m\tau}B_a d\tau$ | 相对偏差 1.2e-21 |
| 自由质点 $\Gamma_j=[T_s^3/6,T_s^2/2]$ | 偏差 0 |
| Van Loan vs Lyapunov 积分 | 相对偏差 1.6e-16，半正定 |
| 三条条件预测式（100 组） | 均值 1.0e-16，协方差 2.6e-16 |
| 蒙特卡洛（2e5 样本，同一 $n[k]$） | 协方差相对 2.5e-3，与创新交叉 6.6e-4 |
| $q_j$ 注入位置/传播方向 | 相对 1.2e-14 / 0，无加速度当步泄漏 |
| 已知零 Jerk 且 $q_j=0$ 的 ZOH 退化 | 均值 3.4e-15，协方差 1.2e-16 |
| 确定性极限（$r,q_b,q_j=0$，60 步） | 偏差 0 |

注：蒙特卡洛的样本协方差误差为 $2.5\times10^{-3}$，是 $2\times10^5$ 样本的统计
误差量级，不是实现偏差；同样本量下阶段一 ZOH 的结果为 $4.5\times10^{-3}$。

## 17. 两类先验在同一组数据上的对比

四组现有记录、统一单位、统一评价窗；FOH 的 $q_j$ 为手工填写（见 §17.2）。
误差为加速度 RMSE，单位 m/s²。

| 记录 | 窗口 | ZOH | FOH | PID 基线 | FOH/ZOH |
|---|---|---|---|---|---|
| unit_step_1 | static | 3.336e-3 | 3.295e-3 | 1.183e-2 | 0.988 |
| unit_step_1 | step | 8.386e-1 | 1.784 | 8.394e-1 | 2.127 |
| unit_step_1 | steady | 4.895e-4 | 5.655e-5 | 1.161e-2 | 0.116 |
| 125hz_sine_1 | startup | 7.586e-2 | 4.028e-2 | 2.660e-2 | 0.531 |
| 125hz_sine_1 | steady | 7.540e-2 | 3.028e-2 | 1.686e-2 | 0.402 |
| 125hz_sin_0p1 | startup | 1.275e-2 | 6.929e-3 | 1.225e-2 | 0.543 |
| 125hz_sin_0p1 | steady | 1.282e-2 | 3.540e-3 | 1.251e-2 | 0.276 |
| 12.5hz_sine_0p1 | startup | 3.477e-3 | 3.645e-3 | 1.303e-2 | 1.048 |
| 12.5hz_sine_0p1 | steady | 2.384e-3 | 3.440e-4 | 1.276e-2 | 0.144 |
| 12.5hz_sine_0p1 | detail | 2.983e-3 | 3.123e-4 | 1.228e-2 | 0.105 |

结论只能是分窗口的：阶跃瞬态 FOH 变差（2.13×），稳态与带限动态 FOH 更好
（0.10×—0.40×）。这不是“FOH 全面优于 ZOH”的证据；阶跃的确定性跳变本来就不是
Jerk 随机游走的样本路径，其 NIS 在阶跃时刻没有意义（FOH 阶跃窗 NIS 达 9e7，
ZOH 1e5），而稳态窗的 NIS 落在 1.4—3.0。

### 17.1 Jerk 诊断

| 记录/窗口 | Jerk 均值 | Jerk 标准差 | 窗内漂移 | $\bar\rho_{aj}$ |
|---|---|---|---|---|
| unit_step_1 static | -2.9e-3 | 9.9e-2 | -0.11 | 0.56 |
| unit_step_1 steady | 1.0e-2 | 1.4e-1 | -9.9e-4 | 0.83 |
| 12.5hz detail | -72.6 | 4.12 | 13.8 | 0.83 |
| 125hz_sine_1 steady | -1.68e3 | 5.14e3 | 1.35e4 | 0.79 |

静态窗的 Jerk 均值与漂移都远小于同一数据的阶跃等效值（$9.81/T_s\approx1.3\times
10^6$ m/s³），没有出现不合理漂移；12.5 Hz 明细窗的 Jerk 均值 $-72.6$ 与解析幅值
$A\cdot2\pi f=77.1$ m/s³ 同量级，但被滤波平滑，标准差仅 4.1。$\bar\rho_{aj}\approx
0.8$ 说明后验中 $a$ 与 $j$ 强相关——这是弱可观测性的直接体现，也是不能把
`jerk_hat` 当作独立测量值的原因。

### 17.2 $q_j$ 的手工选择与敏感性

程序不实现 $q_j$ 扫描或寻优（V2.1 §7）。$q_j$ 由人手工试算：对每组数据在
$\{10^{-6},10^{-3},1,10,10^2,10^3,10^4,10^5,10^6\}$ 中逐点修改配置并重跑，
按“稳态窗 RMSE 最小”选定，四组分别为
$10^{-3}$、$10^{4}$、$10^{2}$、$10^{-3}$ (m/s³)²。完整 40 行试算记录见
`reports/data/foh-qj-trial.jsonl`。

必须注明：这一选法在同一数据、同一窗口上进行，属于样本内调参，会高估 FOH 的
相对收益；因此报告同时给出全部试算值，且不主张任何“固定降噪分贝数”。若要求
严格样本外结论，需要另行提供独立标定数据。

现有四组数据全部为阶跃或正弦激励，工作区中**没有带限随机加速度记录**，V2.1 §7
第三条“带限随机加速度”类别因此未运行；正弦结果只作为确定性带限动态的证据，
不外推到随机激励或全部频率。

### 17.3 低通基准（RC 与 Butterworth）

`util.baseline_filters` 提供两个独立、因果、可复现的低通基准：`RCFilter` 是
$H(s)=\omega_c/(s+\omega_c)$ 在零阶保持输入下的精确离散（$y[k]=\alpha y[k-1]+
\beta x[k]$，$\alpha=e^{-2\pi f_c/f_s}$），`ButterworthFilter` 是二阶 Butterworth
的双线性离散（预畸变，$f_c$ 即数字 $-3$ dB 点）。二者只接收 m/s² 的加速度，
因此比较平台必须先做一次换算：

$$a_{\text{raw}}[k]=-\frac{K_u}{m}\,u[k],\qquad
\frac{K_u}{m}\,\text{的单位为}\ \frac{\text{m/s}^2}{\text{V}},$$

其中 $u$ 是 `data_adapter` 用 `units['u']` 从定点码换来的控制量（V）。
设备配置下 $-K_u/m=5.26084$ (m/s²)/V、$1$ code $=1.068115\times10^{-4}$ V，
合 $5.619\times10^{-4}$ (m/s²)/code；阶跃稳态窗内 $a_{\text{raw}}$ 均值与真值之比
为 $0.999997$，可作换算正确性的一次独立核对。滤波与常数缩放可交换，先换算后
滤波不改变波形，只是让数值可解释。

截止频率是“通带保真与噪声抑制”的折中，不能按采样率定（128 kHz 下抗混叠不是
问题）。传感器工作频带为 0—500 Hz：要求 500 Hz 处幅度衰减不超过 1 dB 时，
一阶 RC 需要 $f_c\approx1$ kHz（$-0.97$ dB），二阶 Butterworth 只需
$\approx700$ Hz（$-0.78$ dB）；同一点上二者的相位分别为 $-25.9^\circ$ 与
$-43.3^\circ$，而 $f_c=500$ Hz 时 Butterworth 在 500 Hz 恰好 $-90^\circ$。
因此工程上取 $f_c=1$ kHz（两滤波器同值，且与模块文档默认值 $f_s/128$ 一致）。

三组正弦已用 1 s 复仿真重算（128001 点，稳定窗 10—990 ms），阶跃记录仍是原来的
8 ms 数据；加速度 RMSE（m/s²，稳定窗）：

| 记录（稳态窗） | ZOH | FOH | PID 原始 | RC 1 kHz | BW 1 kHz |
|---|---|---|---|---|---|
| unit_step_1（阶跃后，8 ms 记录） | 4.895e-4 | 5.655e-5 | 1.161e-2 | 5.249e-4 | 5.282e-5 |
| unit_step_1（静态 1 ms） | 3.336e-3 | 3.295e-3 | 1.183e-2 | 5.731e-4 | 4.703e-5 |
| 125hz_sine_1（1 s） | 7.591e-2 | 2.968e-2 | 1.703e-2 | 8.506e-1 | 1.242 |
| 125hz_sin_0p1（1 s） | 1.296e-2 | 3.333e-3 | 1.281e-2 | 8.507e-2 | 1.242e-1 |
| 12p5hz_sine_0p1（1 s） | 2.435e-3 | 3.443e-4 | 1.275e-2 | 8.536e-3 | 1.229e-2 |

与 8 ms 记录相比，三组正弦的比值只差 0.01—0.02（例如 125 Hz、1 g 的 FOH/ZOH 从
0.402 变为 0.391），说明结论不是短记录或统计涨落的产物。

结论同样是分条件的：信号近似恒定时低通最好（阶跃稳态 BW 5.28e-5，与 FOH 同级；
1 ms 静态窗 BW 4.70e-5 甚至优于仍在收敛的 Kalman），因为相位在那里无关紧要；
而 125 Hz、9.81 m/s² 这类“幅值大、频率高”的信号上，任何因果低通的相位滞后都会
贡献 $2A\sin(\Delta\varphi/2)$ 量级的误差（RC $7.0^\circ$、BW $10.2^\circ$ 对应
0.85 / 1.24 m/s²），比 PID 原始差一到两个数量级，而 Kalman 通过机械模型预测相位，
仍在 $7.6\times10^{-2}$ / $3.0\times10^{-2}$。截止频率的完整敏感性（250 Hz—
4 kHz 的幅频、相频、ENBW 与四组 RMSE）见 `reports/data/fc-study.jsonl` 与报告
第 7.3 节；改变 $f_c$ 只改变数值排序，不改变“静态靠平滑、动态靠模型”的结论。

### 17.4 为什么 PID 原始在 125 Hz 组反而最好（ASD 的读法）

125 Hz、1 g 组（1 s 记录，稳定窗 10—990 ms）里，PID 原始误差的 RMS 是 1.70e-2，
比 ZOH 7.59e-2、FOH 2.97e-2、RC 0.851、BW 1.242 都小；ASD 图上也几乎整条更低。
把误差投影到 125 Hz 正弦并分离残余（`evaluation.tone_decomposition`）：

| 方法 | RMSE | 125 Hz 幅值 | 相干分量 RMS | 残余 RMS |
|---|---|---|---|---|
| ZOH | 7.591e-2 | 8.999e-2 | 6.364e-2 | 4.139e-2 |
| FOH | 2.968e-2 | 4.145e-2 | 2.931e-2 | 4.672e-3 |
| PID 原始 | 1.703e-2 | 1.588e-2 | 1.123e-2 | 1.281e-2 |
| RC | 8.506e-1 | 1.203 | 8.506e-1 | 5.809e-4 |
| BW | 1.242 | 1.756 | 1.242 | 8.571e-5 |

PID 原始的误差以噪声为主（0.0128 残余，125 Hz 只有 0.0159，占 9.81 的 0.16%），
RC/BW 几乎全是 125 Hz 相干分量（相位滞后），ZOH/FOH 居中。相干分量在 ASD 上只占
125 Hz 附近的少数谱线，其余频段是残余噪声，所以“噪声型”的原始基线在低频段整条
曲线更低。这一现象与标定无关（阶跃稳态 $a_{\text{raw}}$ 均值/真值 = 0.999997），
它说明在这个闭环里 PID 输出在 125 Hz 已经是一个很好的瞬时加速度测量，任何因果
估计器只会额外引入滞后。

反过来看 12.5 Hz 组：信号慢到可以用上万点平均噪声，PID 原始无法平均（1.28e-2），
ZOH/FOH 压到 2.44e-3/3.44e-4，于是 ASD 的低频段原始最低、10 kHz 以上反而最高。
最后，原始误差的噪声谱不是白的：$u$ 的噪声为
$\alpha n[k]-K_d n[k-1]=37n[k]-28n[k-1]$ 的一阶滑动平均，低频增益 9、高频 65，
所以其 ASD 随频率上升（约 $+17$ dB 从 DC 到 Nyquist），只是在低频段被更强的
相干分量掩盖。这三点合起来解释了 ASD 图的形状，也说明不能只看“哪条曲线更低”。
