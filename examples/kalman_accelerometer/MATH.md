# 数学推导与符号字典

本文件对应任务书第 11—19 节，是阶段一的交付物：符号定义、连续→离散推导、
相关噪声下的条件预测，以及实现侧的数值证据。所有公式与任务书冻结的形式一致，
实现代码中的行号注释指向对应公式编号。

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
