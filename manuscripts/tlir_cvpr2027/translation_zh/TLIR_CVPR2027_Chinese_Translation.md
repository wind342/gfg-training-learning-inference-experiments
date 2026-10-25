---
title: "跨神经系统的统一训练—学习—推理—强化动力学"
subtitle: "CVPR 2027 会议版论文 · 忠实中文译稿"
author: "原稿署名：匿名 CVPR 投稿（Anonymous CVPR Submission）"
lang: zh-CN
geometry: margin=2.35cm
fontsize: 11pt
linestretch: 1.32
colorlinks: false
---

> **译稿说明**：本稿逐节翻译 **TLIR — CVPR 2027 会议版最新修订稿**，以 `tlir-cvpr2027-format/manuscripts/tlir_cvpr2027/main.tex` 与对应正式 PDF 为依据，而非后来改写的 Science/Gemini 版。依据最新英文版的章节结构与内容同步更新，保留科学主张、实验数据、公式及研究顺序。论文内图形沿用会议版原图（图内英文不另行改绘），图题与图注已翻译；参考文献书目信息按原英文保留。

# 摘要

九项 nanoGPT 实验首先建立了训练—学习—推理（Training–Learning–Inference，TLI）开环动力学。训练使参数—优化器状态演化：同一个实际发生的更新，在不同接收状态下可能产生不同的有限幅度功能响应。学习使分布式功能支撑发生持久重组，当目标跨越其读出边界时，相应能力状态发生改变。推理在持久状态保持冻结的条件下，调用并组合已学得的功能支撑。

TL-E05 中的响应因素分析直接导出了一个预测实际更新后即时目标边界结果的算法。该算法在 5,088 个 nanoGPT 确认集目标上取得 91.43% 的准确率和 91.49% 的四类宏平均召回率。随后，我们在另外五种神经系统中检验这些 TLI 关系及预测方法：ResNet-18、扩散 U-Net、路由式 Vision-MoE、图神经网络和硬脉冲循环网络。

最后，在第七种系统——一个已掌握四项技能的共享 GRU 策略——中，我们研究了持续集中正确正反馈如何重组功能支撑并削弱其他已掌握能力。重新分配反馈后，其他技能的平均准确率从 70.31% 恢复至 99.48%，同时被强化的技能得到保持。

强化（Reinforcement）通过将推理产生的后果与后续训练行动相连接，使实验建立的 TLI 开环动力学闭合，形成统一的训练—学习—推理—强化（Training–Learning–Inference–Reinforcement，TLIR）闭环动力学。

# 1. 引言

训练、泛化、能力形成和推理是神经网络研究的核心问题。不同领域针对这些问题发展出了具有影响力的理论。

在**优化理论**中，神经切线核（Neural Tangent Kernel，NTK）通过网络雅可比矩阵及其对应的核，描述特定训练状态下参数变化所引起的输出变化［Jacot 等，2018］。稳定性边缘（Edge of Stability）研究训练行为与 Hessian 最大特征值、学习率和损失轨迹之间的关系［Cohen 等，2021］。锐度感知最小化（Sharpness-Aware Minimization，SAM）利用参数邻域内的损失变化构造训练更新，以改善泛化能力［Foret 等，2021］。在这些理论中，**当前状态**通过参数以及参数的局部导数或曲率表示；**实际更新**由优化器产生；**响应**表现为模型输出或损失的变化。

在**表征学习与泛化**研究中，频谱偏置（Spectral Bias）考察函数或核模态在不同频率上的学习速度［Rahaman 等，2019］。隐式偏置（Implicit Bias）研究优化器为何选择特定解，例如可分数据上的最大间隔方向［Soudry 等，2018］。神经坍缩（Neural Collapse）测量类别特征均值、类内差异和分类器几何结构，揭示训练准确率饱和后仍在持续发生的表征变化［Papyan 等，2020］。信息瓶颈理论研究输入、隐藏表征与目标之间的互信息，包括与任务有关的信息如何保留及压缩［Tishby 和 Zaslavsky，2015］。这些研究中的**内部组织**可以表现为频谱模态、优化器选择的解、特征几何或信息结构；**功能变化**则体现为学习速度、解的演化和预测表现。

在**能力形成**研究中，Grokking 描述模型拟合训练样本与获得较强测试泛化能力之间存在较长延迟的现象［Power 等，2022］。随后的电路研究追踪这一延迟期间内部计算结构的形成与重组［Nanda 等，2023］。双重下降（Double Descent）研究测试误差如何在插值区域附近上升，然后随着模型规模、训练时间或数据条件变化再次下降［Belkin 等，2019；Nakkiran 等，2020］。这些研究通过训练阶段或插值状态衡量**当前状态**，通过电路与解的演化衡量**内部变化**，通过准确率和泛化误差衡量**可观察能力**。

在**持续学习**中，新任务的训练降低模型在旧任务上的表现时，就会发生灾难性遗忘。弹性权重巩固（Elastic Weight Consolidation，EWC）使用旧任务的 Fisher 信息估计参数重要性，并惩罚重要参数的变化［Kirkpatrick 等，2017］。梯度情景记忆（Gradient Episodic Memory，GEM）利用新任务梯度与旧任务梯度之间的内积约束新更新［Lopez-Paz 和 Ranzato，2017］。正交梯度下降（Orthogonal Gradient Descent，OGD）把新梯度投影到近似保留旧任务输出的方向上［Farajtabar 等，2020］。在这些方法中，**已有状态**包括先前学得的参数和对函数敏感的方向；**更新**是受约束的新任务动作；**相互作用**通过重叠的梯度或功能子空间发生；**能力变化**以保持、遗忘或迁移来衡量。

在**推理研究**中，注意力机制通过查询、键和值计算依赖输入的信息权重［Vaswani 等，2017］。上下文学习研究在模型参数不变的情况下，提供在上下文中的示例与任务信息如何影响预测［Brown 等，2020］。思维链提示研究中间推理步骤对复杂任务表现的影响［Wei 等，2022］。**当前状态**是已经训练的模型；**输入条件**包括查询、上下文和中间计算；**内部计算**随注意力和激活状态变化；**功能表现**由产生的预测来观察。

在**强化与反馈驱动训练**中，基于人类反馈的强化学习（RLHF）利用偏好与奖励信号调整模型行为［Ouyang 等，2022］。自训练利用模型自身预测生成进一步的训练数据［Xie 等，2020］。递归模型坍缩研究考察连续多代模型使用前代生成的数据训练时，低概率模式如何丢失［Shumailov 等，2024］。这里，**反馈**通过奖励、偏好或生成数据分布提供；**训练动作**是由此产生的更新；随后的**功能变化**通过行为、准确率及生成分布观察。

这些研究使用不同的数学对象——核、Hessian 矩阵、梯度、类别特征、互信息、任务分数、注意力权重与奖励信号——但仍可辨认出六类反复出现的要素：**当前状态、实际更新、功能响应、内部组织、可观察能力和反馈**。TLI 核心关系是在开展这种跨领域比较之前通过实验确立的；此处识别反复出现的变量，是为了帮助读者理解理论，而不是描述理论最初的发现过程。

![会议版图 1：TLIR 四个平级角色与反馈回路（原图保留）](figures/fig1.png){width=95%}

**图 1｜TLIR 的四个平级角色。** TLI 核心在六种神经系统中接受检验；RL-E05/E06 则通过共享 GRU 策略中的受控反馈重分配来检验强化过程。

# 2. TLI 开环动力学的实验建立

建立 TLI 的九项 nanoGPT 实验涉及的原生干预、受控比较和独立验证，远比会议论文篇幅所能完整呈现的更加广泛。本节概述其关键实验发现及由此形成的理论关系。完整实验协议、运行记录、负面结果和独立验证程序均见配套仓库。

## 2.1 训练：真实更新与接收状态

在已注册的训练步骤，参数—优化器状态为 $Z_t=(\theta_t,\omega_t)$，真实参数更新为 $\Delta\theta_t=\theta_{t+1}-\theta_t$。

TL-E01 观察到能力的形成、下降与恢复，这些变化无法仅用损失、准确率或训练步数充分表征。TL-E02 进一步表明，暂停有效的参数—优化器状态演化会推迟能力形成，而梯度裁剪阈值的对照干预未能复现这一效应。

TL-E03 通过完整更新／跳过更新分支及相互更新移植，建立了**相同的真实更新在不同接收状态下产生不同功能响应**这一关系：

$$
R_q(\theta_A,\Delta\theta_B)\quad\text{与}\quad R_q(\theta_B,\Delta\theta_B). \tag{1}
$$

其中 $R_q(\theta,\Delta)=m_q(\theta+\Delta)-m_q(\theta)$。独立的优化器记忆交换实验考察历史优化器状态如何影响更新的形成：

$$
\Delta^{(a,b)}=\operatorname{OptStep}(\theta_t,g_t,\omega_{a,b}). \tag{2}
$$

TL-E04 测量实际更新的完整有限幅度功能响应：

$$
r_{t,q}(\alpha)=m_q(\theta_t+\alpha\Delta\theta_t)-m_q(\theta_t). \tag{3}
$$

实验识别出近线性、饱和、加速、回折及符号反转等响应形式。随后，TL-E05 识别了三类主要响应条件因素：目标边界状态（F1）、目标特定的更新几何（F3），以及参数—优化器接收状态（F5）。

## 2.2 学习：分布式支撑与能力边界

TL-E06 通过可逆组件干预，建立了分布式功能支撑的证据，包括组件必要性、备用支撑、替代、协同作用和失效余量。

随后，TL-E07 考察真实训练更新如何使这种功能支撑发生持久重组：

$$
\Delta P_{t,q}=P_q(\theta_t+\Delta\theta_t)-P_q(\theta_t). \tag{4}
$$

在 72 个实验区段中，记录了 1,636 次有效的支撑分配转移和 228 次主要支撑交接。

TL-E08 通过目标身份对齐的读出边界，将这些内部变化与可观察能力联系起来。更新前后边界余量（margin）的符号分别确定能力保持、能力丧失、尚未获得，以及获得或恢复四种状态。由此，学习被识别为持久的功能重组，而能力转移则由目标特定读出边界的跨越决定。

## 2.3 推理：冻结状态下调用已学得的支撑

INF-E01 考察了 13 条 nanoGPT 训练历史和 52 个检查点。在持久学习状态冻结的条件下，实验测量了依赖查询条件的组件效应、非加性的功能支撑交互，以及将已训练组件替换为其精确形成前版本所引起的变化。

在固定的已学习状态 $L^*$ 下，推理调用并组合已学得的功能支撑：

$$
S_{q,c}^{\mathrm{active}}=\Pi_{q,c}(L^*),\qquad
y_{q,c}=H_{q,c}(S_{q,c}^{\mathrm{active}}). \tag{5}
$$

通过以下公式测量组件之间的非加性交互：

$$
I_{ij,q}=f_{\mathrm{full}}-f_{-i}-f_{-j}+f_{-i,-j}. \tag{6}
$$

而已学习版本的依赖性通过以下方式检验：

$$
D_{j,q}=f(q;\theta^*)-f(q;\theta^{(j\leftarrow\mathrm{pre})}). \tag{7}
$$

精确恢复检验证实，上述干预没有持久改变已学习状态。实验由此建立了推理是依赖查询条件，对学习所形成的功能支撑进行调用与组合的过程。

综合这九项实验，相互连接的训练—学习—推理关系得以建立，形成 TLI 开环动力学。

# 3. 预测算法与跨系统验证

## 3.1 理论导出的目标边界预测

TL-E05 的响应因素分析识别出三类主要条件因素：当前目标边界状态（F1）、目标特定的更新几何（F3），以及参数—优化器接收状态（F5）。这些发现直接导出了 TL-P01：一种在真实更新形成后，预测即时目标边界结果的算法。

对于目标类别 $y$、竞争类别 $c$ 和隐藏表征 $h$，令 $b_c=w_y-w_c$，$\Delta b_c=\Delta w_y-\Delta w_c$。利用沿实际更新方向的一阶和二阶方向导数 $h'=Dh[\Delta\theta_t]$ 与 $h''=D^2h[\Delta\theta_t,\Delta\theta_t]$，预测器计算：

$$
\begin{aligned}
\widehat g_{y,c}={}&b_c^\top h+\Delta b_c^\top h+b_c^\top h'\\
&+\Delta b_c^\top h'+\tfrac12 b_c^\top h''.
\end{aligned} \tag{8}
$$

当 $\min_{c\ne y}\widehat g_{y,c}>0$ 时，预测目标正确。将预测的更新后正确性与已知的更新前状态结合，即可得到四类能力状态转移。预测使用已经形成的真实更新及目标响应发生前的信息，而不读取更新后的真实目标输出。

## 3.2 跨系统验证

随后，我们在另外五种神经系统中检验已建立的 TLI 关系及其预测结果。表 1 汇总了本研究使用的七种系统，其中包含第四节单独研究强化的 GRU 系统。

**表 1｜七种神经系统：六条 TLI 实验链及一组 GRU 反馈实验。**

| 系统（优化器） | 任务／已注册的组件 |
|:--|:--|
| nanoGPT（Adam） | 循环式下一词元预测；4 个注意力／MLP 分支 |
| ResNet-18（SGD momentum） | CIFAR-100 分类；4 个残差阶段 |
| 扩散 U-Net（AdamW） | CIFAR-10 去噪；4 条 U-Net 路径 |
| Vision-MoE（AdamW） | CIFAR-100 分类；4 个路由专家 |
| GNN（RMSprop） | 图可达性；4 个消息传递模块 |
| 脉冲 SNN（Adagrad） | 8×8 数字识别；4 个脉冲模块 |
| GRU（RL-E05/E06；AdamW） | 四技能反馈；4 个共享隐藏状态组 |

另外五种 TLI 系统均接受了真实更新干预、接收状态比较、有限幅度功能响应、组件联盟支撑测量、目标边界判定以及冻结推理检验。跨系统实现对四个已注册组件的全部 16 种联盟进行评估，并计算精确 Shapley 效应：

$$
\phi_{j,q}=\sum_{C\subseteq\mathcal C\setminus\{j\}}
\frac{|C|!(3-|C|)!}{4!}
\left[v_q(C\cup\{j\})-v_q(C)\right]. \tag{9}
$$

其中 $v_q(C)$ 表示激活组件联盟 $C$ 时的目标边界余量。

ResNet、扩散 U-Net 和 Vision-MoE 分别开展的 42、18 和 36 次参数接收状态交换，全部超过各自预先注册的响应差异阈值。三种系统中，出现有限幅度非线性响应的目标—更新记录占比分别为 32.23%、19.25% 和 35.71%；超过指定支撑重分配阈值的比例分别为 91.41%、97.02% 和 80.43%。其冻结推理实验进一步验证了依赖查询条件的功能支撑、非加性交互，以及对组件精确训练版本的依赖。

GNN 在三个随机种子中复现了主要 TLI 关系，但其中一个种子未达到已注册的非线性响应频率判据（实际为 1.56%）。硬脉冲 SNN 则在四个随机种子中通过全部十二项已注册的 TLI 判据，将实验结果扩展到了具有不连续脉冲动力学的系统。

![图 2：六系统 TLI 实验链（保留英文会议版原图）](figures/fig2.png){width=95%}

**图 2｜各系统内部的训练—学习—推理实验链。** 一个 GNN 种子满足九项已注册判据中的八项，其非线性响应出现率低于指定阈值；SNN 在四个种子中均通过十二项已注册判据。

预测算法也在上述系统中接受了检验，并依据各系统任务对方向响应计算作相应适配。表 2 报告了即时更新后目标边界分类结果。

**表 2｜六种神经系统中的更新后目标边界分类结果。** 四类宏平均召回率按四种实际状态转移类别分别计算召回率后取平均；单独的高准确率并不充分。各系统的预测器细节和评价数据集有所不同。

| 系统 | 准确率（%） | 四类宏平均召回率（%） |
|:--|--:|--:|
| nanoGPT | 91.43 | 91.49 |
| ResNet-18 | 99.28 | 97.99 |
| Diffusion U-Net | 99.60 | 97.98 |
| Vision-MoE | 99.63 | 96.95 |
| GNN¹ | 98.54 | 98.62 |
| 硬脉冲 SNN¹ | 95.31 | 70.74 |

¹ GNN 和 SNN 的独立预测测试使用各自已注册的训练历史。

预测器在五种系统中取得了较高的四类宏平均召回率。在硬脉冲 SNN 中，不连续的激活阈值限制了基于局部导数的预测，使其尽管达到 95.31% 的准确率，宏平均召回率仍只有 70.74%。综合这些实验结果，TLI 核心关系在不同系统中反复出现，同时各架构的定量响应和预测近似仍存在差异。

# 4. 强化：从 TLI 开环动力学到 TLIR 闭环动力学

## 4.1 理论导出的反馈动力学预测

已经建立的 TLI 关系表明，训练会使分布式功能支撑发生持久重组，而推理则调用并组合这些已学得的功能支撑以产生行为。当推理产生的后果反复反馈至后续训练时，由此产生的更新能够进一步重组未来推理所依赖的功能支撑。

由此产生了一个预测：持续且高度集中的正确正反馈，可能逐渐使功能支撑向被强化的能力倾斜，同时削弱其他已经掌握的能力。这一预测效应源于反馈引导的功能重组，而非错误的奖励信号。

## 4.2 实验检验与结果

RL-E05/E06 在第七种系统——共享的无注意力机制 GRU 策略——中检验了这一预测。实验分支开始前，四项技能均已达到掌握状态。所有分支均从完全相同的参数—AdamW 状态出发。每个完整的两次决策过程，仅在完全正确时获得 $r=1$，否则获得 $r=0$。

RL-E05 比较了选择性反馈、均衡反馈与冻结反馈。RL-E06 进一步改变每次更新中 60 个训练回合的反馈分配，总共进行 3,200 次更新（表 3）。

**表 3｜RL-E06 每次更新分配给各能力的反馈回合数。**

| 反馈条件 | $q_0$ | $q_1$ | $q_2$ | $q_3$ |
|:--|--:|--:|--:|--:|
| 均衡反馈（Balanced） | 15 | 15 | 15 | 15 |
| 轻度集中（Mild） | 30 | 10 | 10 | 10 |
| 高度集中（High） | 45 | 5 | 5 | 5 |
| 独占反馈（Exclusive） | 60 | 0 | 0 | 0 |

RL-E05 观察到相对功能支撑集中、跨技能干扰，以及依赖组件历史版本的能力权衡。不过，其更严格的预注册时间先后判据未能通过。

在 RL-E06 中，全部 12 个正式实验种子均表现出随反馈集中程度变化的有序趋势。被强化技能的平均支撑份额，从均衡反馈条件下的 25.54% 增至独占反馈条件下的 35.20%；其他技能的平均边界余量（margin）则从 9.474 降至 3.962。轻度和高度集中反馈已使这些余量下降，但尚未降低准确率。与均衡反馈相比，独占反馈使其他技能的准确率下降了 39.06 个百分点。

随后，从独占反馈第 800 次更新时的完全相同状态，建立了两个恢复分支。重新均衡反馈使其他技能准确率从 70.31% 恢复至 99.48%，被强化技能仍保持 100%。只训练其他三项技能则使它们的准确率恢复至 99.83%，但先前被强化技能的准确率下降至 89.06%。

这些实验表明，在所测试的共享系统中，正确反馈的分配方式能够改变持久功能支撑的组织方向，影响已经掌握的能力，并通过重新分配反馈逆转这些变化。

## 4.3 TLIR 闭环动力学

强化（Reinforcement）将推理产生的后果与后续训练行动相连接。由此产生的训练更新进一步重组已学得的功能支撑，而这些变化又影响未来的推理及其后果。

这一反馈闭合过程，使通过实验建立的 TLI 开环动力学发展为统一的训练—学习—推理—强化（Training–Learning–Inference–Reinforcement，TLIR）闭环动力学。

# 5. 对正面与负面研究结果的统一解释

**问题与发现顺序。** 另一项针对历史研究中 **21 组正面／负面结果对**的比较，探讨共同的**接收状态—动作—响应—支撑—边界—反馈关系**，是否能同时解释局部理论取得成功的条件，以及限制这些理论适用范围的研究发现。下面的例子展示这种共同解释。

**核动力学与特征学习。** 在相应的无限宽度惰性训练状态下，NTK 动力学提供严格的局部训练描述［Jacot 等，2018］；而特征学习参数化可以产生不同的极限行为［Yang 和 Hu，2021］。TLI 将两者定位于接收状态与参数化方式不同的动作—响应状态中：冻结核投影有效，并不意味着所有有限振幅训练轨迹都保持其特征不变。两类结果处于不同的训练状态。

**尖锐极小值与有效的锐度感知训练。** 保持函数不变的重新参数化可以改变原始参数锐度，却不改变泛化表现［Dinh 等，2017］。然而，SAM 可以通过优化邻域鲁棒的损失，在特定任务中改善表现［Foret 等，2021］。这组结果的共同解释在于：原始参数几何并非完整的已学习功能状态，而锐度感知干预仍然可以有益地改变**实际训练动作**。因此，一个不具有参数化不变性的解释标量，可以与一种有效的局部控制方法同时存在。

**压缩是一种有用机制，而非普遍必要条件。** 在部分架构和测量条件下，关于普遍压缩阶段的主张不能成立［Saxe 等，2018］；与此同时，适当控制信息瓶颈可以得到有效的泛化保证［Kawaguchi 等，2023］。TLI 中的持久功能重组不要求 $I(X;T)$ 单调下降，但压缩可以是在接收状态、噪声和优化目标等具体条件下出现的一种结果。由此预测：**压缩的有效性有其条件范围，而它并非普遍必要。**

**延迟涌现与掌握能力之后的持续学习。** 在 Grokking 中，即使训练表现已经饱和，测试能力仍可能突然出现，而内部计算结构在这一过程中经历可以识别的阶段性演化［Nanda 等，2023］。神经坍缩研究同样记录到训练误差消失之后特征几何仍然持续变化［Papyan 等，2020］。TLI 将持久的功能支撑形成与离散的目标读出边界区分开来。因此，内部支撑持续形成与外部可观察状态突然转变，是同一测量区分下可以共存的结果。

**遗忘、能力保持与恢复。** 持续学习可能破坏旧能力，而 EWC 等方法能够保护某些重要性较高的参数方向［Kirkpatrick 等，2017］。在 TLI 中，旧目标的有符号响应取决于新动作与旧目标功能敏感支撑之间的重叠关系，以及目标边界尚余的安全余量。GRU 反馈分叉提供了匹配干预的例子：集中的**正确**反馈侵蚀了其他已掌握的能力，而重新均衡的反馈则在很大程度上使其恢复。

**有益的自训练与递归坍缩。** 在适当的数据混合条件下，基于伪标签的自训练能够提高准确率［Xie 等，2020］；而反复用模型生成样本替代真实数据，可能使少数模式的尾部分布逐渐丢失［Shumailov 等，2024］；在研究过的特定条件下，持续积累真实数据可以防止这种退化［Gerstgrasser 等，2024］。二者遵循反馈与数据之间的不同关系：推理生成的样本本身既不是天然有益的，也不是天然有害的；重要的是它保留了哪些信息，以及后续引发怎样的训练动作。对于保留率为 $0<\rho<1$ 的某个保留不足模式，一个有条件的示意性递推为：

$$
q_{t+1}=(1-\alpha)\rho q_t+\alpha q_{\mathrm{real}}. \tag{10}
$$

当 $\alpha=0$ 时，该模式发生收缩；当 $\alpha>0$ 且 $q_{\mathrm{real}}>0$ 时，真实数据输入提供一个正的固定点下限。这一递推描述的是某种数据模式在连续生成中反复保留不足的情况。

**理论先行的外推。** 除了历史研究之间的比较，TLIR 关系还被用于在检索相关文献**之前**，推导条件性预测及其检验条件。例如，指数尾损失下能力形成后的最大间隔选择［Soudry 等，2018］，以及不断演化的状态在稳定性边缘处的稳定性［Cohen 等，2021］。这些比较识别了局部数学结果出现所需的特定状态、动作和边界条件。未来的实验可以直接干预这些条件，并检验相应的预测转变。

# 6. 结论

本文通过跨系统的系统性实验，揭示了贯穿神经网络训练、学习、推理与强化过程的动力学关系。从能力的形成、调用到丧失与恢复，TLIR 将可观察的性能变化与内部功能组织及其演化机制建立起可检验的因果联系。这使神经网络的能力演化不再只是黑箱中的性能现象，而成为能够通过精确实验干预与定量预测加以研究的动力学过程。

# 大语言模型使用声明（LLM Usage Statement）

OpenAI ChatGPT 被用于研究构思与讨论、协助实验协议设计与代码开发／审查、解释实验记录，以及论文起草、编辑和 LaTeX 排版。作者对研究设计、全部科学主张和所报告的结果负责。

# 参考文献

以下按会议版参考文献原有的英文著录保留，不另行改变文献信息。

1. M. Belkin, D. Hsu, S. Ma, and S. Mandal. Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS*, 116 (32): 15849–15854, 2019.

2. T. B. Brown et al. Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 2020.

3. J. M. Cohen et al. Gradient descent on neural networks typically occurs at the edge of stability. *Proc. ICLR*, 2021.

4. L. Dinh, R. Pascanu, S. Bengio, and Y. Bengio. Sharp minima can generalize for deep nets. *Proc. ICML*, pages 1019–1028, 2017.

5. M. Farajtabar, N. Azizan, A. Mott, and A. Li. Orthogonal gradient descent for continual learning. *Proc. AISTATS*, pages 3762–3773, 2020.

6. P. Foret, A. Kleiner, H. Mobahi, and B. Neyshabur. Sharpness-aware minimization for efficiently improving generalization. *Proc. ICLR*, 2021.

7. A. Geiger et al. Causal abstraction: A theoretical foundation for mechanistic interpretability. *J. Mach. Learn. Res.*, 26 (83): 1–64, 2025.

8. M. Gerstgrasser et al. Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data. *ICML Workshop on Foundation Models in the Wild*, 2024.

9. A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: Convergence and generalization in neural networks. *Advances in Neural Information Processing Systems*, 2018.

10. K. Kawaguchi et al. How does information bottleneck help deep learning? *Proc. ICML*, 2023.

11. J. Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks. *PNAS*, 114 (13): 3521–3526, 2017.

12. D. Lopez-Paz and M. A. Ranzato. Gradient episodic memory for continual learning. *Advances in Neural Information Processing Systems*, pages 6467–6476, 2017.

13. P. Nakkiran et al. Deep double descent: Where bigger models and more data hurt. *Proc. ICLR*, 2020.

14. N. Nanda, L. Chan, T. Lieberum, J. Smith, and J. Steinhardt. Progress measures for grokking via mechanistic interpretability. *International Conference on Learning Representations*, 2023.

15. L. Ouyang et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 2022.

16. V. Papyan, X. Y. Han, and D. L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. *PNAS*, 117 (40): 24652–24663, 2020.

17. A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022. arXiv:2201.02177, 2022.

18. N. Rahaman et al. On the spectral bias of neural networks. *Proc. ICML*, pages 5301–5310, 2019.

19. A. M. Saxe et al. On the information bottleneck theory of deep learning. *Proc. ICLR*, 2018.

20. I. Shumailov et al. Ai models collapse when trained on recursively generated data. *Nature*, 631: 755–759, 2024.

21. D. Soudry et al. The implicit bias of gradient descent on separable data. *J. Mach. Learn. Res.*, 19 (70): 1–57, 2018.

22. N. Tishby and N. Zaslavsky. Deep learning and the information bottleneck principle, 2015. in IEEE Information Theory Workshop, 2015, arXiv:1503.02406.

23. A. Vaswani et al. Attention is all you need. *Advances in Neural Information Processing Systems*, pages 5998–6008, 2017.

24. J. Wei et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 2022.

25. Q. Xie, M.-T. Luong, E. Hovy, and Q. V. Le. Self-training with noisy student improves imagenet classification. *Proc. CVPR*, pages 10687–10698, 2020.

26. G. Yang and E. J. Hu. Tensor programs iv: Feature learning in infinite-width neural networks. *Proc. ICML*, pages 11727–11737, 2021.
