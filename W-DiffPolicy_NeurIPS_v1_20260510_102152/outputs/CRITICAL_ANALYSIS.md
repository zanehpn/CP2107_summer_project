# Critical Analysis — W-DiffPolicy Landscape

**Date**: 2026-05-09
**Model**: gpt-5.5 (xhigh reasoning)
**Codex thread ID**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`

---

## 核心论断

> 该领域的中心结构性弱点在于：经常将 **behavior density**、**diffusion reverse-process density**、**induced MDP occupancy** 当作可互换。大部分可挖掘的裂缝都来自这种替换。

---

## Critique Manifest（13 条）

### CRITIQUE-01: KL/score-matching 行为正则保留多模态的假设未验证
- **Category**: Unverified Assumption
- **Description**: 该领域假设 KL/score-matching 行为正则在保持 diffusion policy on-support 的同时也保留行为多模态。但 score-matching BC 可以拟合主导密度同时丢失稀有条件 mode，且 Q-guidance 会进一步剪除 mode。
- **Affected Papers**: P01 Diffusion-QL, P04 BDPO, P06 RACTD, P10, P12, P13
- **Why Exploitable**: 若在控制任务上直接证明 mode loss，许多 "safe improvement" 论断从 incomplete 变为 underspecified。

### CRITIQUE-02: BDPO-style 累积 reverse-kernel KL 测量最终策略接近度的假设
- **Category**: Unverified Assumption
- **Description**: BDPO 风格的累积反向 kernel KL 被假定为度量最终策略与行为策略的接近度。但两个 diffusion 过程可能 (a) 在终端 action 分布相似下有不同 reverse path，或 (b) 在终端分布不同下有相似 local kernel penalty。
- **Affected Papers**: P01, P04 BDPO, P06 RACTD, diffusion-path 正则一般性
- **Why Exploitable**: 证明 path-regularization 部分是 parameterization artifact 将重新打开 diffusion-specific behavior constraint 的根基。

### CRITIQUE-03: ICNN-估计的 W₂ 在 offline RL 中的可靠性
- **Category**: Unverified Assumption
- **Description**: ICNN 估计的 W₂ 被当作 offline RL 中可靠的行为正则器。但 Brenier-map 的保证假设理想分布；offline RL 有有限样本、条件 state-action 分布、bootstrapped Q 误差、高维 action manifold。
- **Affected Papers**: P14 Q-DOT, P15 BWD-IQL, P17 SWFP, P18, P22 (基底)
- **Why Exploitable**: 分离真正的 OT 几何 vs 神经 OT 估计误差是可发表的——攻击 OT-regularized offline RL 的核心技术前提。

### CRITIQUE-04: 静态 OT/Brenier 理论被错误推广到 MDP 策略正则
- **Category**: Incorrect Generalization
- **Description**: 静态 OT/Brenier 理论被引入 MDP 策略正则化时，仿佛 per-state action matching 就够了。原始 scope 是固定测度间在 ground cost 下的运输；但 offline RL 在策略变化时 state occupancy 和 Bellman target 都会变。
- **Affected Papers**: P14 Q-DOT, P16 Rethinking OT, P17 W₂ trust region, P18
- **Why Exploitable**: 修正后的理论需要 occupancy-aware 或 value-aware transport，使当前 "W₂ regularization" 论断不完整。

### CRITIQUE-05: P19 KL-mode-collapse 定理从 LM 隐式推广到控制
- **Category**: Incorrect Generalization
- **Description**: P19 的 KL-mode-collapse 定理被隐式从 LMs/化学 LMs 推广到 continuous-control offline RL。但 RL mode 是 state-conditional trajectory/skill mode，不是 token-sequence mode；value guidance 会改变实际采样分布。
- **Affected Papers**: P19 (用作激励 control claim 时), P01/P04 KL 正则的批判
- **Why Exploitable**: 控制特定的验证或反例会很重要——该领域目前缺乏从生成模型 collapse 理论到 MDP 行为正则的桥梁。

### CRITIQUE-06: Diffusion 模型表现力假设从 image/video 推广到 action 生成
- **Category**: Incorrect Generalization
- **Description**: 来自 image/video 生成的 diffusion 模型表现力和样本质量假设被推广到 offline RL 的 action 生成。但在控制中，好的条件密度建模并不蕴含安全的策略改进——Q-guidance、action clipping、horizon 效应、Bellman 误差都可能扭曲采样的 action 分布。
- **Affected Papers**: P01-P13，尤其是效率导向的 P02, P03, P06, P08, P11
- **Why Exploitable**: 暴露许多 "改进" 可能是生成建模改进，而非 offline-RL 改进。

### CRITIQUE-07: Return-only 评估不能区分鲁棒多模态改进 vs 选择单一高 return mode
- **Category**: Experimental Flaw
- **Description**: Return-only 评估不能区分鲁棒多模态改进与选择一个 surviving 的高 return mode。D4RL Kitchen/AntMaze 分数和成功率通常不报告 conditional mode retention、per-mode success、occupancy drift。
- **Affected Papers**: P01-P14, P20 LOM, P21 RAMAC, NeoRL-2 比较
- **Why Exploitable**: 在 mode-sensitive 诊断下重新评估现有方法可以**改变 SOTA 解读而无需新算法**。

### CRITIQUE-08: 计算/推理/采样预算未一致归一化
- **Category**: Experimental Flaw
- **Description**: 计算、推理、采样预算未一致归一化。多步 diffusion、shortcut model、consistency distillation、Q ensembles、temporal-resolution planner 在不同 NFE、update count、model size、tuning budget 下被比较。
- **Affected Papers**: P02 EDP, P03 SORL, P04 BDPO, P06 RACTD, P07, P08, P12
- **Why Exploitable**: 一些速度/分数增益可能是 budget-allocation artifact 而非算法优越性。

### CRITIQUE-09: Offline RL 模型选择依赖特权 online validation
- **Category**: Experimental Flaw
- **Description**: Offline RL 模型选择常依赖特权 online validation 或 benchmark 后见之明。正则强度、Q-guidance scale、diffusion steps、OT weight、distillation temperature 恰恰是决定 support violation 的旋钮。
- **Affected Papers**: P01, P04, P06, P14, P17，和大多数 D4RL-tuned offline RL 论文
- **Why Exploitable**: 若方法只能在 online-tuned 正则下工作，其 offline-robustness 论断显著弱化。

### CRITIQUE-10: 数据集选择混淆多模态/数据质量/horizon/状态覆盖/奖励稀疏性
- **Category**: Experimental Flaw
- **Description**: D4RL 和 NeoRL-2 被当作真实的多模态压力测试，但它们很少隔离 number of modes、mode overlap、mode value、behavior-policy mixture composition。
- **Affected Papers**: P01-P14, P23-P27, 尤其是 NeoRL-2 "无法击败 behavior" 的结论
- **Why Exploitable**: 受控因子分解能展示失败究竟来自多模态、support 差、稀疏奖励还是 value 估计误差。

### CRITIQUE-11: OT 几何中 Euclidean 运输 cost 在控制中的语义错位
- **Category**: Cross-Domain Misfit
- **Description**: 来自生成建模/domain adaptation 的 OT 几何假设 Euclidean 运输 cost 是语义上有意义的。但在控制中，**小** action-space 移动可能跨越 contact 或 dynamics 断点，而**大** Euclidean 移动可能在 symmetry 下行为等价。
- **Affected Papers**: P14, P16-P18, P24 OTDF；通用 "W₂ smoother than KL" 论断
- **Why Exploitable**: 攻击 W₂-接近度是否对应行为安全性（而非仅仅是数学平滑性）。

### CRITIQUE-12: Domain-adaptation "对齐分布、然后迁移" 逻辑被错误引入 cross-domain offline RL
- **Category**: Cross-Domain Misfit
- **Description**: 但匹配 transition marginal 可能擦除稀有但 Bellman 相关的 transition、skill mode 或 failure case，这些往往比平均分布相似性更重要。
- **Affected Papers**: P23 PUORL, P24 OTDF, P25 Dual-Robust, P26 Beyond OOD
- **Why Exploitable**: 在表面 OT 对齐良好下证明 negative transfer 将挑战 cross-domain offline RL filtering 的核心前提。

### CRITIQUE-13: 生成建模的 diversity / mode-count 指标不直接是 RL 目标
- **Category**: Cross-Domain Misfit
- **Description**: 保留每一个行为 mode 也可能保留糟糕、不安全或不相关的 mode；如果任务有唯一最优 mode，collapse 到一个 mode 可以是合理的。
- **Affected Papers**: P20 LOM, P21 RAMAC, P19-灵感的 mode-collapse 论断, G3 风格诊断
- **Why Exploitable**: 任何严肃的 mode-preservation 论断必须定义 value-conditioned 和 safety-conditioned mode，而非纯密度 cluster。

---

## 跨 critique 的元主题

1. **密度替换 metonymy**: behavior density / diffusion reverse density / induced occupancy 被互换 → CRITIQUE-01, 02, 04, 06
2. **静态几何 vs 动态控制**: OT 是否能从静态测度推到 MDP induced occupancy → CRITIQUE-04, 11
3. **评估盲点**: return-only / NFE-不归一 / online tuning leakage / 数据集混淆 → CRITIQUE-07, 08, 09, 10
4. **生成模型 vs 控制目标的对齐**: diffusion 表现力是否等于策略安全性，diversity 是否等于 RL value → CRITIQUE-06, 13
5. **理论搬迁裂缝**: P19 KL-collapse 定理（LM）→ control 的搬迁未验证 → CRITIQUE-05
