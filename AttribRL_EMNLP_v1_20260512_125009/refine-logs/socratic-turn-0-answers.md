# Socratic Turn 0 — Answers (Claude)

## Q1. Matched distractor 构造规则

**总原则**: distractor 必须满足三条 filters: (a) **task-relevant**（属于同一 task type）, (b) **format-isomorphic**（schema/length 接近）, (c) **evidence-free**（不含 answer key span）。每个 benchmark 有 *benchmark-specific* 自动规则：

- **BFCL** (function call): 同 task type（如 `simple-math-call`）的池子内，从 *其他* 任务的 function-call result 中均匀采样；同时 reject criterion 是：将该 distractor 作为 obs 代入原 task 的 oracle answer extractor 后，extract 失败（即不含原 task 的 answer evidence）。Pool size ~5000 per task type。
- **ToolBench** (multi-step API): 用 same API schema 的其他 task 的 API response（不同 query parameters 返回的不同结果）；evidence-free 判定：将 distractor 中所有 string 值与原 task answer 做 token overlap < 0.3。
- **API-Bank**: same API category（calendar/weather/calculator etc.）的其他 instance 输出，evidence-free 判定同 ToolBench。
- **WebArena-Lite**: same domain 的其他页面 HTML，rule 通过 DOM-similar selectors 找；evidence-free 判定: required answer slot 缺失。
- **SWE-Bench-Lite**: 同 repo 的其他 file/diff 片段（length ratio ∈ [0.7, 1.4]）；evidence-free 判定: 不包含 bug-fix 关键 token (issue title 中的 identifier / function name)。

**自动 on-support validation 协议**（写入 paper）: 训练时记录 distractor task-validity score（gpt-5.5-mini binary "是否看起来像该 task 的合法 tool output"），report mean ± std；正常应 > 0.85，否则触发 alert 并切换 fallback (扩大 pool 或下调 K)。

## Q2. KL 的计算对象与 token-level normalization

**KL 计算单位**: 在 *structured tool-call action* 的 *first-decision-token* 分布上计算，**不是整个 multi-token JSON action 分布**。理由：(i) 整 sequence KL 随长度爆炸（reviewer 担心的 normalization 问题）；(ii) tool-call agent 的"决策点"通常在 action 的前几个 token（function name / argument key），后续 token 多为 schema-determined。

**精确定义**: 设 agent 在 step t 输出 action 序列 $a_t = (a_t^{(1)}, a_t^{(2)}, ..., a_t^{(L_t)})$。  
**首决策 token** $a_t^{(1)}$ = action 序列中**第一个分布熵 > τ** 的 token (default τ=0.5)；用 entropy threshold 找"决策点"而非简单取第一个 token，避免 schema 前缀 ("{"function_name":") 占主导。

- $R_{cf,t} = \text{KL}(p_\theta(a_t^{(1)} | h_t, o_t) \,\|\, p_\theta(a_t^{(1)} | h_t, \tilde{o}_t))$

**Mean-token KL fallback**: 若首-决策-token 难以识别（W=3 forward setting），用 length-normalized average per-token KL: $\frac{1}{L} \sum_l \text{KL}(p_\theta(a_t^{(l)} | h_t, a_t^{(<l)}, o_t) \| p_\theta(a_t^{(l)} | h_t, a_t^{(<l)}, \tilde{o}_t))$。

这是 explicit normalization 处理（回应 reviewer "长 action 得到更大 KL"）。

## Q3. Delayed `R_cf-fwd,t` 的 W-step rollout

**关键 design decision**: **同一 trajectory 上 forward 重算 logits，不重新 rollout 多条 trajectory**。这是 *off-policy importance* 风格的估计，避免 env state divergence。

精确算法：
1. 已有 sampled trajectory τ = (..., $h_t$, $o_t$, $a_{t+1}$, $o_{t+1}$, ..., $a_{t+W}$, ...)
2. 替换 $o_t \to \tilde{o}_t$，构造 counterfactual prefix $\tilde{h}_{t+1} = (h_t, \tilde{o}_t)$
3. 对原 trajectory 的后续 action sequence $a_{t+1:t+W}$（**不重新 sample**），逐步计算 counterfactual 概率 $p_\theta(a_{t+l} | \tilde{h}_{t+l})$ where $\tilde{h}_{t+l} = (\tilde{h}_{t+1}, a_{t+1:t+l-1}, o_{t+1:t+l-1})$ —— **保留原 observation 在 t+1 之后的状态**
4. KL = $\sum_{l=1}^{W} \text{KL}(p_\theta(a_{t+l}^{(1)} | h_{t+l}) \| p_\theta(a_{t+l}^{(1)} | \tilde{h}_{t+l}))$ at first-decision-token

**为什么不 re-rollout**: re-rollout 会让 env state 改变（因 tool 实际不同 → 后续 obs 不同），破坏 comparability；用 *frozen action sequence* 上的 logit 差异作为 "agent 内在 utilization 信号" 的 surrogate（标准 off-policy 解释）。

**Limitation 承认**: 这是 *internal sensitivity* 度量，不等于 full environment counterfactual。在 paper 中明确写为 "policy-level counterfactual sensitivity" 而非 "environment-level ATE"。

## Q4. RuleBasedExtract 区分 utilization vs 表面复述

**关键 design**: 不只用 surface rouge-L。Span extractor 输出 *(span_text, span_semantic_role)* tuple，rouge-L 计算 *只针对 semantic role 为 "answer-evidence" 的 span*（不是 schema 字段、不是 metadata、不是 boilerplate）。

具体：
- BFCL: extractor 仅提取 function-call result 的 `value`/`result`/`data` 字段（不提 schema key, function name, error code）
- ToolBench: 仅 API JSON response 的 leaf string values（不是嵌套 key names）
- SWE-Bench: diff 中 `+` 行的代码 + 报错信息（不是 line numbers, file headers）

**区分计算 vs 复述**: 加 *transformed answer recognizer*：
- Numeric: 若 obs 中是 "$1024"，agent reasoning 中出现 "$1024 → 应分两年付款 → 每年 $512"，则即使 $512 不在 obs 中，前置的 "$1024" 是 utilization 证据 → rouge-L 仍计 hit。
- 单位转换：若 obs 是 "200 km"，agent 输出 "124.27 miles"，则 200 必须在 reasoning 任一步出现（intermediate computation evidence）。
- Code diff: span 是 diff 中 `+`/`-` 的 token；agent reasoning 中包含相同 identifier/function 即 hit。

**Coverage ≥80%** 指: **answer-evidence span 计**（按 span instance）。pilot 验证: 在 100 task instance 上手动标记 answer-evidence span，对比 rule-based extractor recall。若 recall < 0.80，迭代扩 rule。

**Surface paraphrase 边界**: 若 agent 把整个 obs 复制到 reasoning 中（verbatim copy hacking）→ rouge-L=1 但 *evidence span coverage rate* (paper-level metric) 也 1.0；通过 Q5 中的 structural defense 阻止此 case。

## Q5. Structural defense formalism

**承认 reviewer 的批评是对的**：加权和 `αR_cf + (1-α)R_lex` 不是 *严格* 防御 verbatim copy；在非 K-sparse 步 lexical 可独自驱动 reward。

**修正后的 design**: 引入 **gated mixture** 替代 plain weighted sum：

$$U_t = D_t \cdot \min(R_{cf-\text{ema}}^{(t)}, R_{lex,t}) \cdot \text{Mask}(t)$$

其中:
- $R_{cf-\text{ema}}^{(t)}$ = K-sparse 步上计算的 $R_{cf}$ 的 **exponential moving average**，延伸到非 K-sparse 步（让每步都有 cf-aware signal）
- $\min(\cdot, \cdot)$ = **conjunctive gate**（reviewer 要求的 product/min gating）—— 强制双源同时高
- $\text{Mask}(t) = \mathbb{1}[\text{VerbatimCopyRatio}(a_t) < 0.7]$ —— **硬 cap copy penalty**: 若 agent 把 ≥ 70% obs token 直接复制到 action/reasoning，整步 reward 归零

**为什么 min 而非 product**: product 会让 small R_cf 严重 dampen reward → 早期训练 vanishing gradient；min 保留 "limiting factor" 性质 + better gradient flow。**Adaptive α 改为 EMA confidence weight**: $\alpha = \sigma(\beta \cdot |R_{cf-\text{ema}} - R_{lex}|)$，即两者不一致时偏向 cf。

**形式化 verbatim defense 引理**（写入 paper 的 §Theoretical Grounding）:

> **Lemma 1 (verbatim copy defense)**: 设 $\pi_{\text{copy}}$ 为完全 verbatim copy policy（action 复制 obs token）。则:
> (a) $R_{lex,t}(\pi_{\text{copy}}) \to 1$ 因 rouge-L 高;
> (b) 由于 copy 不依赖 obs 内容差异，$\pi_{\text{copy}}(a_t | h_t, o_t) = \pi_{\text{copy}}(a_t | h_t, \tilde{o}_t) \cdot \mathbb{1}[\text{copy-format}]$，故 $R_{cf,t}(\pi_{\text{copy}}) \to 0$;
> (c) $\text{Mask}(t) = 0$ for copy actions;
> 综合: $U_t(\pi_{\text{copy}}) = D_t \cdot \min(0, 1) \cdot 0 = 0$。

**Update**: 修正后的 mixture formula 替代原版 `α R_cf + (1-α) R_lex` 公式。这是对 reviewer Q5 的核心 mechanism upgrade。

---

## 修正总结（待集成进 round-1-refinement）

1. **§Method >Counterfactual path > matched distractor**: 增加 5 benchmark-specific 自动构造规则 + on-support validation 协议
2. **§Method > KL definition**: 改为 first-decision-token KL with entropy-threshold τ=0.5；length-normalized fallback for delayed setting
3. **§Method > Delayed extension**: 明确为 *frozen action sequence 上 logit 重算*，写明 limitation 是 "policy-level counterfactual sensitivity" 而非 environment ATE
4. **§Method > Lexical**: span extractor 输出 (text, semantic_role)，rouge-L 仅对 answer-evidence span；加 transformed answer recognizer (numeric / unit / code)
5. **§Method > Mixture**: 修正为 gated min + EMA + hard copy cap + adaptive α；增加 Lemma 1 verbatim defense formalization
