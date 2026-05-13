# Socratic Turn 0 — Questions

(via Codex MCP, gpt-5.5 xhigh, threadId 019e17d4-3da8-7633-97d7-624fdcc569cd)

1. **Matched distractor 构造规则**：BFCL/ToolBench/SWE-Bench 中如何自动判定 distractor 与原 observation 同任务、同格式、on-support 但不含 answer evidence？
2. **KL 的计算对象**：`π_θ(·|h_t, o_t)` 是 next-token 分布、完整 action 序列分布、还是 structured tool-call action 分布？如果 action 是多 token JSON，如何 token-level normalize 以避免长 action 得到更大 KL？
3. **Delayed `R_cf-fwd,t` 的 W-step rollout**：在同一 sampled trajectory 上 forward 重算 logits，还是从 counterfactual state 重新 rollout 多条 trajectory 估计 expectation？若后续 observation 依赖被改 action，env state 如何 comparable？
4. **`RuleBasedExtract` 区分 utilization vs 表面复述**：对计算、比较、重命名、单位转换、code diff 这些 case，rouge-L 如何区分 genuine utilization 和 surface paraphrase？coverage ≥80% 是按 observation step / task instance / answer-evidence span 计？
5. **Structural defense formalism**：公式是加权和 `αR_cf + (1-α)R_lex`，不是 product/min gating。若 agent 在非 K-sparse 步大量 copy 或用高 R_lex 补偿低 R_cf，机制上如何阻止？是否需要 hard cap / copy penalty / `R_lex` conditioned on positive `R_cf`？
