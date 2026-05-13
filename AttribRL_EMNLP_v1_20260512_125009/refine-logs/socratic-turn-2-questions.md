# Socratic Turn 2 — Questions

1. EMA update rule on non-probe steps: `raw` 是上一 probe normalized cf / 上一时刻 EMA / 不更新？
2. Absolute KL floor / null-distractor calibration（防 batch 内全 0 时相对高被奖励）
3. First-decision-token KL 如何定位到 argument-slot（evidence-dependent）而非 tool-choice？
4. D_t LLM judge: "agent 看见"还是 "obs 中含 evidence"？non-evidence boilerplate 复制是否 D_t=1？
5. Ablation 如何实证 gain 来自 action-sensitivity 而非 dense lexical shaping / 强 outcome curriculum?
