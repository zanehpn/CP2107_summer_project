# Socratic Turn 1 — Questions

1. **R_cf-ema 传播规则**: 沿 trajectory 时间维 / task instance 聚合 / observation type 全局 EMA？无 nearby cf probe 时初值与衰减？
2. **量纲校准**: KL 无上界 vs rouge-L ∈ [0,1]，取 min 前如何 normalize / clip / temperature-scale，避免单边截断？
3. **Lemma 1 中 π_copy 精确定义**: 复制原 observation 固定字符串 / 任意 obs 局部字符串 / schema-boilerplate？三者 KL 行为不同。
4. **VerbatimCopyRatio 定义**: 分母是 action token 数 / answer-evidence span token 数 / LCS？BFCL 这类合法 value copying 如何避免误杀？
5. **训练成本预算**: probe rate、batch caching、2×4090 wall-clock per epoch budget assumption？
