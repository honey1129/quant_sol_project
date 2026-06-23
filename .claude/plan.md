# 优化计划：加大止损 + 震荡过滤

## 背景

当前问题：
- 自适应止损最小值 0.65%，在低波动震荡市中被频繁扫到
- 简单规则模式完全绕过 regime 过滤器，在震荡市也会开仓
- 近30天胜率仅 26%，连续亏损

## 修改方案

### 修改1：加大止损下限

**文件**: VPS 上的 `.env`

当前配置：
```
ADAPTIVE_STOP_LOSS_MIN=0.0065   # 0.65% - 太紧
ADAPTIVE_STOP_LOSS_MAX=0.022
```

修改为：
```
ADAPTIVE_STOP_LOSS_MIN=0.012    # 1.2% - 与静态STOP_LOSS一致
ADAPTIVE_STOP_LOSS_MAX=0.03     # 3.0% - 给极端波动留空间
```

同时调整止盈下限配合：
```
ADAPTIVE_TAKE_PROFIT_MIN=0.025  # 2.5%（之前1.2%）
```

**效果**: 止损从 0.65% 提升到至少 1.2%，减少被假突破扫掉的概率

---

### 修改2：简单规则模式增加震荡过滤

**文件**: `core/strategy_core.py`，`_resolve_directional_target_ratio` 方法

当前问题：简单规则模式在 line 414 直接 return，完全跳过了后面的 regime filter 逻辑。

修改：在简单规则模式返回前，增加 regime 检查：

```python
# 简单规则模式 - 直接返回固定仓位
if hasattr(self, '_simple_rule_mode') and self._simple_rule_mode:
    # ★ 新增：震荡市过滤 - 不在 range/range_high_vol 中开新仓
    if self.regime_filter_enabled and market_regime:
        regime_lower = str(market_regime).lower()
        if regime_lower in {"range", "range_high_vol"}:
            return 0.0, prob_gap, long_prob if long_prob >= short_prob else short_prob, "RegimeFilter(震荡市不交易)", None, 0.0

    # 原有逻辑继续...
```

**效果**: 当 regime 判定为震荡时，即使 EMA 显示趋势也不开仓，避免在震荡市反复被止损

---

### 修改3：同时禁止 range 市场交易

**文件**: VPS 上的 `.env`

```
REGIME_RANGE_ALLOW_TRADES=false   # 之前是 true
```

这确保即使切回 ML 模式，range 震荡也不交易。

---

## 验证方式

修改后在VPS上快速回测30天对比（已有简单规则基准 -5.49%），确认优化版本表现更好后再部署。

## 影响分析

- 交易频率会降低（震荡期不交易），从每天5-6笔降到2-3笔
- 止损更宽意味着单笔亏损可能更大，但总止损次数应大幅减少
- 手续费支出减少（交易更少）
- 趋势行情中表现不变（regime=trend_long/trend_short 不受影响）
