# -*- coding: utf-8 -*-
"""把 TorchUncertainty 的 risk_coverage.py 单文件移植到 code/uq/。

手法与 NIG 头移植一致（方案 A）：
  1) 加 `from __future__ import annotations`（源文件用 X | None 的 3.10 语法，py3.9 需兜）
  2) 该文件系统无 torch_uncertainty 内部依赖（只依赖 torchmetrics/numpy/matplotlib）→ 可直接搬
  3) 追加"回归适配层"：复用 TU 原生 _aurc_rejection_rate_compute 核心，
     只把输入从"分类 probs + 0/1 误差"换成"置信度（越大越可信）+ 连续误差 |y-yhat|"。
"""
import io

SRC = ("/mnt/data/underwater/bpinn_uq/code/torch-uncertainty/src/"
       "torch_uncertainty/metrics/classification/risk_coverage.py")
DST = "/mnt/data/underwater/bpinn_uq/code/uq/tu_risk_coverage.py"

src = io.open(SRC, encoding="utf-8").read()
if not src.lstrip().startswith("from __future__"):
    src = "from __future__ import annotations\n" + src

ADAPTER = '''
# ===========================================================================
# 回归适配层（逐像素 UQ 拒识评测用）—— 追加，非 TU 原生
#
# TU 原版 AURC/RiskAtxCov/CovAtxRisk.update() 面向分类：输入 probs(N,C) + 标签，
# 内部取 max-prob 当分数、0/1 分类误差。逐像素回归里"误差"是连续的 |y - yhat|，
# 无法走 update()。
#
# 但核心计算 `_aurc_rejection_rate_compute(scores, errors)` 本身与任务无关：
#   按 scores 降序排（= 置信度从高到低），对 errors 做累积均值 → 得到随覆盖率递增的 risk 曲线。
# 因此适配层只做"换输入"，排序+累积均值的算法完全复用 TU 实现，未重写任何算法。
#   conf   : (N,) 置信度，越大越可信（回归里传 -sigma）
#   errors : (N,) 连续误差 |y - yhat|（需落在 [0,1]，我们的 [0,1] 像素域满足）
# 注意：单调变换（如标量重校准 ratio）不改变排序 → 风险-覆盖率曲线对 ratio 不变。
# ===========================================================================
import math as _math


def regression_risk_curve(conf, errors):
    """返回按覆盖率从低到高排列的 risk 曲线（长度 N）。float64 防累积误差。"""
    return _aurc_rejection_rate_compute(conf.double().flatten(),
                                        errors.double().flatten())


def regression_aurc(conf, errors):
    """Area Under the Risk-Coverage curve（越低越好）。"""
    curve = regression_risk_curve(conf, errors)
    n = curve.size(0)
    if n < 2:
        return float("nan")
    cov = torch.arange(1, n + 1, device=curve.device, dtype=curve.dtype) / n
    return float(_auc_compute(cov, curve) / (1 - 1 / n))


def regression_augrc(conf, errors):
    """Area Under the Generalized Risk-Coverage curve（越低越好）。"""
    curve = regression_risk_curve(conf, errors)
    n = curve.size(0)
    if n < 2:
        return float("nan")
    cov = torch.arange(1, n + 1, device=curve.device, dtype=curve.dtype) / n
    return float(_auc_compute(cov, curve * cov) / (1 - 1 / n))


def regression_risk_at_cov(conf, errors, cov_threshold):
    """给定覆盖率下的风险（如 0.8 = 保留 80% 最可信像素后的误差）。"""
    curve = regression_risk_curve(conf, errors)
    n = curve.size(0)
    _risk_coverage_checks(cov_threshold)
    return float(curve[_math.ceil(n * cov_threshold) - 1])


def regression_cov_at_risk(conf, errors, risk_threshold):
    """给定风险上限下能达到的最大覆盖率。"""
    curve = regression_risk_curve(conf, errors)
    n = curve.size(0)
    admissible = torch.nonzero(curve <= risk_threshold, as_tuple=False).flatten()
    if admissible.numel() == 0:
        return float("nan")
    return float((admissible[-1] + 1) / n)
'''

io.open(DST, "w", encoding="utf-8").write(src + "\n" + ADAPTER)
print("ported ->", DST)
print("TU 原文行数:", len(src.splitlines()), "| 追加适配层行数:", len(ADAPTER.splitlines()))
