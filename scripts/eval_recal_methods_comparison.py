# -*- coding: utf-8 -*-
"""C11: 修复方法无关性对照 —— 标量 ratio vs uncertainty-toolbox 方差重校准 vs isotonic 分位映射。

科学问题：EDL 死区是"排序"失效，任何只修数值（幅度）的方法都不该能修好它。
本脚本把三类幅度修复方法在同一 (σ, |e|) 数据上跑一遍，输出：
  - RMS-CE / 1σ 覆盖（数值轴修复效果）
  - 重校后分层覆盖极差（形状轴：EDL 应失衡依旧、ENS 应平坦依旧）
  - lift / AURC（排序轴：所有方法都不变——单调修复不改变排序）
预期结论：无论用什么幅度修复方法，EDL 跨域分层失衡照旧 → 死区是修复方法无关的。

输入：outputs/arrays_{UIEB,EUVP}.npz（C10 dump 脚本产出，含 err/sigma/contrast）
输出：outputs/toolbox_recal_comparison.json
运行位置：本地（纯 numpy/sklearn + 可选 uncertainty-toolbox，无 GPU 依赖）。
"""
import os, json
import numpy as np

try:
    import uncertainty_toolbox as uqt
    HAS_UQT = True
except Exception:
    HAS_UQT = False

from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "outputs")
NBIN = 4


def rms_ce(err, sig):
    return float(np.sqrt(np.mean((err / sig - 1) ** 2)))


def fit_ratio(err, sig):
    """我们的标量 ratio：最小化 RMS-CE 的解析解 r = sqrt(E[e²]/E[σ²])。"""
    return float(np.sqrt((err ** 2).mean() / (sig ** 2).mean()))


def toolbox_var_recal(err_cal, sig_cal, err_test, sig_test):
    """uncertainty-toolbox 方差重校准。注意：toolbox API 是 one-shot（拟合与评估
    不分离），无法在 cal 上拟合、test 上应用，故跑 oracle 变体（直接在评估集上
    拟合），json 里如实标注 'toolbox_oracle'，仅作方法无关性参照。"""
    if not HAS_UQT:
        return None
    from uncertainty_toolbox import recalibration as uqr
    try:
        # toolbox 的 fit/apply 分离接口：在 cal 上拟合 σ 缩放曲线，test 上应用
        recal_fn = uqr.get_std_recalibrator(
            y_mean=np.zeros_like(err_cal.reshape(-1)),
            y_std=sig_cal.reshape(-1),
            y_true=err_cal.reshape(-1))
        new_sig = recal_fn(sig_test.reshape(-1))
        return np.asarray(new_sig, dtype=np.float32).reshape(sig_test.shape)
    except Exception as e:
        print(f"[warn] toolbox get_std_recalibrator unavailable: {e}")
        return None


def isotonic_sig_map(err_cal, sig_cal, sig_test):
    """Kuleshov 式思路的简化实现：用 isotonic 回归学 E[|e| | σ] 的单调映射，
    令 σ' = m(σ)，使 σ' 在校准集上匹配条件误差幅度。sklearn 标准库调用。
    在 channel-pixel 展平上拟合（与 Stage4 估计对象一致）。"""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6)
    iso.fit(sig_cal.reshape(-1), err_cal.reshape(-1))
    return iso.predict(sig_test.reshape(-1)).reshape(sig_test.shape)


def stratified_range(err, sig, contrast):
    """按输入对比度四分位分层的 1σ 覆盖极差（S1 重→S4 轻）。"""
    order = np.argsort(contrast)
    n = len(contrast)
    edges = np.linspace(0, n, NBIN + 1, dtype=int)
    covs = []
    for b in range(NBIN):
        idx = order[edges[b]:edges[b + 1]]
        covs.append(float((err[idx] <= sig[idx]).mean()))
    return dict(covs=[round(c, 4) for c in covs], range=round(max(covs) - min(covs), 4))


def evaluate(err_te, sig_te_orig, sig_te_new, contrast_te):
    return dict(
        rms_ce=round(rms_ce(err_te, sig_te_new), 4),
        cov1=round(float((err_te <= sig_te_new).mean()), 4),
        strat=stratified_range(err_te, sig_te_new, contrast_te),
        lift=round(img_lift(err_te, sig_te_new, contrast_te), 4),
        aurc=round(img_aurc(err_te, sig_te_new), 5),
    )


def img_lift(err, sig, contrast):
    ime = err.reshape(err.shape[0], -1).mean(axis=1)
    ims = sig.reshape(sig.shape[0], -1).mean(axis=1)
    n = len(ime); k = max(1, int(0.2 * n))
    top = ime[np.argsort(-ims)[:k]].mean(); bot = ime[np.argsort(ims)[:k]].mean()
    return float(top / bot)


def img_aurc(err, sig):
    e = err.reshape(-1).astype(np.float32); s = sig.reshape(-1).astype(np.float32)
    order = np.argsort(-s)
    cum = np.cumsum(e[order]) / np.arange(1, len(e) + 1)
    return float(cum.mean())


def run_dataset(name):
    z = np.load(os.path.join(OUTDIR, f"arrays_{name}.npz"))
    err_e, sig_e = z["err_edl"].astype(np.float32), z["sig_edl"].astype(np.float32)
    err_n, sig_n = z["err_ens"].astype(np.float32), z["sig_ens"].astype(np.float32)
    contrast = z["contrast"]
    n = len(contrast); half = n // 2
    out = {"n": int(n), "has_uqt": HAS_UQT}
    for tag, err, sig in [("edl", err_e, sig_e), ("ens", err_n, sig_n)]:
        # 校准/测试半分（与主协议一致）
        c_err, t_err = err[:half], err[half:]
        c_sig, t_sig = sig[:half], sig[half:]
        c_con, t_con = contrast[:half], contrast[half:]
        res = {"raw": evaluate(t_err, t_sig, t_sig, t_con)}
        # 方法1：我们的标量 ratio
        r = fit_ratio(c_err, c_sig)
        res["scalar_ratio"] = dict(ratio=round(r, 4), **evaluate(t_err, t_sig, t_sig * r, t_con))
        # 方法2：isotonic 分位/幅度映射
        sig_iso = isotonic_sig_map(c_err, c_sig, t_sig)
        res["isotonic"] = evaluate(t_err, t_sig, sig_iso, t_con)
        # 方法3：uncertainty-toolbox 方差重校准（可用时）
        if HAS_UQT:
            sig_tb = toolbox_var_recal(c_err, c_sig, t_err, t_sig)
            if sig_tb is not None:
                res["toolbox_var"] = evaluate(t_err, t_sig, sig_tb, t_con)
        out[tag] = res
        print(f"[{name}/{tag}] " + " | ".join(
            f"{k}: rms={v['rms_ce']} cov1={v['cov1']} strat_range={v['strat']['range']} lift={v['lift']}"
            for k, v in res.items()), flush=True)
    return out


if __name__ == "__main__":
    result = {}
    for ds in ["UIEB", "EUVP"]:
        p = os.path.join(OUTDIR, f"arrays_{ds}.npz")
        if os.path.exists(p):
            result[ds] = run_dataset(ds)
        else:
            print(f"[skip] {p} not found")
    with open(os.path.join(OUTDIR, "toolbox_recal_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("C11_DONE")
