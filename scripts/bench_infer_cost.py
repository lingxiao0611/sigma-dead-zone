# -*- coding: ascii -*-
"""Inference-cost measurement for the deployment table.

Measures, per audited arm: parameter count (from the actual checkpoint),
forward passes per image, single-image latency, batch-8 throughput, and peak
allocated GPU memory. No new algorithm is written here: the model assemblies are
copied verbatim from code/eval_rejection.py and code/eval_ushape_rejection.py,
and every arm is loaded from the checkpoint the paper already audits.

Read-only w.r.t. the project: writes one JSON summary under outputs/.
"""
import json
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = "/mnt/data/underwater/bpinn_uq"
sys.path.insert(0, ROOT + "/code/funie_gan/PyTorch")
sys.path.insert(0, ROOT + "/code/u_shape_transformer")
sys.path.insert(0, ROOT + "/code")

from nets.funiegan import GeneratorFunieGAN          # noqa: E402
from net.Ushape_Trans import Generator               # noqa: E402
from uq.nig_layer import NormalInverseGammaConvNd    # noqa: E402

SEEDS = [101, 202, 303, 404, 505]
OUT = ROOT + "/outputs/infer_cost.json"
DEV = "cuda"


class FunieNIG(nn.Module):
    """Copied verbatim from eval_rejection.py."""

    def __init__(self):
        super().__init__()
        g = GeneratorFunieGAN()
        self.down1, self.down2, self.down3 = g.down1, g.down2, g.down3
        self.down4, self.down5 = g.down4, g.down5
        self.up1, self.up2, self.up3, self.up4 = g.up1, g.up2, g.up3, g.up4
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)),
            NormalInverseGammaConvNd(nn.Conv2d, event_dim=3, in_channels=64,
                                     kernel_size=4, padding=1))

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        return self.final(self.up4(u3, d1))


class UshapeDet(nn.Module):
    """Copied verbatim from bench_ushape.py."""

    def __init__(self):
        super().__init__()
        self.g = Generator()

    def forward(self, x):
        return self.g(x)[-1]


class UshapeNIG(nn.Module):
    """Copied verbatim from eval_ushape_rejection.py."""

    def __init__(self):
        super().__init__()
        g = Generator()
        self.g = g
        self.g.feature_to_rgb[0] = NormalInverseGammaConvNd(
            nn.Conv2d, event_dim=3, in_channels=32, kernel_size=1)

    def forward(self, x):
        return self.g(x)[-1]


def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def n_params_m(state):
    return round(sum(v.numel() for v in state.values()) / 1e6, 4)


def bench(fn, bs, iters=20, warm=5):
    """Return (latency_ms_per_batch, images_per_s, peak_mem_MB)."""
    x = torch.randn(bs, 3, 256, 256, device=DEV)
    with torch.no_grad():
        for _ in range(warm):
            fn(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    mem = torch.cuda.max_memory_allocated() / 1024 ** 2
    return dt * 1000.0, bs / dt, mem


def measure(name, fn, n_pass, params, bs=1, iters=20):
    lat, ips, mem = bench(fn, bs=bs, iters=iters)
    rec = {"arm": name, "passes_per_image": n_pass, "params_M": params,
           "bs": bs, "latency_ms": round(lat, 3),
           "ms_per_image": round(lat / bs, 3), "img_per_s": round(ips, 1),
           "peak_mem_MB": round(mem, 1)}
    print("%-34s pass=%2d  params=%8.4fM  bs=%d  %8.3f ms  %7.3f ms/img  %8.1f img/s  %7.1f MB"
          % (name, n_pass, params, bs, lat, lat / bs, ips, mem), flush=True)
    return rec


def main():
    out = {"gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "resolution": "256x256", "dtype": "float32",
           "arms": []}

    print("=== inference cost, %s, %s ===" % (out["gpu"], out["torch"]), flush=True)

    # ---- FUnIE deterministic ----
    sd = load_sd(ROOT + "/ckpt/funie_baseline/best.pth")
    m = GeneratorFunieGAN().to(DEV)
    m.load_state_dict(sd)
    m.eval()
    out["arms"].append(measure("FUnIE det", lambda x: m(x), 1, n_params_m(sd), 1))
    out["arms"].append(measure("FUnIE det", lambda x: m(x), 1, n_params_m(sd), 8))

    # ---- FUnIE + MC-BatchNorm surrogate: same net, train-mode norm, T passes ----
    m.train()
    out["arms"].append(measure("FUnIE MC-BN (T=30)", lambda x: [m(x) for _ in range(30)],
                               30, n_params_m(sd), 1))
    m.eval()

    # ---- FUnIE + EDL ----
    sd_e = load_sd(ROOT + "/ckpt/funie_edl/best.pth")
    me = FunieNIG().to(DEV)
    me.load_state_dict(sd_e)
    me.eval()
    out["arms"].append(measure("FUnIE + EDL", lambda x: me(x)["loc"], 1, n_params_m(sd_e), 1))

    # ---- FUnIE + deep ensemble (5 members) ----
    ms, tot = [], 0.0
    for s in SEEDS:
        sd_k = load_sd(ROOT + "/ckpt/funie_ensemble/seed_%d/best.pth" % s)
        mk = GeneratorFunieGAN().to(DEV)
        mk.load_state_dict(sd_k)
        mk.eval()
        ms.append(mk)
        tot += n_params_m(sd_k)

    def ens5(x):
        o = None
        for mk in ms:
            o = mk(x) if o is None else o + mk(x)
        return o

    out["arms"].append(measure("FUnIE + ensemble (5)", ens5, 5, round(tot, 4), 1))

    # ---- U-shape deterministic ----
    sd_u = load_sd(ROOT + "/ckpt/ushape_baseline/best.pth")
    mu = UshapeDet().to(DEV)
    mu.load_state_dict(sd_u)
    mu.eval()
    out["arms"].append(measure("U-shape det", lambda x: mu(x), 1, n_params_m(sd_u), 1))

    # ---- U-shape + EDL ----
    sd_ue = load_sd(ROOT + "/ckpt/ushape_edl/best.pth")
    mue = UshapeNIG().to(DEV)
    mue.load_state_dict(sd_ue)
    mue.eval()
    out["arms"].append(measure("U-shape + EDL", lambda x: mue(x)["loc"], 1, n_params_m(sd_ue), 1))

    # ---- probe: other paradigms, class availability only ----
    probe = {}
    try:
        sys.path.insert(0, ROOT + "/code/puie_net")
        import PUIENet_MC as P1
        probe["puie_mc"] = [k for k in dir(P1) if k[:1].isupper()][:12]
    except Exception as e:
        probe["puie_mc"] = "ERR " + str(e)[:90]
    try:
        sys.path.insert(0, ROOT + "/code/puie_net")
        import PUIENet_MP as P2
        probe["puie_mp"] = [k for k in dir(P2) if k[:1].isupper()][:12]
    except Exception as e:
        probe["puie_mp"] = "ERR " + str(e)[:90]
    out["probe"] = probe
    print("probe: " + json.dumps(probe), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE " + OUT, flush=True)


if __name__ == "__main__":
    main()
