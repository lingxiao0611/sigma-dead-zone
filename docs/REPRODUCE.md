# Reproducing the audit

## 1. Environment

Python 3.9 with CUDA 12.1. The original environment was a conda env named `bpinn_uq` holding
`torch 2.5.1+cu121` and `pytorch-lightning 2.4.0`; the Mamba-UIE arm additionally needs
`causal-conv1d` and `mamba-ssm` built against the same CUDA toolkit.

```bash
conda create -n sigma_dead_zone python=3.9 -y
conda activate sigma_dead_zone
pip install -r requirements.txt
```

TorchUncertainty itself is **not** installed: the whole package requires Python >= 3.10, so the
three files we need are ported instead (see `docs/TO_FETCH_FROM_SERVER.md`). This is why the audit
can run on 3.9 at all.

## 2. Datasets

All four datasets are public. Download and lay them out as:

```
datasets/
  EUVP_Dataset/            # 10,435 paired training images
  UIEB_Dataset/UIEB Dataset/{raw,reference}/   # 890 pairs + 60 challenge
  underwater-test-dataset-U45-/upload/U45/{U45,blue,green,haze}
  RUIE_Dataset/RUIE/{UIQS/{A..E},UCCS/{blue,green,blue-green}}
```

| Dataset | Role | Reference images |
|---|---|---|
| EUVP | in-domain water body; 500 calibration + 500 test | yes |
| UIEB | the transfer target — the only cross-water set with full references | yes |
| U45 | reference-free severity check (blue/green/haze, 15 each) | no |
| RUIE-UIQS | reference-free severity check, grades A (best) to E (worst) | no |

All images are resized to 256x256. Note that RUIE-UIQS runs **A = best, E = worst**.

## 3. Upstream models

Clone the repositories listed in `docs/UPSTREAM_REPOS.md` into `code/` and apply the changes noted
there. Then copy the private modules described in `docs/TO_FETCH_FROM_SERVER.md`.

## 4. Run order

```
train_ushape_baseline.py     -> deterministic baseline on the U-shape backbone
train_ushape_edl.py          -> evidential arm
train_ushape_ens4090.py      -> five-member ensemble arm
train_funie_gauss.py         -> Gaussian-head control (Sec. VI-B)
train_mamba_fp32.py          -> Mamba-UIE reference (fp32 only; bf16 corrupts training)

eval_*                       -> one script per experiment, see the mapping table in README.md
make_figures.py              -> regenerates every figure
```

Training used a 30-epoch budget with no learning-rate schedule, no weight decay, Adam at 1e-4 with
betas (0.5, 0.999), batch size 8, fp32 throughout, inputs normalized to [-1, 1], and a single
random horizontal flip applied identically to input and reference. The one-to-many baseline alone
reverts to its authors' batch size of 1.

## 5. Expected outputs

Each evaluation script writes JSON into `outputs/`; `records/` mirrors the local `outputs/` tree and
holds every JSON the paper cites. Byte-identical reproduction is not guaranteed across GPU models and
library versions, but every ordering and every reported verdict in the paper is stable across runs;
the split is fixed by `ckpt/funie_baseline/split.json` and all other sampling uses fixed seeds.

Two conventions are worth stating because they are easy to get wrong.

`records/ushape_ens_audit.json` is the protocol-corrected run for the U-shape ensemble arm. An
earlier run of the same script evaluated the full EUVP validation block (1000 images, calibration
fold included) and the full 890-image UIEB set, which violates the 500-calibration/500-test and
445-image UIEB protocols the rest of the paper uses; it is not shipped. The shipped record is the
re-evaluation on the correct splits, and it is the source of Table VII's U-shape+ENS column.

Risk-coverage AUCs in `records/` follow the TorchUncertainty convention, in which the curve is built
by retaining the pixels assigned the highest confidence and confidence is `-sigma`. A hand-written
`argsort(-sigma)` sweep builds the opposite curve and cannot be compared against MAE; every AURC we
report uses the TorchUncertainty direction, so a lower AURC than the `aurc_random` field in the same
JSON means the ranking carries information.
