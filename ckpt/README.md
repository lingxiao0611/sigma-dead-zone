# ckpt/ — data split and checkpoints

Everything in this directory is gitignored; it is small enough to attach to a GitHub release or a
Zenodo record instead.

| Path | Why it matters |
|---|---|
| `funie_baseline/split.json` | **Required.** Fixes the 500-image calibration / 500-image test partition of EUVP that every number in the paper depends on. Without it, results will not reproduce. |
| `funie_baseline/best.pth` | Deterministic FUnIE-GAN baseline. |
| `funie_edl/best.pth` | FUnIE-GAN with the evidential head. |
| `funie_ensemble/seed_{0..4}/best.pth` | The five ensemble members. |
| `ushape_*/...` | U-shape Transformer arms (deterministic, EDL, five-member ensemble). |

Total size is modest — FUnIE-GAN is 7.0M parameters per model and the five-member ensemble is 35.1M
in total — so a GitHub release is a reasonable home for the FUnIE checkpoints. The U-shape arms at
31.6M parameters each are better placed on Zenodo. Note that the ensemble therefore outweighs a
single U-shape Transformer in parameters; the ensemble's real cost is inference, five forward passes
against one.

See `../docs/TO_FETCH_FROM_SERVER.md` for the exact source paths on the training host.
