# Upstream repositories

This audit evaluates released models and builds on released tooling. **None of the third-party
code is vendored into this repository** — clone the upstream projects yourself so their licenses
stay intact, and cite them alongside this work.

## Backbones and audited paradigms

| Component | Upstream | What we changed |
|---|---|---|
| FUnIE-GAN (main backbone, ~2M) | `yangyang0122/funiegan` (PyTorch port: `woozydani/funiegan`) | Removed the discriminator for the deterministic baseline; exposed the bottleneck feature map so an uncertainty head can be attached. |
| U-shape Transformer (31.6M) | `LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement` | Replaced the final `feature_to_rgb` layer with the evidential head for the EDL arm. |
| U-shape Transformer (public reimplementation) | `Lucare11/U-shape-Transformer` | Used for the U-shape ensemble arm. |
| Mamba-UIE | `zhangsong1213/Mamba-UIE` | Retrained in fp32; the released bf16 autocast path corrupts training (see Sec. VI-C of the paper). |
| BEM | `Aloz3r/BEM` | None. Audited as released, then excluded from the main comparison on protocol grounds. |
| PUIE-Net | `HRLTY/PUIE-Net` (ECCV 2022) | None. Audited as released; used through its CVAE sampler. |
| MC-BatchNorm | `1conan/switchable_norm` (SwitchNorm) | None. The normalization layer is used exactly as released. |

## Uncertainty tooling

| Component | Upstream | What we changed |
|---|---|---|
| Evidential regression head | TorchUncertainty (`torch-uncertainty/torch-uncertainty`), MIT — normal-inverse-gamma layer and DER loss, from Amini et al. 2020 | Ported three files to Python 3.9 (`uq/nig_layer.py`, `uq/normal_inverse_gamma.py`, `uq/der_loss.py`). The evidential *mechanism* is untouched; only the port is ours. |
| AURC / AUGRC | TorchUncertainty `risk_coverage.py` | `scripts/port_risk_coverage.py` produces `uq/tu_risk_coverage.py`, which reuses the native `_aurc_rejection_rate_compute` core and swaps the inputs from classification (probabilities, 0/1 errors) to regression (confidence, $|y-\hat y|$). The algorithm is not rewritten. |
| Recalibration baseline | `uncertainty-toolbox` (`uncertainty-toolbox/uncertainty-toolbox`), MIT | Called directly for the variance recalibrator compared in Sec. VI-D. |
| SSIM / UIQM | `mamba_uie`'s `utils/SSIM_loss.py`, `utils/UIQM_loss.py` | Called directly for the no-reference checks. |

## Note on licensing

Our scripts are MIT (`LICENSE`). Each upstream project keeps its own terms — check them before
redistributing a combined bundle. The upstream projects are cloned, not committed, precisely so
that this repository can stay MIT.
