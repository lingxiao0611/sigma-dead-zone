import sys
sys.path.insert(0, '/mnt/datadisk/underwater/bpinn_uq/code')
from eval_bem_calib import build_model, psnr_det, val_pairs_euvp

m = build_model()
print('det PSNR(first1000) =', round(psnr_det(m, val_pairs_euvp(), n=1000), 3), flush=True)
print('det PSNR(last500)   =', round(psnr_det(m, val_pairs_euvp()[500:], n=500), 3), flush=True)
print('DET_CHECK_DONE', flush=True)
