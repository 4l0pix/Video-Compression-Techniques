import torch
from PIL import Image
import numpy as np
import math
from pytorch_msssim import ms_ssim
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor, cheng2020_attn)

# Models used in main experiment
MAIN_MODELS = {
    "bmshj2018_factorized": bmshj2018_factorized,
    "bmshj2018_hyperprior": bmshj2018_hyperprior,
    "mbt2018": mbt2018,
    "mbt2018_mean": mbt2018_mean,
    "cheng2020_anchor": cheng2020_anchor,
    "cheng2020_attn": cheng2020_attn,
}

# Subset used for ablation (lighter, faster)
ABLATION_MODELS = {
    "bmshj2018_factorized": bmshj2018_factorized,
    "bmshj2018_hyperprior": bmshj2018_hyperprior,
}


#load image
def load_image(path, device="cuda", size=(256, 256)):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size)
    x = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)

#metrics computation
def compute_metrics(x, out_net, out_dec):
    num_pixels = x.size(-2) * x.size(-1)
    # bits per pixel
    bpp = sum(len(s[0]) for s in out_net["strings"]) * 8.0 / num_pixels
    # PSNR
    mse = torch.mean((x - out_dec["x_hat"]).pow(2))
    psnr = 10 * math.log10(1.0 / mse.item())
    # MS-SSIM
    msssim_val = ms_ssim(out_dec["x_hat"], x, data_range=1.0).item()
    return bpp, psnr, msssim_val

