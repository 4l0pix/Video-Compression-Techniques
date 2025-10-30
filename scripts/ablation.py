import os
import time
import torch
import pandas as pd
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior
from utils import load_image, compute_metrics
from utils import ABLATION_MODELS as MODELS
# ---------------------------------------------------
# Ablation: smaller input resolution (192x192 instead of 256x256)
# ---------------------------------------------------
ABLATION_DESC = "resize_192x192"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Ablation ({ABLATION_DESC}) on device: {device}")


QUALITIES = [1, 2, 3, 4, 5, 6]

data = []
image_dir = "../data/images"
results_csv = f"../results/image_rd_ablation_{ABLATION_DESC}.csv"
os.makedirs("../results", exist_ok=True)

# Main loop
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(image_dir, img_name)
    img = load_image(img_path, device=device, size=(192, 192))  # <- smaller resize

    for model_name, model_fn in MODELS.items():
        for q in QUALITIES:
            try:
                net = model_fn(quality=q, pretrained=True).eval().to(device)
            except Exception as e:
                print(f"[WARN] Skipping {model_name} q={q}: {e}")
                continue

            # Compress
            t0 = time.time()
            out_enc = net.compress(img)
            enc_time = time.time() - t0

            # Decompress
            t1 = time.time()
            out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
            dec_time = time.time() - t1

            # Metrics
            bpp, psnr, ms_ssim = compute_metrics(img, out_enc, out_dec)
            data.append([
                img_name, model_name, q, bpp, psnr, ms_ssim,
                enc_time, dec_time, ABLATION_DESC
            ])

# Save results
pd.DataFrame(
    data,
    columns=["img", "model", "level", "bpp", "psnr", "ms_ssim",
             "enc_time", "dec_time", "ablation"]
).to_csv(results_csv, index=False)

print(f"Ablation experiment results saved to: {results_csv}")
