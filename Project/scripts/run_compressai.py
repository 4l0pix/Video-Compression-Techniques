import os, time, torch
import pandas as pd
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018,  cheng2020_anchor, cheng2020_attn)
from utils import load_image, compute_metrics
from utils import MAIN_MODELS as MODELS

# Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


QUALITIES = [1, 2, 3, 4, 5, 6]


data = []
image_dir = "../data/images"
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue  # αγνόησε οτιδήποτε άλλο
    img_path = os.path.join(image_dir, img_name)
    img = load_image(img_path, device=device)

for model_name, model_fn in MODELS.items():
        for q in QUALITIES:
            try:
                net = model_fn(quality=q, pretrained=True).eval().to(device)
                print(img.device, next(net.parameters()).device)
            except Exception as e:
                print(f"[WARN] Could not load model {model_name} (q={q}): {e}")
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
            data.append([img_name, model_name, q, bpp, psnr, ms_ssim, enc_time, dec_time])

pd.DataFrame(data, columns=["img","model","level","bpp","psnr","ms_ssim","enc_time","dec_time"]) .to_csv("../results/image_rd_kodak.csv", index=False)
