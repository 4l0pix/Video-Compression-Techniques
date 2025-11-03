import pandas as pd
import matplotlib.pyplot as plt
from utils import MAIN_MODELS, ABLATION_MODELS

df_main = pd.read_csv("../results/image_rd_kodak.csv")

for metric in ["psnr", "ms_ssim"]:
    plt.figure()
    for model in df_main["model"].unique():
        sub = df_main[df_main["model"] == model]
        grouped = sub.groupby("bpp")[metric].mean()
        plt.plot(grouped.index, grouped.values, label=model)
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel(metric.upper())
    plt.grid()
    plt.legend()
    plt.title(f"Rate–Distortion curve ({metric})")
    plt.savefig(f"../plots/rd_{metric}.png")


#load new data
df_ablation = pd.read_csv("../results/image_rd_ablation_resize_192x192.csv")

plt.figure(figsize=(8,6))

#only iterate ablation models
for model in ABLATION_MODELS.keys():
    sub_main = df_main[df_main["model"] == model]
    sub_ab = df_ablation[df_ablation["model"] == model]

    # Skip if model is missing from either dataset
    if sub_main.empty or sub_ab.empty:
        print(f"[INFO] Skipping {model} — no data found in one of the files.")
        continue

    plt.plot(sub_main["bpp"], sub_main["psnr"], 'o-', label=f"{model} (baseline)")
    plt.plot(sub_ab["bpp"], sub_ab["psnr"], 'x--', label=f"{model} (ablation)")

plt.xlabel("Bits per pixel (bpp)")
plt.ylabel("PSNR (dB)")
plt.title("Ablation Study: Image Size Effect")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../plots/rd_ablation_psnr.png", dpi=300)
plt.close()


