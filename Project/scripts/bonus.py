import os
import time
import torch
import psutil
import subprocess
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean
from utils import compute_metrics
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION WITH GPU-ONLY MODELS
# ============================================================

# Debug GPU information
print("=" * 60)
print("GPU STATUS CHECK")
print("=" * 60)
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
print("=" * 60)

# Device configuration
if torch.cuda.is_available():
    DEVICE_CODEC = "cuda:0"
    DEVICE_YOLO = "cuda:0"
    # Enable benchmark mode for faster convolutions
    torch.backends.cudnn.benchmark = True
else:
    DEVICE_CODEC = "cpu"
    DEVICE_YOLO = "cpu"

print(f"[INFO] Using device for codec: {DEVICE_CODEC}")
print(f"[INFO] Using device for YOLO: {DEVICE_YOLO}")

BONUS_1 = True   # Accuracy-vs-Rate
BONUS_2 = True   # Complexity & Resources
BONUS_3 = True   # VMAF correlation

VIDEO_SEQUENCES = [
    {"name": "Bosphorus", "path": "../data/uvg/Bosphorus.yuv", "width": 1920, "height": 1080, "frames": 100},
    {"name": "HoneyBee",  "path": "../data/uvg/HoneyBee.yuv",  "width": 1920, "height": 1080, "frames": 100},
]

QUALITIES = [3, 4]
MODELS = {
    "bmshj2018_factorized": bmshj2018_factorized,
    "bmshj2018_hyperprior": bmshj2018_hyperprior,
    "mbt2018_mean": mbt2018_mean,
}

RESULTS_CSV = "../results/bonus_all_results.csv"
os.makedirs("../results", exist_ok=True)
os.makedirs("../plots", exist_ok=True)
TEMP_DIR = "../temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load YOLO with explicit device handling
print("Loading YOLO detector...")
try:
    detector = YOLO("yolov8n.pt").to(DEVICE_YOLO)
    print(f"YOLO loaded successfully on: {next(detector.model.parameters()).device}")
except Exception as e:
    print(f"YOLO loading failed: {e}")
    detector = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pad_to_multiple(tensor, multiple=64):
    """Pad tensor to be multiple of given value."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        # Use reflection padding which works better for images
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        tensor = torch.nn.functional.pad(tensor, padding, mode='reflect')

    return tensor, (h, w, pad_h, pad_w)

def unpad_tensor(tensor, original_shape):
    """Remove padding from tensor."""
    h, w, pad_h, pad_w = original_shape
    return tensor[:, :, :h, :w]

def read_yuv_frame(file, width, height, frame_idx):
    """Reads a single YUV420 frame from file."""
    frame_size = width * height * 3 // 2
    file.seek(frame_idx * frame_size, 0)
    y = np.frombuffer(file.read(width * height), dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(file.read(width * height // 4), dtype=np.uint8).reshape((height // 2, width // 2))
    v = np.frombuffer(file.read(width * height // 4), dtype=np.uint8).reshape((height // 2, width // 2))

    u = cv2.resize(u, (width, height), interpolation=cv2.INTER_NEAREST)
    v = cv2.resize(v, (width, height), interpolation=cv2.INTER_NEAREST)

    yuv = cv2.merge([y, u, v])
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)
    return rgb

def compute_vmaf(ref_path, dist_path):
    """Computes VMAF using ffmpeg/libvmaf if available."""
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except:
        return np.nan

    cmd = [
        "ffmpeg", "-y",
        "-i", dist_path, "-i", ref_path,
        "-lavfi", "libvmaf=model_path=vmaf_v0.6.1.json:log_fmt=json",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stderr.splitlines():
            if "VMAF score" in line:
                return float(line.split(":")[-1])
    except Exception as e:
        return np.nan
    return np.nan

def save_checkpoint(df):
    """Saves intermediate CSV every few frames."""
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Checkpoint saved: {len(df)} records")

def print_gpu_memory():
    """Prints current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def preload_models():
    """Preload all models to avoid repeated loading overhead."""
    print("Preloading GPU-compatible compression models...")
    loaded_models = {}

    for model_name, model_fn in MODELS.items():
        loaded_models[model_name] = {}
        for q in QUALITIES:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                net = model_fn(quality=q, pretrained=True).eval().to(DEVICE_CODEC)
                loaded_models[model_name][q] = net
                print(f"  Loaded {model_name} q={q} on {next(net.parameters()).device}")

            except Exception as e:
                print(f"  Failed to load {model_name} q={q}: {e}")
                loaded_models[model_name][q] = None

    return loaded_models

# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

def main():
    records = []

    # Preload all models once
    loaded_models = preload_models()

    for seq in VIDEO_SEQUENCES:
        seq_name = seq["name"]
        path = seq["path"]
        width, height, nframes = seq["width"], seq["height"], seq["frames"]

        print(f"\n{'='*50}")
        print(f"Processing sequence: {seq_name} ({width}x{height}, {nframes} frames)")
        print(f"{'='*50}")

        with open(path, "rb") as f:
            for frame_idx in tqdm(range(nframes), desc=f"{seq_name} frames"):
                try:
                    frame = read_yuv_frame(f, width, height, frame_idx)
                except Exception as e:
                    print(f"WARNING: Could not read frame {frame_idx}: {e}")
                    break

                # Create tensor and ensure it's on GPU
                img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img = img.to(DEVICE_CODEC, non_blocking=True)

                if frame_idx == 0:
                    print(f"First frame tensor on: {img.device}")
                    print(f"Original frame shape: {img.shape}")

                # Save reference image for VMAF
                ref_path = os.path.join(TEMP_DIR, f"ref_{seq_name}_{frame_idx}.png")
                Image.fromarray(frame).save(ref_path)

                for model_name in MODELS.keys():
                    for q in QUALITIES:
                        net = loaded_models[model_name][q]
                        if net is None:
                            continue

                        try:
                            with torch.no_grad():
                                torch.cuda.synchronize() if torch.cuda.is_available() else None

                                # Pad image to compatible dimensions
                                img_padded, padding_info = pad_to_multiple(img, multiple=64)
                                if frame_idx == 0 and q == 1:
                                    print(f"Padded frame shape: {img_padded.shape}")

                                # Encode (fully on GPU)
                                t0 = time.time()
                                out_enc = net.compress(img_padded)
                                torch.cuda.synchronize() if torch.cuda.is_available() else None
                                enc_time = time.time() - t0

                                # Decode (fully on GPU)
                                t1 = time.time()
                                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                                torch.cuda.synchronize() if torch.cuda.is_available() else None
                                dec_time = time.time() - t1

                                # Remove padding to get original dimensions
                                x_hat = unpad_tensor(out_dec["x_hat"], padding_info)

                                # Create modified out_dec for metrics computation
                                out_dec_original = {"x_hat": x_hat}

                                # Compute metrics
                                bpp, psnr, ms_ssim = compute_metrics(img, out_enc, out_dec_original)

                                # Save reconstruction
                                dist_path = os.path.join(TEMP_DIR, f"recon_{seq_name}_{frame_idx}_{model_name}_q{q}.png")
                                recon_img = (
                                    x_hat
                                    .clamp(0, 1)
                                    .squeeze()
                                    .permute(1, 2, 0)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                Image.fromarray((recon_img * 255).astype(np.uint8)).save(dist_path)

                                # BONUS-1: YOLO object detection
                                detections = np.nan
                                if BONUS_1 and detector is not None:
                                    try:
                                        results = detector(dist_path, verbose=False)
                                        detections = len(results[0].boxes)
                                    except Exception as e:
                                        print(f"YOLO detection failed: {e}")

                                # BONUS-2: resource usage
                                ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
                                vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0

                                # BONUS-3: compute VMAF (sampled to save time)
                                vmaf_score = np.nan
                                if BONUS_3 and frame_idx % 10 == 0:
                                    vmaf_score = compute_vmaf(ref_path, dist_path)

                                # Record results
                                records.append([
                                    seq_name, frame_idx, model_name, q,
                                    bpp, psnr, ms_ssim,
                                    detections, enc_time, dec_time,
                                    ram_mb, vram_mb, vmaf_score
                                ])

                                # Clean up temporary files
                                if os.path.exists(dist_path):
                                    os.remove(dist_path)

                        except RuntimeError as e:
                            print(f"ERROR: Frame {frame_idx} {model_name} q={q}: {e}")
                            if "CUDA out of memory" in str(e):
                                torch.cuda.empty_cache()
                            continue
                        except Exception as e:
                            print(f"ERROR: Unexpected error: {e}")
                            continue

                # Clean up reference image
                if os.path.exists(ref_path):
                    os.remove(ref_path)

                # Save checkpoint and print progress
                if frame_idx % 10 == 0 and records:
                    df_temp = pd.DataFrame(records, columns=[
                        "seq", "frame", "method", "level", "bitrate", "psnr", "ms_ssim",
                        "detections", "enc_time", "dec_time", "cpu_ram_mb", "gpu_vram_mb", "vmaf"
                    ])
                    save_checkpoint(df_temp)
                    print_gpu_memory()

    # ============================================================
    # SAVE FINAL RESULTS
    # ============================================================

    if records:
        df = pd.DataFrame(records, columns=[
            "seq", "frame", "method", "level", "bitrate", "psnr", "ms_ssim",
            "detections", "enc_time", "dec_time", "cpu_ram_mb", "gpu_vram_mb", "vmaf"
        ])
        df.to_csv(RESULTS_CSV, index=False)
        print(f"All results saved to {RESULTS_CSV}")
        print(f"Total records: {len(records)}")

        # Generate plots
        generate_plots(df)
    else:
        print("No records were generated!")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def generate_plots(df):
    """Generate all bonus plots."""
    print("Generating plots...")

    # BONUS-1: Accuracy vs Rate
    if BONUS_1 and "detections" in df.columns:
        plt.figure(figsize=(10, 6))
        for model in df["method"].unique():
            model_data = df[df["method"] == model]
            if model_data["detections"].notna().any():
                grouped = model_data.groupby("bitrate")["detections"].mean()
                plt.plot(grouped.index, grouped.values, 'o-', linewidth=2, markersize=6, label=model)
        plt.xlabel("Bitrate (bits per pixel)", fontsize=12)
        plt.ylabel("Average Detections", fontsize=12)
        plt.title("BONUS-1: Object Detection Accuracy vs Bitrate", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("../plots/bonus1_accuracy_vs_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Bonus 1 plot saved")

    # BONUS-2: Encoding time vs Rate
    if BONUS_2:
        plt.figure(figsize=(10, 6))
        for model in df["method"].unique():
            grouped = df[df["method"] == model].groupby("bitrate")["enc_time"].mean()
            plt.plot(grouped.index, grouped.values, 's-', linewidth=2, markersize=6, label=model)
        plt.xlabel("Bitrate (bits per pixel)", fontsize=12)
        plt.ylabel("Encoding Time (seconds)", fontsize=12)
        plt.title("BONUS-2: Encoding Time vs Bitrate", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("../plots/bonus2_encoding_time_vs_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Bonus 2 plot saved")

    # BONUS-3: VMAF vs PSNR correlation
    if BONUS_3 and "vmaf" in df.columns:
        df_vmaf = df[df["vmaf"].notnull() & df["vmaf"].between(0, 100)]
        if not df_vmaf.empty:
            plt.figure(figsize=(10, 6))
            for model in df_vmaf["method"].unique():
                sub = df_vmaf[df_vmaf["method"] == model]
                plt.scatter(sub["psnr"], sub["vmaf"], label=model, alpha=0.6, s=50)
            plt.xlabel("PSNR (dB)", fontsize=12)
            plt.ylabel("VMAF", fontsize=12)
            plt.title("BONUS-3: Correlation between PSNR and VMAF", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("../plots/bonus3_vmaf_vs_psnr.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Bonus 3 plot saved")
        else:
            print("No VMAF data available for plotting")

    print("All plots generated successfully")

# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    start_time = time.time()
    print("Starting enhanced compression analysis with GPU-only models...")

    try:
        main()
    except KeyboardInterrupt:
        print("Experiment interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
