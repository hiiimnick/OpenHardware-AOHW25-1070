#!/usr/bin/env python3
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import time
import json
import platform
import copy
from datetime import datetime

from evaluation import evaluate 

torch.set_float32_matmul_precision("medium")

is_rocm = "rocm" in torch.__version__.lower()
is_nvidia = torch.cuda.is_available() and not is_rocm
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# === ROCm Environment Setup ===
def setup_env_rocm():
    rocm_optimizations = {
        'HIP_PLATFORM': 'amd',
        'HIP_LAUNCH_BLOCKING': '0',
        'ROCM_FORCE_CUDA_COMPAT': '1',
        'HSA_OVERRIDE_GFX_VERSION': '11.0.3',
        'HIP_FORCE_DEV_KERNARG': '1',
        'HSA_FORCE_FINE_GRAIN_PCIE': '1',
        'HIP_HOST_COHERENT': '0',
        'HSA_LARGE_BAR': '1',
        'ROC_ENABLE_LARGE_BAR': '1',
        'HSA_ENABLE_INTERRUPT': '0'
    }

    for key, value in rocm_optimizations.items():
        os.environ[key] = value

    if is_rocm:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        return True
    return False

rocm_available = setup_env_rocm()

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (128, 128)
batch_size = 128
epochs = 20
learning_rate = 1e-4

if is_rocm:
    if torch.cuda.is_available():
        memory_fraction = 0.85
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.cuda.empty_cache()
    use_channels_last = True
    use_aggressive_fusion = False
    use_torch_compile = False
elif is_nvidia:
    use_channels_last = True
    use_aggressive_fusion = False
    use_torch_compile = True
else:
    use_channels_last = False
    use_aggressive_fusion = False
    use_torch_compile = False

print(f"Running on device: {device}")
print(f"ROCm detected: {is_rocm}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

DATASET_PATH = "/home/opt/openhw2025_gpu_adaptive/dataset" # Adjust to your dataset path

# === Dataset ===
class PatchDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = int(self.dataframe.iloc[idx]['stenoza_label'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def load_dataframe(folders, csvs):
    df_list = []
    for folder, csv in zip(folders, csvs):
        df = pd.read_csv(csv)
        df['filepath'] = df['patch_filename'].apply(lambda f: os.path.join(folder, f))
        df['stenoza_label'] = df['stenoza_label'].astype(int)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# === Dataset configurations === 

sets = { # Adjust paths as necessary
    "train": {"folders": [f"{DATASET_PATH}/train1/images", f"{DATASET_PATH}/train2/images"],
              "csvs": [f"{DATASET_PATH}/train1/labels.csv", f"{DATASET_PATH}/train2/labels.csv"]},
    "val": {"folders": [f"{DATASET_PATH}/val1/images", f"{DATASET_PATH}/val2/images"],
            "csvs": [f"{DATASET_PATH}/val1/labels.csv", f"{DATASET_PATH}/val2/labels.csv"]},
    "test": {"folders": [f"{DATASET_PATH}/test1/images", f"{DATASET_PATH}/test2/images"],
             "csvs": [f"{DATASET_PATH}/test1/labels.csv", f"{DATASET_PATH}/test2/labels.csv"]}
}

# === Transforms ===
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
val_test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

# === GPU Warmup ===
def gpu_warmup():
    if torch.cuda.is_available():
        print("Performing GPU warmup...")
        torch.cuda.empty_cache()
        for size in [32, 64, 128]:
            try:
                warmup_tensor = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)
                if use_channels_last:
                    warmup_tensor = warmup_tensor.to(memory_format=torch.channels_last)
                result = torch.sum(warmup_tensor)
                del warmup_tensor, result
                torch.cuda.synchronize()
                time.sleep(0.1)
            except Exception as e:
                print(f"Warmup with size {size} failed: {e}")
                break
        torch.cuda.empty_cache()
        print("GPU warmup completed successfully")

# === Main Training Loop ===
if __name__ == '__main__':
    gpu_warmup()

    train_df = load_dataframe(sets["train"]["folders"], sets["train"]["csvs"])
    val_df = load_dataframe(sets["val"]["folders"], sets["val"]["csvs"])
    test_df = load_dataframe(sets["test"]["folders"], sets["test"]["csvs"])

    # Label distribution
    labels_np = train_df['stenoza_label'].values
    bincounts = np.bincount(labels_np)
    print("Label distribution in train:", dict(enumerate(bincounts)))

    # Weighted sampler
    class_weights = 1. / torch.tensor(bincounts, dtype=torch.float)
    sample_weights = class_weights[labels_np]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoaders
    num_workers = min(6, os.cpu_count() - 1)
    train_loader = DataLoader(PatchDataset(train_df, train_transform), batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)
    val_loader = DataLoader(PatchDataset(val_df, val_test_transform), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)
    test_loader = DataLoader(PatchDataset(test_df, val_test_transform), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)

    # Model setup
    print("Creating optimized model...")
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    if use_torch_compile and not is_rocm:
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
        except Exception as e:
            print(f"torch.compile failed: {e}")
            use_torch_compile = False
    elif is_rocm:
        print("torch.compile causes issues on ROCm, disabling")
        use_torch_compile = False
    else:
        print("torch.compile disabled")

    # Loss and optimizer
    pos_weight = torch.tensor([bincounts[0] / bincounts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Mixed precision
    if device.type == 'cuda':
        amp_dtype = torch.float16
        if is_rocm:
            scaler = GradScaler(device_type, init_scale=2.**12, growth_factor=1.5,
                                backoff_factor=0.7, growth_interval=1000)
        else:
            scaler = GradScaler(device_type)
        use_amp = True
    else:
        scaler = None
        use_amp = False
        amp_dtype = torch.float32
    print(f"Mixed precision enabled: {use_amp} (dtype={amp_dtype})")

    # === Training Loop ===
    best_val_acc = 0.0
    training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        try:
            for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
                imgs = imgs.to(device, non_blocking=True)
                if use_channels_last:
                    imgs = imgs.to(memory_format=torch.channels_last)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                optimizer.zero_grad()

                if use_amp:
                    with autocast(device_type, dtype=amp_dtype):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                    correct += (preds == labels.int()).sum().item()
                    total += labels.size(0)

                if is_rocm and batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"Training error in epoch {epoch+1}: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            else:
                raise

        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            try:
                for batch_idx, (imgs, labels) in enumerate(val_loader):
                    imgs = imgs.to(device, non_blocking=True)
                    if use_channels_last:
                        imgs = imgs.to(memory_format=torch.channels_last)
                    labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
                    if use_amp:
                        with autocast(device_type, dtype=amp_dtype):
                            outputs = model(imgs)
                    else:
                        outputs = model(imgs)
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                    correct += (preds == labels.int()).sum().item()
                    total += labels.size(0)
            except RuntimeError as e:
                print(f"Validation error in epoch {epoch+1}: {e}")
                torch.cuda.empty_cache()

        val_acc = correct / total if total > 0 else 0
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f} sec")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

        if is_rocm and epoch % 5 == 0:
            torch.cuda.empty_cache()

    total_training_time = time.time() - training_start_time
    print(f"Training time: {int(total_training_time // 3600)}h {int((total_training_time % 3600) // 60)}m")

    model.load_state_dict(torch.load("best_model.pt"))
    sample_input = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    if use_channels_last:
        sample_input = sample_input.to(memory_format=torch.channels_last)

    def optimize_model_for_inference(model, sample_input, use_torchscript=True):
        print("Optimizing model for inference...")
        clean_model = copy.deepcopy(model)
        clean_model.eval()
        if use_channels_last:
            clean_model = clean_model.to(memory_format=torch.channels_last)

        if use_torchscript and not is_rocm:
            try:
                scripted_model = torch.jit.script(clean_model)
                quantized_model = torch.ao.quantization.quantize_dynamic(
                    scripted_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                return quantized_model, True
            except Exception as e:
                print(f"TorchScript optimization failed: {e}")
        return clean_model, False

    optimized_model, is_torchscript = optimize_model_for_inference(model, sample_input, use_torchscript=True)
    if is_torchscript:
        torch.jit.save(optimized_model, "best_model_optimized.pt")
    else:
        torch.save(optimized_model.state_dict(), "best_model_optimized.pth")

    print("Evaluating optimized model...")
    
    all_preds, all_labels, all_heatmaps, inference_time, saved_bad_count = evaluate(
        model=optimized_model,
        test_loader=test_loader,
        test_df=test_df,
        device=device,
        use_channels_last=use_channels_last,
        use_amp=use_amp,
        is_torchscript=is_torchscript,
        device_type=device_type,
    )

    if not (all_preds and all_labels):
        print("Evaluation incomplete due to errors")

    torch.cuda.empty_cache()
    print("Training and evaluation completed successfully!")
