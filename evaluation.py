import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

def setup_chunking_directories():
    RESULTS_DIR = 'chunking_results'
    HEATMAP_BASE_DIR = os.path.join(RESULTS_DIR, 'heatmaps')
    HEATMAP_RAW_DIR = os.path.join(HEATMAP_BASE_DIR, 'bad_raw')
    HEATMAP_OVERLAY_DIR = os.path.join(HEATMAP_BASE_DIR, 'bad_overlays')
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HEATMAP_BASE_DIR, exist_ok=True)
    os.makedirs(HEATMAP_RAW_DIR, exist_ok=True)
    os.makedirs(HEATMAP_OVERLAY_DIR, exist_ok=True)
    
    return RESULTS_DIR, HEATMAP_RAW_DIR, HEATMAP_OVERLAY_DIR

# === Default configuration ===

BAD_IMAGE_CRITERION = os.getenv('BAD_IMAGE_CRITERION', 'either').lower()
BAD_CONF_THRESHOLD = float(os.getenv('BAD_CONF_THRESHOLD', '0.5'))
SAVE_HEATMAP_RAW = os.getenv('SAVE_HEATMAP_RAW', '1') == '1'
SAVE_HEATMAP_OVERLAY = os.getenv('SAVE_HEATMAP_OVERLAY', '1') == '1'
SAMPLE_BAD_RATIO = float(os.getenv('SAMPLE_BAD_RATIO', '0.75'))

CHUNK_SIZE = 64               # Size of each chunk
OVERLAP = 16                  # Overlap between chunks
STRIDE = CHUNK_SIZE - OVERLAP # Step size for moving the chunk window

# === Chunking Functions ===

def chunk_image(image_tensor, chunk_size=CHUNK_SIZE, stride=STRIDE):
    B, C, H, W = image_tensor.shape
    chunks = []
    positions = []
    
    for y in range(0, H - chunk_size + 1, stride):
        for x in range(0, W - chunk_size + 1, stride):
            chunk = image_tensor[:, :, y:y+chunk_size, x:x+chunk_size]
            chunks.append(chunk)
            positions.append((y, x))
    
    if chunks:
        return torch.cat(chunks, dim=0), positions
    return None, []

def reconstruct_heatmap(predictions, positions, original_shape, chunk_size=CHUNK_SIZE):
    H, W = original_shape
    heatmap = torch.zeros((H, W))
    count_map = torch.zeros((H, W))
    
    for pred, (y, x) in zip(predictions, positions):
        if isinstance(pred, torch.Tensor):
            pred_value = pred.item() if pred.numel() == 1 else pred.mean().item()
        else:
            pred_value = pred
            
        heatmap[y:y+chunk_size, x:x+chunk_size] += pred_value
        count_map[y:y+chunk_size, x:x+chunk_size] += 1
    
    heatmap = heatmap / torch.clamp(count_map, min=1) # Average overlapping regions
    return heatmap

def detect_chunked(imgs, labels, model, device, use_channels_last=False, use_amp=False, is_torchscript=False, device_type='cuda'):
    batch_predictions = []
    batch_heatmaps = []
    
    for i in range(imgs.shape[0]):
        single_img = imgs[i:i+1]
        
        chunks, positions = chunk_image(single_img)
        
        if chunks is None or len(positions) == 0:
            chunks = single_img.to(device, non_blocking=True)
            if use_channels_last:
                chunks = chunks.to(memory_format=torch.channels_last)
            
            if use_amp and not is_torchscript:
                with autocast(device_type):
                    outputs = model(chunks)
            else:
                outputs = model(chunks)
            
            pred = torch.sigmoid(outputs).item()
            batch_predictions.append(pred > 0.5)
            batch_heatmaps.append(torch.full((imgs.shape[2], imgs.shape[3]), pred))
        else:
            # Process chunks
            chunks = chunks.to(device, non_blocking=True)
            if use_channels_last:
                chunks = chunks.to(memory_format=torch.channels_last)
            
            if use_amp and not is_torchscript:
                with autocast(device_type):
                    outputs = model(chunks)
            else:
                outputs = model(chunks)
            
            chunk_probs = torch.sigmoid(outputs).cpu()
            
            if chunk_probs.dim() > 1:
                chunk_probs = chunk_probs.squeeze(-1)  # Remove last dimension if present
            
            heatmap = reconstruct_heatmap(chunk_probs, positions, 
                                       (imgs.shape[2], imgs.shape[3]))
            
            overall_pred = torch.mean(chunk_probs).item() > 0.5 # Overall prediction (average of chunks or max)
            batch_predictions.append(overall_pred)
            batch_heatmaps.append(heatmap)
    
    return batch_predictions, batch_heatmaps

# === Visualization Functions ===

def save_detailed_results(all_preds, all_labels, all_heatmaps, test_df):
    os.makedirs('chunking_results', exist_ok=True)
    
    print("Saving visualization results...")
    
    results_df = pd.DataFrame({
        'filename': test_df['patch_filename'].iloc[:len(all_preds)],
        'filepath': test_df['filepath'].iloc[:len(all_preds)],
        'true_label': all_labels,
        'predicted_label': all_preds,
        'prediction_confidence': [hm.mean().item() if isinstance(hm, torch.Tensor) else np.mean(hm) for hm in all_heatmaps]
    })
    results_df.to_csv('chunking_results/predictions.csv', index=False)
    
    create_sample_visualizations(results_df, all_heatmaps)
    
    print(f"\nResults saved to 'chunking_results/' directory")
    print(f"predictions.csv: Detailed predictions")
    print(f"sample_results.png: Visual results for sample images")

def create_sample_visualizations(results_df, all_heatmaps, num_samples=12):
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    axes = axes.flatten()

    bad_indices = []
    good_indices = []
    total = min(len(results_df), len(all_heatmaps))
    for i in range(total):
        try:
            pred_bool = bool(results_df.iloc[i]['predicted_label'])
            conf = float(results_df.iloc[i]['prediction_confidence'])
            if _is_bad_image(pred_bool, conf):
                bad_indices.append(i)
            else:
                good_indices.append(i)
        except Exception:
            good_indices.append(i) # Consider it as good for errors

    n_bad_target = int(round(num_samples * max(0.0, min(1.0, SAMPLE_BAD_RATIO))))
    n_bad = min(n_bad_target, len(bad_indices))
    n_good = num_samples - n_bad
    if n_good > len(good_indices):
        extra = min(len(bad_indices) - n_bad, n_good - len(good_indices))
        n_bad += max(0, extra)
        n_good = min(len(good_indices), n_good)

    def pick_even(indices, k):
        if k <= 0 or len(indices) == 0:
            return []
        if k >= len(indices):
            return indices
        return list(sorted({int(x) for x in np.linspace(0, len(indices) - 1, k)}))

    selected_bad = [bad_indices[j] for j in pick_even(bad_indices, n_bad)]
    selected_good = [good_indices[j] for j in pick_even(good_indices, n_good)]
    sample_indices = selected_bad + selected_good

    # If there's not enough samples, pad with random indices
    if len(sample_indices) < num_samples:
        remaining_pool = [idx for idx in range(total) if idx not in set(sample_indices)]
        pad = pick_even(remaining_pool, num_samples - len(sample_indices))
        sample_indices.extend([remaining_pool[j] for j in pad])

    # Truncate if somehow too many
    sample_indices = sample_indices[:num_samples]
    
    for i, idx in enumerate(sample_indices):
        try:
            img_path = results_df.iloc[idx]['filepath']
            if os.path.exists(img_path):
                original_img = Image.open(img_path).convert('RGB')
                original_img = np.array(original_img.resize((128, 128)))
            else:
                original_img = np.zeros((128, 128, 3), dtype=np.uint8)
            
            if isinstance(all_heatmaps[idx], torch.Tensor):
                heatmap = all_heatmaps[idx].numpy()
            else:
                heatmap = np.array(all_heatmaps[idx])
            
            true_label = results_df.iloc[idx]['true_label']
            pred_label = results_df.iloc[idx]['predicted_label']
            confidence = results_df.iloc[idx]['prediction_confidence']
            
            axes[i*2].imshow(original_img)
            axes[i*2].set_title(f'Original {idx+1}\nTrue: {true_label}, Pred: {pred_label}', 
                               fontsize=10)
            axes[i*2].axis('off')
            
            axes[i*2+1].imshow(original_img, alpha=0.6)
            im = axes[i*2+1].imshow(heatmap, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
            axes[i*2+1].set_title(f'Heatmap {idx+1}\nConf: {confidence:.3f}', 
                                 fontsize=10)
            axes[i*2+1].axis('off')
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            axes[i*2].text(0.5, 0.5, f'Error loading\nsample {idx}', 
                          ha='center', va='center', transform=axes[i*2].transAxes)
            axes[i*2].axis('off')
            axes[i*2+1].axis('off')

    max_used = len(sample_indices) * 2
    for j in range(max_used, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Sample Results: Original Images vs Heatmap Overlays', fontsize=16)
    plt.tight_layout()
    plt.savefig('chunking_results/sample_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def _is_bad_image(pred_bool: bool, mean_conf: float) -> bool:
    criterion = BAD_IMAGE_CRITERION
    if criterion == 'pred':
        return bool(pred_bool)
    if criterion == 'confidence':
        return float(mean_conf) >= float(BAD_CONF_THRESHOLD)
    # default: either
    return bool(pred_bool) or (float(mean_conf) >= float(BAD_CONF_THRESHOLD))

def _save_heatmap_outputs(heatmap: torch.Tensor, original_img_path: str, out_name_base: str):
    try:
        _, HEATMAP_RAW_DIR, HEATMAP_OVERLAY_DIR = setup_chunking_directories()
        
        if SAVE_HEATMAP_RAW:
            raw_path = os.path.join(HEATMAP_RAW_DIR, f"{out_name_base}.pt")
            torch.save(heatmap.cpu(), raw_path)
        
        if SAVE_HEATMAP_OVERLAY and os.path.exists(original_img_path):
            img = Image.open(original_img_path).convert('RGB')
            h_np = heatmap.detach().cpu().numpy()

            h_min, h_max = float(np.min(h_np)), float(np.max(h_np))
            if h_max > h_min:
                h_norm = (h_np - h_min) / (h_max - h_min)
            else:
                h_norm = np.zeros_like(h_np)
            
            if img.size != (h_np.shape[1], h_np.shape[0]):
                img = img.resize((h_np.shape[1], h_np.shape[0]))
            
            colored = cm.inferno(h_norm)[..., :3]
            colored_img = Image.fromarray((colored * 255).astype(np.uint8))
            
            blended = Image.blend(img, colored_img, alpha=0.5)
            overlay_path = os.path.join(HEATMAP_OVERLAY_DIR, f"{out_name_base}.png")
            blended.save(overlay_path)
    except Exception as e:
        print(f"Warning: failed to save heatmap outputs for {out_name_base}: {e}")

def evaluate(model, test_loader, test_df, device, use_channels_last=False, 
                          use_amp=False, is_torchscript=False, device_type='cuda', batch_size=128):

    RESULTS_DIR, HEATMAP_RAW_DIR, HEATMAP_OVERLAY_DIR = setup_chunking_directories()
    
    print("Starting evaluation...")
    model.eval()
    all_preds, all_labels = [], []
    all_heatmaps = []
    
    inference_start_time = time.time()
    with torch.no_grad():
        saved_bad_count = 0
        for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader, desc="Testing optimised model: ")):
            batch_preds, batch_heatmaps = detect_chunked(imgs, labels, model, device, 
                                                       use_channels_last, use_amp, 
                                                       is_torchscript, device_type)
            
            all_preds.extend(batch_preds)
            all_labels.extend(labels.cpu().numpy())
            all_heatmaps.extend(batch_heatmaps)

            start_index = batch_idx * batch_size
            for i, heatmap in enumerate(batch_heatmaps):
                try:
                    if isinstance(heatmap, torch.Tensor):
                        mean_conf = float(heatmap.mean().item())
                    else:
                        mean_conf = float(np.mean(heatmap))
                    pred_bool = bool(batch_preds[i])
                    if _is_bad_image(pred_bool, mean_conf):
                        global_idx = start_index + i
                        # Guard in case of mismatch
                        if 0 <= global_idx < len(test_df):
                            row = test_df.iloc[global_idx]
                            fname = os.path.splitext(os.path.basename(str(row['patch_filename'])))[0]
                            true_lbl = int(row['stenoza_label']) if 'stenoza_label' in row else -1
                            out_name = f"heatmap_{global_idx:06d}_pred{'pos' if pred_bool else 'neg'}_true{true_lbl}_conf{mean_conf:.3f}_{fname}"
                            _save_heatmap_outputs(heatmap if isinstance(heatmap, torch.Tensor) else torch.tensor(heatmap), 
                                                str(row['filepath']), out_name)
                            saved_bad_count += 1
                except Exception as e:
                    print(f"Warning: failed to process heatmap save for batch {batch_idx} item {i}: {e}")
    
    inference_time = time.time() - inference_start_time
    print(f"Inference time: {inference_time:.2f} seconds")
    
    if all_preds and all_labels and all_heatmaps:
        print("\n=== Creating Visualizations ===")
        save_detailed_results(all_preds, all_labels, all_heatmaps, test_df)

        print("\n=== Classification Report ===")
        print(classification_report(all_labels, all_preds, digits=4))
        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(all_labels, all_preds))
        
        print(f"\nSaved heatmaps for {saved_bad_count} bad images")
    else:
        print("Evaluation incomplete due to errors")
    
    return all_preds, all_labels, all_heatmaps, inference_time, saved_bad_count
