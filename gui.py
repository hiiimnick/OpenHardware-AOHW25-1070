#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms
import os
import numpy as np
from matplotlib import cm
from evaluation import detect_chunked

def load_model(model_path, device="cuda"):
    model = models.densenet121(pretrained=False)

    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, 1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def generate_heatmap(image_path, model_path, device="cuda"):
    model = load_model(model_path, device)

    img = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        _, heatmaps = detect_chunked(tensor, [0], model, device)

    heatmap = heatmaps[0]
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()

    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    h_norm = (heatmap - h_min) / (h_max - h_min + 1e-8)

    if img.size != (h_norm.shape[1], h_norm.shape[0]):
        img = img.resize((h_norm.shape[1], h_norm.shape[0]))
    colored = cm.inferno(h_norm)[..., :3]
    colored_img = Image.fromarray((colored * 255).astype("uint8"))
    blended = Image.blend(img, colored_img, alpha=0.5)

    return blended


# === GUI ===
class HeatmapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heatmap Generator")

        self.image_path = None
        self.model_path = None

        # Buttons
        tk.Button(root, text="Load Image", command=self.load_image).pack(pady=5)
        tk.Button(root, text="Load Model (.pt)", command=self.load_model).pack(pady=5)
        tk.Button(root, text="Generate Heatmap", command=self.run_heatmap).pack(pady=10)

        # Canvas
        self.canvas = tk.Label(root)
        self.canvas.pack()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if path:
            self.image_path = path
            img = Image.open(path).resize((256, 256))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.configure(image=img_tk)
            self.canvas.image = img_tk

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if path:
            self.model_path = path
            messagebox.showinfo("Model Loaded", f"Loaded model:\n{path}")

    def run_heatmap(self):
        if not self.image_path or not self.model_path:
            messagebox.showerror("Error", "Please load both image and model.")
            return
        try:
            blended = generate_heatmap(self.image_path, self.model_path)
            img_tk = ImageTk.PhotoImage(blended.resize((256, 256)))
            self.canvas.configure(image=img_tk)
            self.canvas.image = img_tk
            
            out_path = os.path.join("chunking_results", "gui_output.png")
            os.makedirs("chunking_results", exist_ok=True)
            blended.save(out_path)
            messagebox.showinfo("Saved", f"Heatmap saved to {out_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = HeatmapGUI(root)
    root.mainloop()
