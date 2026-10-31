"""Test NaFlex preprocessing vs HF processor."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/vllm')
from qwen3_vl_vllm.preprocess import preprocess_image, load_image

from transformers import AutoImageProcessor
model_dir = '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl'
processor = AutoImageProcessor.from_pretrained(model_dir)

img_dir = '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl-native/native/fixtures'
names = ['square', 'landscape', 'portrait', 'odd', 'tiny', 'wide']

for name in names:
    # Our preprocessing
    img = load_image(f"{img_dir}/{name}.png")
    ours = preprocess_image(img)
    
    # HF preprocessing
    hf_inputs = processor(images=[f"{img_dir}/{name}.png"], return_tensors="pt")
    hf_pv = hf_inputs['pixel_values'][0].numpy()  # [256, 768]
    hf_pam = hf_inputs.get('pixel_attention_mask', torch.ones(256, dtype=torch.int64))[0].numpy()
    hf_ss = hf_inputs.get('spatial_shapes', torch.tensor([16, 16]))[0].numpy()
    
    # Compare
    pv_diff = np.max(np.abs(ours['pixel_values'] - hf_pv))
    pam_diff = np.max(np.abs(ours['pixel_attention_mask'] - hf_pam))
    ss_diff = np.max(np.abs(ours['spatial_shapes'] - hf_ss))
    
    print(f"{name}: pv_max_diff={pv_diff:.2e}, mask_diff={pam_diff}, ss_diff={ss_diff}")
    print(f"  ours grid: {ours['spatial_shapes']}, valid: {ours['pixel_attention_mask'].sum()}")
    print(f"  hf grid: {hf_ss}, valid: {hf_pam.sum()}")

print("Done!")
