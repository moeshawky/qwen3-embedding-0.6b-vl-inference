"""Test image encoding directly - bypass vLLM config, create model manually."""
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from transformers import Qwen3Config, Siglip2VisionConfig, Siglip2VisionModel
import torch.nn as nn

sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl')

# Load composite config
config = AutoConfig.from_pretrained(
    '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl',
    trust_remote_code=True,
)
print("Config type:", type(config).__name__)
tc = config.get_text_config()
vc_config = config.get_vision_config()
print("Text config:", type(tc).__name__, "hidden:", tc.hidden_size)
print("Vision config:", type(vc_config).__name__, "hidden:", vc_config.hidden_size)

# Build the model components manually (mirroring Qwen3Siglip2EmbeddingModel)
text_model_cfg = tc
text_model = None  # We'll skip text model for image test

# Vision model
vision_model = Siglip2VisionModel(vc_config).cuda().float()
vision_model.eval()
for p in vision_model.parameters():
    p.requires_grad = False

# Vision projection
vision_proj = nn.Linear(config.projection_in_dim, config.projection_out_dim, bias=True).cuda().float()
for p in vision_proj.parameters():
    p.requires_grad = False

# Load weights from safetensors
from safetensors.torch import load_file
state = load_file('/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl/model.safetensors')

# Load vision weights (strip vision_model. prefix)
vision_state = {}
proj_state = {}
for k, v in state.items():
    if k.startswith('vision_model.'):
        vision_state[k[len('vision_model.'):]] = v
    elif k.startswith('vision_projection.'):
        proj_state[k[len('vision_projection.'):]] = v

missing, unexpected = vision_model.load_state_dict(vision_state, strict=False)
print(f"Vision missing: {len(missing)}, unexpected: {len(unexpected)}")
missing2, unexpected2 = vision_proj.load_state_dict(proj_state, strict=False)
print(f"Proj missing: {len(missing2)}, unexpected: {len(unexpected2)}")

# Test image encoding
model_dir = '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl'
ref_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32).eval().cuda()
processor = AutoImageProcessor.from_pretrained(model_dir)

img_dir = '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl-native/native/fixtures'
names = ['square', 'landscape', 'portrait', 'odd', 'tiny', 'wide']

ref_inputs = processor(images=[f"{img_dir}/{n}.png" for n in names], return_tensors="pt")
pv = ref_inputs['pixel_values'].cuda().float()
pam = ref_inputs.get('pixel_attention_mask', torch.ones(pv.shape[0], pv.shape[2], dtype=torch.int64)).cuda()
ss = ref_inputs.get('spatial_shapes', torch.tensor([[16, 16]] * 6)).cuda()

print(f"pixel_values shape: {pv.shape}")

# Encode through our manual vision model
import torch.nn.functional as F
with torch.no_grad():
    out = vision_model(
        pixel_values=pv,
        pixel_attention_mask=pam,
        spatial_shapes=ss,
    )
    v = F.normalize(out.pooler_output.float(), p=2, dim=-1)
    z = vision_proj(v)
    our_embeds = F.normalize(z, p=2, dim=-1)

# Reference through PyTorch
with torch.no_grad():
    ref_embeds = ref_model.get_image_features(
        pixel_values=pv,
        pixel_attention_mask=pam,
        spatial_shapes=ss,
    )

print(f"Our embeds shape: {our_embeds.shape}")
print(f"Ref embeds shape: {ref_embeds.shape}")

for i, name in enumerate(names):
    cos = F.cosine_similarity(our_embeds[i:i+1], ref_embeds[i:i+1])
    norm_o = torch.norm(our_embeds[i]).item()
    norm_r = torch.norm(ref_embeds[i]).item()
    print(f"{name}: our vs PyTorch cosine = {cos.item():.8f}, norm_ours={norm_o:.6f}, norm_ref={norm_r:.6f}")

print("Done!")
