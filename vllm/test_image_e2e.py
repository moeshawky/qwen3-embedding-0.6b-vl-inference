"""Full end-to-end image embedding test - bypass vLLM config, instantiate manually."""
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoImageProcessor, Siglip2VisionConfig, Siglip2VisionModel
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/vllm')
sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl')
os.environ['VLLM_PLUGINS'] = 'qwen3_vl_vllm.register'

from qwen3_vl_vllm import register  # noqa: F401
from qwen3_vl_vllm.preprocess import preprocess_images, load_image

# Build the model components manually
config = AutoConfig.from_pretrained(
    '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl',
    trust_remote_code=True,
)
vc_config = config.get_vision_config()

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
state = load_file('/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl/model.safetensors')

vision_state = {k[len('vision_model.'):]: v for k, v in state.items() if k.startswith('vision_model.')}
proj_state = {k[len('vision_projection.'):]: v for k, v in state.items() if k.startswith('vision_projection.')}
missing, unexpected = vision_model.load_state_dict(vision_state, strict=False)
missing2, unexpected2 = vision_proj.load_state_dict(proj_state, strict=False)
assert not missing and not missing2, f"Vision missing: {missing}, {missing2}"
print("Vision weights loaded successfully")

# Define encode_images function (mirrors Qwen3Siglip2EmbeddingModel.encode_images)
def encode_images(pixel_values, pixel_attention_mask, spatial_shapes):
    vision_model.eval()
    with torch.no_grad():
        out = vision_model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        v = F.normalize(out.pooler_output.float(), p=2, dim=-1)
        z = vision_proj(v)
        return F.normalize(z, p=2, dim=-1)

# Load images and preprocess
img_dir = '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl-native/native/fixtures'
names = ['square', 'landscape', 'portrait', 'odd', 'tiny', 'wide']
images = [load_image(f'{img_dir}/{n}.png') for n in names]

inputs = preprocess_images(images)
pv = torch.from_numpy(inputs['pixel_values']).cuda().float()
pam = torch.from_numpy(inputs['pixel_attention_mask']).cuda()
ss = torch.from_numpy(inputs['spatial_shapes']).cuda()

print(f"pixel_values: {pv.shape}, mask: {pam.shape}, shapes: {ss}")

# Encode through our pipeline
our_embeds = encode_images(pv, pam, ss)

# Reference through PyTorch
ref_model = AutoModel.from_pretrained('/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl', trust_remote_code=True, dtype=torch.float32).eval().cuda()
processor = AutoImageProcessor.from_pretrained('/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl')
ref_inputs = processor(images=[f'{img_dir}/{n}.png' for n in names], return_tensors="pt")
ref_pv = ref_inputs['pixel_values'].cuda().float()
ref_pam = ref_inputs.get('pixel_attention_mask', torch.ones(ref_pv.shape[0], ref_pv.shape[2], dtype=torch.int64)).cuda()
ref_ss = ref_inputs.get('spatial_shapes', torch.tensor([[16, 16]] * 6)).cuda()

with torch.no_grad():
    ref_embeds = ref_model.get_image_features(pixel_values=ref_pv, pixel_attention_mask=ref_pam, spatial_shapes=ref_ss)

print(f"Our embeds: {our_embeds.shape}, Ref embeds: {ref_embeds.shape}")

all_pass = True
for i, name in enumerate(names):
    cos = F.cosine_similarity(our_embeds[i:i+1], ref_embeds[i:i+1])
    norm_o = torch.norm(our_embeds[i]).item()
    norm_r = torch.norm(ref_embeds[i]).item()
    status = "PASS" if cos.item() > 0.999 else "FAIL"
    if cos.item() <= 0.999:
        all_pass = False
    print(f"{name}: {status} cos={cos.item():.8f}, norm_ours={norm_o:.6f}, norm_ref={norm_r:.6f}")

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
