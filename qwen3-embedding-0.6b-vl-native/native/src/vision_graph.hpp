#pragma once
// Native GGML implementation of the SigLIP2 naflex vision path.
//
// Graph (mirrors `transformers.models.siglip2.modeling_siglip2`):
//
//   patch_embeds   = Linear(v.patch_embd) applied to the 256x768 patch vectors
//   embeddings     = patch_embeds + interpolated position embeddings
//   27 x { LayerNorm -> MHA(16 heads x 72) -> residual
//          LayerNorm -> MLP(1152->4304, gelu_tanh, 4304->1152) -> residual }
//   post_layernorm
//   MAP head       = MHA-pooling with a learned probe, then residual + LayerNorm
//                    + MLP
//   image_embeds   = normalize(normalize(pooler) @ mm.model.fc + bias)
//
// Attention masking follows `pixel_attention_mask`: padded patch positions get
// an additive -inf logit, so they contribute nothing to the softmax.

#include "sidecar.hpp"

#include <ggml.h>

#include <cstdint>
#include <string>
#include <vector>

namespace qvl {

/// Static architecture constants (mirrors config.json's vision_config).
struct VisionConfig {
    int n_embd = 1152;
    int n_head = 16;
    int head_dim = 72;
    int n_layer = 27;
    int ffn_dim = 4304;
    int patch_dim = 768;   ///< 3 * 16 * 16
    int patch_size = 16;
    int max_patches = 256;
    int out_dim = 1024;
    float ln_eps = 1e-6f;
    bool use_cuda = false;  ///< Use CUDA backend instead of CPU
};

/// Model inputs for one image.
struct VisionInputs {
    std::vector<float> pixel_values;  ///< [max_patches, patch_dim], patch-major
    std::vector<int32_t> mask;        ///< [max_patches], 1 = real patch
    int grid_h = 0;
    int grid_w = 0;
};

/// Model inputs for one image.
struct VisionInputs {
    std::vector<float> pixel_values;  ///< [max_patches, patch_dim], patch-major
    std::vector<int32_t> mask;        ///< [max_patches], 1 = real patch
    int grid_h = 0;
    int grid_w = 0;
};

/// Model outputs (kept in numpy-friendly row-major layouts).
struct VisionOutputs {
    std::vector<float> patch_plus_position;  ///< [max_patches, n_embd]
    std::vector<float> layer_0;              ///< [max_patches, n_embd]
    std::vector<float> layer_1;
    std::vector<float> layer_13;
    std::vector<float> layer_26;
    std::vector<float> post_layernorm;       ///< [max_patches, n_embd]
    std::vector<float> map_pooler;           ///< [n_embd]
    std::vector<float> map_pooler_norm;      ///< [n_embd]
    std::vector<float> linear_projection;    ///< [out_dim]
    std::vector<float> image_embeds;         ///< [out_dim]
};

/// Executes the vision graph on a CPU ggml backend.
class VisionRunner {
public:
    VisionRunner(Sidecar & sidecar, const VisionConfig & cfg);
    ~VisionRunner();

    VisionRunner(const VisionRunner &) = delete;
    VisionRunner & operator=(const VisionRunner &) = delete;

    /// Sets the worker thread count (0 = ggml default).
    void set_threads(int n_threads);

    /// Runs the graph and returns the outputs.
    VisionOutputs run(const VisionInputs & in, const std::vector<float> & positions);

private:
    Sidecar & sidecar_;
    VisionConfig cfg_;
    ggml_backend_t backend_ = nullptr;
};

} // namespace qvl
