#include "vision_graph.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace qvl {
namespace {

/// y = x @ W^T + b  with W stored as ggml ne {in, out}.
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    return ggml_add(ctx, y, b);
}

/// LayerNorm over the feature dimension (ne[0]) with the given eps.
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    return ggml_add(ctx, y, b);
}

/// Exact L2 normalisation: x / ||x||_2.
///
/// NOTE: ggml_norm is a *LayerNorm* (it subtracts the mean), so it cannot be
/// used for L2 normalisation; the norm is built from sqr -> sum -> sqrt.
ggml_tensor * l2_normalize(ggml_context * ctx, ggml_tensor * x, int n) {
    (void) n;
    ggml_tensor * sq = ggml_sqr(ctx, x);
    ggml_tensor * sum = ggml_sum(ctx, sq);
    ggml_tensor * nrm = ggml_sqrt(ctx, sum);
    return ggml_div(ctx, x, nrm);
}

/// Multi-head attention over `n_tokens` positions.
///
/// q, k, v are [n_embd, n_tokens]; the returned tensor is [n_embd, n_tokens]
/// with the heads concatenated as (head * head_dim + dim).
ggml_tensor * attention(ggml_context * ctx,
                        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                        int n_query, int n_key, int n_head, int head_dim,
                        ggml_tensor * mask_bias, float scale) {
    const int n_embd = n_head * head_dim;
    // Split the feature dimension into heads. The head stride is head_dim
    // floats inside a token row; the token stride is the parent row stride.
    ggml_tensor * qv = ggml_view_3d(ctx, q, head_dim, n_query, n_head, q->nb[1], head_dim * sizeof(float), 0);
    ggml_tensor * kv = ggml_view_3d(ctx, k, head_dim, n_key, n_head, k->nb[1], head_dim * sizeof(float), 0);
    ggml_tensor * vv = ggml_view_3d(ctx, v, head_dim, n_key, n_head, v->nb[1], head_dim * sizeof(float), 0);

    // kq[ne0 = kv, ne1 = query, ne2 = head]
    ggml_tensor * kq = ggml_mul_mat(ctx, kv, qv);
    kq = ggml_scale(ctx, kq, scale);
    if (mask_bias != nullptr) {
        kq = ggml_add(ctx, kq, mask_bias);
    }
    kq = ggml_soft_max(ctx, kq);

    // V has to be transposed so that the contracted dimension (kv) is the
    // contiguous one, which is what ggml_mul_mat requires.
    ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vv));

    // att[ne0 = head_dim, ne1 = query, ne2 = head]
    ggml_tensor * att = ggml_mul_mat(ctx, vt, kq);

    // Merge the heads: (head_dim, head, query) -> (n_embd, query)
    att = ggml_cont(ctx, ggml_permute(ctx, att, 0, 2, 1, 3));
    return ggml_reshape_2d(ctx, att, n_embd, n_query);
}

} // namespace

VisionRunner::VisionRunner(Sidecar & sidecar, const VisionConfig & cfg)
    : sidecar_(sidecar), cfg_(cfg) {
    if (cfg_.use_cuda) {
        backend_ = ggml_backend_cuda_init();
        if (backend_ == nullptr) {
            throw std::runtime_error("ggml: could not initialise the CUDA backend");
        }
    } else {
        backend_ = ggml_backend_cpu_init();
        if (backend_ == nullptr) {
            throw std::runtime_error("ggml: could not initialise the CPU backend");
        }
    }
}

VisionRunner::~VisionRunner() {
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

void VisionRunner::set_threads(int n_threads) {
    if (n_threads > 0 && !cfg_.use_cuda) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
}

VisionOutputs VisionRunner::run(const VisionInputs & in, const std::vector<float> & positions) {
    const int n_embd = cfg_.n_embd;
    const int n_tokens = cfg_.max_patches;
    const int patch_dim = cfg_.patch_dim;

    if (static_cast<int>(in.pixel_values.size()) != n_tokens * patch_dim) {
        throw std::runtime_error("vision: pixel_values has the wrong size");
    }
    if (static_cast<int>(positions.size()) != n_tokens * n_embd) {
        throw std::runtime_error("vision: position embeddings have the wrong size");
    }

    // Graph context: no_alloc, the allocator assigns the memory.
    struct ggml_init_params params = {
        /*mem_size =*/ 512u * 1024u * 1024u,
        /*mem_buffer =*/ nullptr,
        /*no_alloc =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml: could not create the graph context");
    }

    // --- inputs -----------------------------------------------------------
    ggml_tensor * pv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, patch_dim, n_tokens);
    ggml_set_name(pv, "pixel_values");
    ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(pos, "position_embeds");

    // Additive attention bias: 0 for real patches, -inf for padded ones.
    ggml_tensor * mask_bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_tokens, 1, 1);
    ggml_set_name(mask_bias, "attn_mask_bias");

    std::vector<float> mask_values(static_cast<size_t>(n_tokens), 0.0f);
    for (int i = 0; i < n_tokens; ++i) {
        const bool valid = i < static_cast<int>(in.mask.size()) && in.mask[static_cast<size_t>(i)] != 0;
        mask_values[static_cast<size_t>(i)] = valid ? 0.0f : -std::numeric_limits<float>::infinity();
    }

    // --- embeddings -------------------------------------------------------
    ggml_tensor * w_patch = sidecar_.get("v.patch_embd.weight");   // {768, 1152}
    ggml_tensor * b_patch = sidecar_.get("v.patch_embd.bias");     // {1152}
    ggml_tensor * embeds = ggml_add(ctx, linear(ctx, w_patch, b_patch, pv), pos);

    ggml_tensor * h = embeds;
    ggml_tensor * layer_0 = nullptr;
    ggml_tensor * layer_1 = nullptr;
    ggml_tensor * layer_13 = nullptr;
    ggml_tensor * layer_26 = nullptr;

    const float att_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

    for (int il = 0; il < cfg_.n_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        ggml_tensor * ln1_w = sidecar_.get(p + "ln1.weight");
        ggml_tensor * ln1_b = sidecar_.get(p + "ln1.bias");
        ggml_tensor * ln2_w = sidecar_.get(p + "ln2.weight");
        ggml_tensor * ln2_b = sidecar_.get(p + "ln2.bias");
        ggml_tensor * wq = sidecar_.get(p + "attn_q.weight");
        ggml_tensor * bq = sidecar_.get(p + "attn_q.bias");
        ggml_tensor * wk = sidecar_.get(p + "attn_k.weight");
        ggml_tensor * bk = sidecar_.get(p + "attn_k.bias");
        ggml_tensor * wv = sidecar_.get(p + "attn_v.weight");
        ggml_tensor * bv = sidecar_.get(p + "attn_v.bias");
        ggml_tensor * wo = sidecar_.get(p + "attn_out.weight");
        ggml_tensor * bo = sidecar_.get(p + "attn_out.bias");
        ggml_tensor * wfu = sidecar_.get(p + "ffn_up.weight");
        ggml_tensor * bfu = sidecar_.get(p + "ffn_up.bias");
        ggml_tensor * wfd = sidecar_.get(p + "ffn_down.weight");
        ggml_tensor * bfd = sidecar_.get(p + "ffn_down.bias");

        // --- attention block ---
        ggml_tensor * x1 = layer_norm(ctx, h, ln1_w, ln1_b, cfg_.ln_eps);
        ggml_tensor * q = linear(ctx, wq, bq, x1);
        ggml_tensor * k = linear(ctx, wk, bk, x1);
        ggml_tensor * v = linear(ctx, wv, bv, x1);
        ggml_tensor * att = attention(ctx, q, k, v, n_tokens, n_tokens, cfg_.n_head, cfg_.head_dim, mask_bias, att_scale);
        att = linear(ctx, wo, bo, att);
        h = ggml_add(ctx, h, att);

        // --- MLP block ---
        ggml_tensor * x2 = layer_norm(ctx, h, ln2_w, ln2_b, cfg_.ln_eps);
        ggml_tensor * ff = linear(ctx, wfu, bfu, x2);
        ff = ggml_gelu(ctx, ff);
        ff = linear(ctx, wfd, bfd, ff);
        h = ggml_add(ctx, h, ff);

        if (il == 0) layer_0 = h;
        if (il == 1) layer_1 = h;
        if (il == 13) layer_13 = h;
        if (il == 26) layer_26 = h;
    }

    // --- post layernorm ---------------------------------------------------
    ggml_tensor * post_ln_w = sidecar_.get("v.post_ln.weight");
    ggml_tensor * post_ln_b = sidecar_.get("v.post_ln.bias");
    ggml_tensor * last = layer_norm(ctx, h, post_ln_w, post_ln_b, cfg_.ln_eps);

    // --- MAP head ---------------------------------------------------------
    ggml_tensor * probe = sidecar_.get("p.probe");                  // {1152,1,1}
    ggml_tensor * w_q = sidecar_.get("p.attention.in_proj_weight.q");
    ggml_tensor * w_k = sidecar_.get("p.attention.in_proj_weight.k");
    ggml_tensor * w_v = sidecar_.get("p.attention.in_proj_weight.v");
    ggml_tensor * b_q = sidecar_.get("p.attention.in_proj_bias.q");
    ggml_tensor * b_k = sidecar_.get("p.attention.in_proj_bias.k");
    ggml_tensor * b_v = sidecar_.get("p.attention.in_proj_bias.v");
    ggml_tensor * w_out = sidecar_.get("p.attention.out_proj.weight");
    ggml_tensor * b_out = sidecar_.get("p.attention.out_proj.bias");
    ggml_tensor * map_ln_w = sidecar_.get("p.layernorm.weight");
    ggml_tensor * map_ln_b = sidecar_.get("p.layernorm.bias");
    ggml_tensor * w_m1 = sidecar_.get("p.mlp.fc1.weight");
    ggml_tensor * b_m1 = sidecar_.get("p.mlp.fc1.bias");
    ggml_tensor * w_m2 = sidecar_.get("p.mlp.fc2.weight");
    ggml_tensor * b_m2 = sidecar_.get("p.mlp.fc2.bias");

    // The probe is the query; the last hidden state supplies keys and values.
    ggml_tensor * probe2d = ggml_reshape_2d(ctx, probe, n_embd, 1);
    ggml_tensor * q = linear(ctx, w_q, b_q, probe2d);
    ggml_tensor * k = linear(ctx, w_k, b_k, last);
    ggml_tensor * v = linear(ctx, w_v, b_v, last);

    // The MAP head pools with a single probe query over all patch positions.
    ggml_tensor * pool_att = attention(ctx, q, k, v, 1, n_tokens, cfg_.n_head, cfg_.head_dim, mask_bias, att_scale);
    pool_att = linear(ctx, w_out, b_out, pool_att);

    ggml_tensor * residual = pool_att;
    ggml_tensor * y = layer_norm(ctx, residual, map_ln_w, map_ln_b, cfg_.ln_eps);
    ggml_tensor * mlp = linear(ctx, w_m1, b_m1, y);
    mlp = ggml_gelu(ctx, mlp);
    mlp = linear(ctx, w_m2, b_m2, mlp);
    ggml_tensor * pooler = ggml_add(ctx, residual, mlp);

    // --- projection -------------------------------------------------------
    ggml_tensor * w_fc = sidecar_.get("mm.model.fc.weight");        // {1152, 1024}
    ggml_tensor * b_fc = sidecar_.get("mm.model.fc.bias");          // {1024}
    ggml_tensor * pooler_norm = l2_normalize(ctx, pooler, n_embd);
    ggml_tensor * proj = linear(ctx, w_fc, b_fc, pooler_norm);
    ggml_tensor * image_embeds = l2_normalize(ctx, proj, cfg_.out_dim);

    // --- graph ------------------------------------------------------------
    // Every tensor that is read back must be COPIED: the ggml allocator
    // reuses the memory of an intermediate as soon as its last consumer has
    // run, so a post-hoc read would observe overwritten data.
    ggml_tensor * o_embeds = ggml_dup(ctx, embeds);
    ggml_tensor * o_layer_0 = ggml_dup(ctx, layer_0);
    ggml_tensor * o_layer_1 = ggml_dup(ctx, layer_1);
    ggml_tensor * o_layer_13 = ggml_dup(ctx, layer_13);
    ggml_tensor * o_layer_26 = ggml_dup(ctx, layer_26);
    ggml_tensor * o_last = ggml_dup(ctx, last);
    ggml_tensor * o_pooler = ggml_dup(ctx, pooler);
    ggml_tensor * o_pooler_norm = ggml_dup(ctx, pooler_norm);
    ggml_tensor * o_proj = ggml_dup(ctx, proj);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, image_embeds);
    ggml_build_forward_expand(gf, o_embeds);
    ggml_build_forward_expand(gf, o_layer_0);
    ggml_build_forward_expand(gf, o_layer_1);
    ggml_build_forward_expand(gf, o_layer_13);
    ggml_build_forward_expand(gf, o_layer_26);
    ggml_build_forward_expand(gf, o_last);
    ggml_build_forward_expand(gf, o_pooler);
    ggml_build_forward_expand(gf, o_pooler_norm);
    ggml_build_forward_expand(gf, o_proj);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (galloc == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml: could not create the graph allocator");
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        throw std::runtime_error("ggml: could not allocate the graph");
    }

    // Upload the inputs.
    ggml_backend_tensor_set(pv, in.pixel_values.data(), 0, in.pixel_values.size() * sizeof(float));
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(float));
    ggml_backend_tensor_set(mask_bias, mask_values.data(), 0, mask_values.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend_, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        throw std::runtime_error("ggml: graph compute failed");
    }

    // --- collect outputs --------------------------------------------------
    VisionOutputs out;
    auto fetch = [&](ggml_tensor * t) {
        std::vector<float> buf(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        return buf;
    };

    // ggml ne {n_embd, n_tokens} already matches the numpy [n_tokens, n_embd]
    // row-major layout, so no transpose is needed.
    out.patch_plus_position = fetch(o_embeds);
    out.layer_0 = fetch(o_layer_0);
    out.layer_1 = fetch(o_layer_1);
    out.layer_13 = fetch(o_layer_13);
    out.layer_26 = fetch(o_layer_26);
    out.post_layernorm = fetch(o_last);
    out.map_pooler = fetch(o_pooler);
    out.map_pooler_norm = fetch(o_pooler_norm);
    out.linear_projection = fetch(o_proj);
    out.image_embeds = fetch(image_embeds);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return out;
}

} // namespace qvl
