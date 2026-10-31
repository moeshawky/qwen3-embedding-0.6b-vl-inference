#include "text_embed.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <gguf.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace qvl {
namespace {

/// L2-normalises a vector in place.
void l2_normalize(std::vector<float> & v) {
    double sum = 0.0;
    for (float x : v) {
        sum += static_cast<double>(x) * static_cast<double>(x);
    }
    const float norm = static_cast<float>(std::sqrt(sum));
    if (norm <= 0.0f) {
        throw std::runtime_error("text: cannot normalise a zero vector");
    }
    for (float & x : v) {
        x /= norm;
    }
}

} // namespace

TextEmbedder::TextEmbedder(const std::string & model_path, const std::string & compat_head_path,
                           int n_threads, bool use_cuda) {
    n_threads_ = n_threads > 0 ? n_threads : 1;

    llama_backend_init();

    // llama.cpp is chatty at graph-reservation time; keep stderr for errors
    // only (the caller parses stdout).
    llama_log_set(
        [](ggml_log_level level, const char * text, void * /*ud*/) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                std::fputs(text, stderr);
            }
        },
        nullptr);

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = use_cuda ? 33 : 0;  // Offload all 28+ layers to GPU
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model_ == nullptr) {
        throw std::runtime_error("text: failed to load " + model_path);
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    cparams.n_threads = n_threads_;
    cparams.n_threads_batch = n_threads_;
    cparams.no_perf = true;
    cparams.embeddings = true;
    ctx_ = llama_init_from_model(model_, cparams);
    if (ctx_ == nullptr) {
        throw std::runtime_error("text: failed to create the llama context");
    }

    load_head(compat_head_path, use_cuda);
}

TextEmbedder::~TextEmbedder() {
    free_head();
    if (ctx_ != nullptr) {
        llama_free(ctx_);
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
    }
    llama_backend_free();
}

void TextEmbedder::load_head(const std::string & path, bool use_cuda) {
    struct gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr) {
        throw std::runtime_error("text: failed to open the compat head " + path);
    }

    const int64_t w_id = gguf_find_tensor(gguf, "compat_head.weight");
    const int64_t b_id = gguf_find_tensor(gguf, "compat_head.bias");
    if (w_id < 0 || b_id < 0) {
        gguf_free(gguf);
        throw std::runtime_error("text: the compat head artifact is missing its tensors");
    }
    if (gguf_get_tensor_type(gguf, w_id) != GGML_TYPE_F32 ||
        gguf_get_tensor_type(gguf, b_id) != GGML_TYPE_F32) {
        gguf_free(gguf);
        throw std::runtime_error("text: the compat head must be stored in F32");
    }

    const int64_t * wne = gguf_get_tensor_ne(gguf, w_id);
    const int64_t * bne = gguf_get_tensor_ne(gguf, b_id);
    const int64_t dim_out = wne[1];
    const int64_t dim_in = wne[0];
    if (bne[0] != dim_out || dim_in != dim_out) {
        gguf_free(gguf);
        throw std::runtime_error("text: unexpected compat head shapes");
    }

    // Use CUDA backend for the head if requested.
    ggml_backend_t head_backend = use_cuda ? ggml_backend_cuda_init() : ggml_backend_cpu_init();
    if (head_backend == nullptr) {
        gguf_free(gguf);
        throw std::runtime_error("text: could not initialise head backend");
    }
    head_backend_ = head_backend;

    struct ggml_init_params hparams = {
        /*mem_size =*/ 8u * 1024u * 1024u,
        /*mem_buffer =*/ nullptr,
        /*no_alloc =*/ true,
    };
    head_ctx_ = ggml_init(hparams);
    head_w_ = ggml_new_tensor_2d(head_ctx_, GGML_TYPE_F32, dim_in, dim_out);
    head_b_ = ggml_new_tensor_2d(head_ctx_, GGML_TYPE_F32, dim_out, 1);
    head_buf_ = ggml_backend_alloc_ctx_tensors(head_ctx_, head_backend_);
    if (head_buf_ == nullptr) {
        gguf_free(gguf);
        throw std::runtime_error("text: failed to allocate the compat head buffer");
    }

    // The GGUF payload is read through the same file mapping the sidecar uses.
    gguf_free(gguf);

    // Re-open through a plain read so the payload bytes are contiguous.
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        throw std::runtime_error("text: could not read the compat head payload");
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(static_cast<size_t>(size));
    if (std::fread(blob.data(), 1, blob.size(), f) != blob.size()) {
        std::fclose(f);
        throw std::runtime_error("text: short read on the compat head payload");
    }
    std::fclose(f);

    // Locate the tensor payloads inside the file: header + kv + tensor infos,
    // then the aligned data blob.
    struct gguf_init_params p2 = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * g2 = gguf_init_from_buffer(blob.data(), blob.size(), p2);
    if (g2 == nullptr) {
        throw std::runtime_error("text: failed to parse the compat head");
    }
    const size_t data_off = gguf_get_data_offset(g2);
    const size_t w_off = data_off + gguf_get_tensor_offset(g2, gguf_find_tensor(g2, "compat_head.weight"));
    const size_t b_off = data_off + gguf_get_tensor_offset(g2, gguf_find_tensor(g2, "compat_head.bias"));
    gguf_free(g2);

    ggml_backend_tensor_set(head_w_, blob.data() + w_off, 0,
                             static_cast<size_t>(dim_in * dim_out) * sizeof(float));
    ggml_backend_tensor_set(head_b_, blob.data() + b_off, 0,
                             static_cast<size_t>(dim_out) * sizeof(float));

    head_dim_ = static_cast<int>(dim_out);
}

void TextEmbedder::free_head() {
    if (head_buf_ != nullptr) {
        ggml_backend_buffer_free(head_buf_);
        head_buf_ = nullptr;
    }
    if (head_ctx_ != nullptr) {
        ggml_free(head_ctx_);
        head_ctx_ = nullptr;
    }
    if (head_backend_ != nullptr) {
        ggml_backend_free(head_backend_);
        head_backend_ = nullptr;
    }
}

std::vector<float> TextEmbedder::embed(const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(model_);

    // Tokenise without special tokens, then append the EOS terminator: this is
    // the model's frozen recipe (see processing_qwen_siglip2_embedding.py).
    std::vector<llama_token> tokens(text.size() + 8);
    int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                tokens.data(), static_cast<int32_t>(tokens.size()),
                                /*add_special =*/ false, /*parse_special =*/ false);
    if (n < 0) {
        throw std::runtime_error("text: tokenisation failed");
    }
    tokens.resize(static_cast<size_t>(n));
    tokens.push_back(llama_vocab_eos(vocab));
    last_tokens_ = static_cast<int>(tokens.size());

    if (std::getenv("QVL_DEBUG_TOKENS") != nullptr) {
        std::fprintf(stderr, "[tokens n=%d]", last_tokens_);
        for (llama_token t : tokens) {
            std::fprintf(stderr, " %d", (int) t);
        }
        std::fprintf(stderr, "\n");
    }

    // The KV cache must be cleared between prompts: llama_decode keeps the
    // previous sequence's keys/values, and a causal model would then attend
    // across prompts.
    llama_memory_clear(llama_get_memory(ctx_), true);

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx_, batch) != 0) {
        throw std::runtime_error("text: llama_decode failed");
    }

    const float * pooled = llama_get_embeddings_seq(ctx_, 0);
    if (pooled == nullptr) {
        throw std::runtime_error("text: no pooled embedding was produced");
    }

    // 1) L2-normalise the last-token hidden state.
    std::vector<float> v(static_cast<size_t>(head_dim_));
    std::memcpy(v.data(), pooled, v.size() * sizeof(float));
    l2_normalize(v);

    // 2) Compatibility head, evaluated in ggml.
    struct ggml_init_params gparams = {
        /*mem_size =*/ 8u * 1024u * 1024u,
        /*mem_buffer =*/ nullptr,
        /*no_alloc =*/ true,
    };
    ggml_context * gctx = ggml_init(gparams);
    ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, head_dim_, 1);
    ggml_tensor * y = ggml_add(gctx, ggml_mul_mat(gctx, head_w_, x), head_b_);
    // Exact L2 normalisation of the projected vector.
    ggml_tensor * sq = ggml_sqr(gctx, y);
    ggml_tensor * nrm = ggml_sqrt(gctx, ggml_sum(gctx, sq));
    ggml_tensor * z = ggml_div(gctx, y, nrm);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 64, false);
    ggml_build_forward_expand(gf, z);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(head_backend_));
    if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_free(gctx);
        throw std::runtime_error("text: failed to allocate the compat head graph");
    }
    ggml_backend_tensor_set(x, v.data(), 0, v.size() * sizeof(float));
    if (ggml_backend_graph_compute(head_backend_, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(gctx);
        throw std::runtime_error("text: compat head compute failed");
    }

    std::vector<float> out(static_cast<size_t>(head_dim_));
    ggml_backend_tensor_get(z, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(gctx);
    return out;
}

} // namespace qvl
