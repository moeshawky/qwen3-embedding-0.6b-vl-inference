#pragma once
// Native text-branch embedding for moeshawky/qwen3-embedding-0.6b-vl.
//
// The text tower is the published `Qwen3-Embedding-0.6B` Q8_0 GGUF, executed
// by llama.cpp (CPU, GGML). The composite model then applies its trained
// 1024->1024 "compat head" and a final L2 normalisation:
//
//     v = normalize(last_token_hidden_state)
//     z = compat_head(v)
//     return normalize(z)
//
// The compat head is NOT part of the published text GGUF and that GGUF must not
// be mutated, so the head is supplied as a separate opt-in artifact
// (`compat_head_qwen3_embedding_0.6b.f32.gguf`, exported by
// tools/export_compat_head.py from the source checkpoint).
//
// Tokenisation follows the model's frozen recipe: no special tokens are added
// by the tokenizer, the EOS id is appended as the terminator, and pooling
// happens at that terminator (the last token), which matches the GGUF's
// `qwen3.pooling_type = last`.

#include <ggml.h>
#include <llama.h>

#include <cstdint>
#include <string>
#include <vector>

namespace qvl {

/// Executes the text branch and the compatibility head.
class TextEmbedder {
public:
    /// Loads the text GGUF and the compatibility head. Throws on failure.
    TextEmbedder(const std::string & model_path, const std::string & compat_head_path, int n_threads, bool use_cuda = false);
    ~TextEmbedder();

    TextEmbedder(const TextEmbedder &) = delete;
    TextEmbedder & operator=(const TextEmbedder &) = delete;

    /// Embeds one text into a 1024-d L2-normalised vector.
    std::vector<float> embed(const std::string & text);

    /// Number of tokens produced for the last embedded text (content + EOS).
    int last_token_count() const { return last_tokens_; }

private:
    void load_head(const std::string & path);
    void free_head();

    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;

    // Compatibility head, held in its own ggml context + CPU buffer.
    ggml_context * head_ctx_ = nullptr;
    ggml_backend_buffer_t head_buf_ = nullptr;
    ggml_backend_t head_backend_ = nullptr;
    ggml_tensor * head_w_ = nullptr;
    ggml_tensor * head_b_ = nullptr;
    int head_dim_ = 0;

    int n_threads_ = 1;
    int last_tokens_ = 0;
};

} // namespace qvl
