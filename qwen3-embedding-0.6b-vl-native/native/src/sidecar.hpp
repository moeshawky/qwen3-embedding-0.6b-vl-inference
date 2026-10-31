#pragma once
// Sidecar (mmproj) GGUF loading for the native GGML vision path.
//
// The sidecar is a 450-tensor GGUF holding the SigLIP2 vision tower, the
// multi-head attention pooling (MAP) head and the 1152->1024 vision
// projection. Weights are dequantised to F32 once at load time so the compute
// graph runs in full FP32: the Q8_0 rounding error then comes only from the
// stored weights, not from on-the-fly activation quantisation.
//
// Memory layout contract (verified against the published sidecar):
//   * GGUF stores tensors in ggml `ne` order (ne[0] is the fastest dim).
//   * For a torch nn.Linear weight W of shape [out, in], the sidecar stores
//     W.T, i.e. ggml ne = {in, out}. A mul_mat(a, w) with a->ne[0] == in
//     therefore reproduces the torch Linear exactly.
//   * `v.blk.*.ffn_down.weight` and `p.mlp.fc2.weight` carry 16 zero-padded
//     extra *input* features (ggml ne[0] = 4320 instead of 4304). The padding
//     is trimmed here so the graph sees the true 4304-wide FFN.

#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qvl {

/// A dequantised F32 weight, held in ggml memory order (ne[0] fastest).
struct RawWeight {
    std::string name;
    std::vector<int64_t> ne;   ///< ggml ne order, always 4 entries
    std::vector<float> data;   ///< ne[0]*ne[1]*ne[2]*ne[3] floats
};

/// Owns the dequantised sidecar weights inside a ggml context + CPU buffer.
class Sidecar {
public:
    /// Opens a sidecar GGUF and dequantises every tensor to F32.
    /// @param use_cuda  When true, weights are allocated on the CUDA backend.
    explicit Sidecar(const std::string & path, bool use_cuda = false);
    ~Sidecar();

    Sidecar(const Sidecar &) = delete;
    Sidecar & operator=(const Sidecar &) = delete;

    /// Returns the weight tensor, throwing if the name is absent.
    ggml_tensor * get(const std::string & name) const;

    /// True when the sidecar carries the named tensor.
    bool has(const std::string & name) const;

    /// Number of tensors read from the file.
    int64_t tensor_count() const { return n_tensors_; }

    /// Names of every tensor in the file (file order).
    const std::vector<std::string> & names() const { return names_; }

    /// Raw (host-side) copy of a dequantised tensor, for host-side math.
    const RawWeight & raw(const std::string & name) const;

private:
    void load(const std::string & path);

    gguf_context * ctx_ = nullptr;
    ggml_context * wctx_ = nullptr;
    ggml_backend_buffer_t wbuf_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int64_t n_tensors_ = 0;
    std::vector<std::string> names_;
    std::map<std::string, ggml_tensor *> tensors_;
    std::map<std::string, RawWeight> raw_;
};

} // namespace qvl
