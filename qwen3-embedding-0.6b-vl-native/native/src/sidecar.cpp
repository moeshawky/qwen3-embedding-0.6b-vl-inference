#include "sidecar.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace qvl {
namespace {

/// Number of elements per Q8_0 block (32 int8 values + one fp16 scale).
constexpr int64_t QK8_0 = 32;

/// Bytes per Q8_0 block on disk: 2 (fp16 scale) + 32 (int8 values).
constexpr size_t Q8_0_BLOCK_BYTES = 2 + QK8_0;

/// True when a tensor's ggml ne[0] was padded with zero columns by the
/// exporter; `keep` receives the number of kept columns (the true in_features).
bool has_trim(const std::string & name, int64_t & keep) {
    if (name.rfind("v.blk.", 0) == 0) {
        if (name.find(".ffn_down.weight") != std::string::npos) {
            keep = 4304;
            return true;
        }
    }
    if (name == "p.mlp.fc2.weight") {
        keep = 4304;
        return true;
    }
    return false;
}

/// Dequantise a Q8_0 block row into F32 (ggml's on-disk Q8_0 layout:
/// one fp16 scale followed by 32 int8 values, per 32-element block).
void dequant_q8_0(const uint8_t * src, float * dst, int64_t n) {
    const int64_t n_blocks = n / QK8_0;
    for (int64_t b = 0; b < n_blocks; ++b) {
        ggml_fp16_t h;
        std::memcpy(&h, src + b * Q8_0_BLOCK_BYTES, sizeof(h));
        const float d = ggml_fp16_to_fp32(h);
        const int8_t * q = reinterpret_cast<const int8_t *>(src + b * Q8_0_BLOCK_BYTES + 2);
        for (int64_t j = 0; j < QK8_0; ++j) {
            dst[b * QK8_0 + j] = d * static_cast<float>(q[j]);
        }
    }
}

/// Read-only memory map of the sidecar file, so the tensor payloads can be
/// addressed without copying the whole 458 MB file into the heap.
class MappedFile {
public:
    explicit MappedFile(const std::string & path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("could not open " + path);
        }
        struct stat st;
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("could not stat " + path);
        }
        size_ = static_cast<size_t>(st.st_size);
        void * addr = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (addr == MAP_FAILED) {
            throw std::runtime_error("could not mmap " + path);
        }
        data_ = static_cast<const uint8_t *>(addr);
    }

    ~MappedFile() {
        if (data_ != nullptr) {
            ::munmap(const_cast<uint8_t *>(data_), size_);
        }
    }

    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;

    const uint8_t * data() const { return data_; }

private:
    const uint8_t * data_ = nullptr;
    size_t size_ = 0;
};

} // namespace

Sidecar::Sidecar(const std::string & path, bool use_cuda) {
    if (use_cuda) {
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
    load(path);
}

Sidecar::~Sidecar() {
    if (wbuf_ != nullptr) {
        ggml_backend_buffer_free(wbuf_);
    }
    if (wctx_ != nullptr) {
        ggml_free(wctx_);
    }
    if (ctx_ != nullptr) {
        gguf_free(ctx_);
    }
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

void Sidecar::load(const std::string & path) {
    // Map the file first: the GGUF header is parsed from the same bytes.
    MappedFile file(path);

    struct gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (ctx_ == nullptr) {
        throw std::runtime_error("gguf: failed to open sidecar " + path);
    }

    // The tensor payload blob starts after the metadata, aligned.
    const uint8_t * blob = file.data() + gguf_get_data_offset(ctx_);

    n_tensors_ = gguf_get_n_tensors(ctx_);
    names_.reserve(static_cast<size_t>(n_tensors_));

    struct Pending {
        std::string name;
        int64_t ne[4];
        ggml_type type;
        const void * data;
    };
    std::vector<Pending> pending;
    pending.reserve(static_cast<size_t>(n_tensors_));

    for (int64_t i = 0; i < n_tensors_; ++i) {
        const int64_t * ne = gguf_get_tensor_ne(ctx_, i);
        const ggml_type type = gguf_get_tensor_type(ctx_, i);
        if (type != GGML_TYPE_F32 && type != GGML_TYPE_Q8_0) {
            throw std::runtime_error(std::string("gguf: unsupported tensor type in sidecar for ") +
                                     gguf_get_tensor_name(ctx_, i));
        }
        Pending p;
        p.name = gguf_get_tensor_name(ctx_, i);
        for (int d = 0; d < 4; ++d) {
            p.ne[d] = ne[d];
        }
        p.type = type;
        p.data = blob + gguf_get_tensor_offset(ctx_, i);
        names_.push_back(p.name);
        pending.push_back(p);
    }

    // Weights live in their own context so they outlive the per-run graphs.
    // The context is metadata-only (no_alloc), so a small reservation is
    // enough; the actual bytes live in the backend buffer.
    struct ggml_init_params wparams = {
        /*mem_size =*/ 64u * 1024u * 1024u,
        /*mem_buffer =*/ nullptr,
        /*no_alloc =*/ true,
    };
    wctx_ = ggml_init(wparams);
    if (wctx_ == nullptr) {
        throw std::runtime_error("ggml: failed to create the weight context");
    }

    for (const Pending & p : pending) {
        int64_t keep = p.ne[0];
        has_trim(p.name, keep);

        ggml_tensor * t = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, keep, p.ne[1]);
        if (t == nullptr) {
            throw std::runtime_error("ggml: out of memory creating weight " + p.name);
        }
        ggml_set_name(t, p.name.c_str());

        RawWeight rw;
        rw.name = p.name;
        rw.ne = { keep, p.ne[1], 1, 1 };
        rw.data.assign(static_cast<size_t>(keep) * static_cast<size_t>(p.ne[1]), 0.0f);

        const int64_t stride = p.ne[0];
        const int64_t rows = p.ne[1];
        const size_t bytes_per_row = (p.type == GGML_TYPE_F32)
            ? static_cast<size_t>(stride) * sizeof(float)
            : static_cast<size_t>(stride / QK8_0) * Q8_0_BLOCK_BYTES;
        const uint8_t * base = reinterpret_cast<const uint8_t *>(p.data);
        std::vector<float> tmp;
        if (p.type == GGML_TYPE_Q8_0) {
            tmp.resize(static_cast<size_t>(stride));
        }
        for (int64_t r = 0; r < rows; ++r) {
            const uint8_t * src_row = base + static_cast<size_t>(r) * bytes_per_row;
            float * dst_row = rw.data.data() + r * keep;
            if (p.type == GGML_TYPE_F32) {
                std::memcpy(dst_row, src_row, static_cast<size_t>(keep) * sizeof(float));
            } else {
                // Q8_0 rows are contiguous blocks of 32 values each.
                dequant_q8_0(src_row, tmp.data(), stride);
                std::memcpy(dst_row, tmp.data(), static_cast<size_t>(keep) * sizeof(float));
            }
        }

        raw_[p.name] = std::move(rw);
        tensors_[p.name] = t;
    }

    // Derived tensors: the packed attention in-projection is split into its
    // q, k and v halves so the probe can be projected separately from the
    // token sequence.
    const int64_t n_embd = 1152;
    for (int part = 0; part < 3; ++part) {
        const char * suffix = part == 0 ? ".q" : (part == 1 ? ".k" : ".v");
        const int64_t row0 = part * n_embd;

        const RawWeight & w = raw("p.attention.in_proj_weight");
        ggml_tensor * t = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, w.ne[0], n_embd);
        ggml_set_name(t, (std::string("p.attention.in_proj_weight") + suffix).c_str());
        std::vector<float> data(static_cast<size_t>(w.ne[0]) * static_cast<size_t>(n_embd));
        for (int64_t r = 0; r < n_embd; ++r) {
            std::memcpy(data.data() + static_cast<size_t>(r) * static_cast<size_t>(w.ne[0]),
                        w.data.data() + static_cast<size_t>(row0 + r) * static_cast<size_t>(w.ne[0]),
                        static_cast<size_t>(w.ne[0]) * sizeof(float));
        }
        const std::string wname = std::string("p.attention.in_proj_weight") + suffix;
        raw_[wname] = RawWeight{ wname, { w.ne[0], n_embd, 1, 1 }, data };
        tensors_[wname] = t;

        const RawWeight & b = raw("p.attention.in_proj_bias");
        ggml_tensor * tb = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, 1);
        const std::string bname = std::string("p.attention.in_proj_bias") + suffix;
        ggml_set_name(tb, bname.c_str());
        std::vector<float> bdata(b.data.begin() + row0, b.data.begin() + row0 + n_embd);
        raw_[bname] = RawWeight{ bname, { n_embd, 1, 1, 1 }, bdata };
        tensors_[bname] = tb;
    }

    // Allocate and upload all weights.
    wbuf_ = ggml_backend_alloc_ctx_tensors(wctx_, backend_);
    if (wbuf_ == nullptr) {
        throw std::runtime_error("ggml: failed to allocate the weight buffer");
    }

    for (auto & kv : tensors_) {
        const RawWeight & rw = raw_.at(kv.first);
        ggml_backend_tensor_set(kv.second, rw.data.data(), 0, rw.data.size() * sizeof(float));
    }
}

ggml_tensor * Sidecar::get(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("sidecar: missing tensor " + name);
    }
    return it->second;
}

bool Sidecar::has(const std::string & name) const {
    return tensors_.find(name) != tensors_.end();
}

const RawWeight & Sidecar::raw(const std::string & name) const {
    auto it = raw_.find(name);
    if (it == raw_.end()) {
        throw std::runtime_error("sidecar: missing tensor " + name);
    }
    return it->second;
}

} // namespace qvl
