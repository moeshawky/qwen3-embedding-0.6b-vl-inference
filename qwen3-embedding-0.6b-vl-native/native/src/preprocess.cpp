#include "preprocess.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace qvl {
namespace {

/// One-dimensional antialias filter, as used by
/// `torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False,
/// antialias=True)`.
///
/// The rule (derived from, and verified against, PyTorch's one-hot probes):
///   s        = in_size / out_size                 (scale factor)
///   support  = max(s, 1)                          (filter half-width, input px)
///   center_j = (j + 0.5) * s - 0.5                (align_corners=False)
///   w_i      = max(0, 1 - |i - center_j| / support), renormalised over the
///              taps that fall inside [0, in_size)
/// For s <= 1 (upsampling) the filter degenerates to the classic 2-tap
/// bilinear kernel; for s > 1 (downsampling) it is the classic triangle
/// (box-in-box) antialias kernel.
struct AaFilter {
    std::vector<std::vector<int>> idx;  ///< per output: contributing source indices
    std::vector<std::vector<double>> w; ///< per output: normalised weights
};

AaFilter aa_filter(int in_size, int out_size) {
    const double s = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double support = s > 1.0 ? s : 1.0;
    AaFilter f;
    f.idx.resize(static_cast<size_t>(out_size));
    f.w.resize(static_cast<size_t>(out_size));
    for (int j = 0; j < out_size; ++j) {
        const double center = (static_cast<double>(j) + 0.5) * s - 0.5;
        int lo = static_cast<int>(std::ceil(center - support));
        int hi = static_cast<int>(std::floor(center + support));
        if (lo < 0) lo = 0;
        if (hi > in_size - 1) hi = in_size - 1;
        std::vector<double> ws;
        double sum = 0.0;
        for (int i = lo; i <= hi; ++i) {
            double w = 1.0 - std::fabs(static_cast<double>(i) - center) / support;
            if (w < 0.0) w = 0.0;
            ws.push_back(w);
            sum += w;
        }
        if (sum <= 0.0) {
            const int nearest = std::min(std::max(static_cast<int>(std::lround(center)), 0), in_size - 1);
            f.idx[static_cast<size_t>(j)].push_back(nearest);
            f.w[static_cast<size_t>(j)].push_back(1.0);
            continue;
        }
        for (size_t k = 0; k < ws.size(); ++k) {
            f.idx[static_cast<size_t>(j)].push_back(lo + static_cast<int>(k));
            f.w[static_cast<size_t>(j)].push_back(ws[k] / sum);
        }
    }
    return f;
}

/// `get_scaled_image_size` from transformers' SigLIP2 image processor.
int scaled_image_size(double scale, int size, int patch_size) {
    const double scaled = static_cast<double>(size) * scale;
    const double ceiled = std::ceil(scaled / static_cast<double>(patch_size)) * static_cast<double>(patch_size);
    const int v = static_cast<int>(ceiled);
    return v > patch_size ? v : patch_size;
}

} // namespace

ImageU8 load_image(const std::string & path) {
    int w = 0, h = 0, comp = 0;
    unsigned char * data = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (data == nullptr) {
        throw std::runtime_error("could not decode image " + path + ": " +
                                 (stbi_failure_reason() ? stbi_failure_reason() : "unknown error"));
    }
    ImageU8 img;
    img.width = w;
    img.height = h;
    img.data.assign(data, data + static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    stbi_image_free(data);
    return img;
}

void compute_target_size(int height, int width, int patch_size, int max_num_patches,
                         int & out_height, int & out_width) {
    // Port of transformers' `get_image_size_for_max_num_patches` (eps = 1e-5).
    const double eps = 1e-5;
    double scale_min = eps / 10.0;
    double scale_max = 100.0;
    while ((scale_max - scale_min) >= eps) {
        const double scale = (scale_min + scale_max) / 2.0;
        const int th = scaled_image_size(scale, height, patch_size);
        const int tw = scaled_image_size(scale, width, patch_size);
        const double num_patches = (static_cast<double>(th) / patch_size) *
                                   (static_cast<double>(tw) / patch_size);
        if (num_patches <= static_cast<double>(max_num_patches)) {
            scale_min = scale;
        } else {
            scale_max = scale;
        }
    }
    const double scale = scale_min;
    out_height = scaled_image_size(scale, height, patch_size);
    out_width = scaled_image_size(scale, width, patch_size);
}

std::vector<uint8_t> resize_bilinear_aa(const ImageU8 & src, int out_height, int out_width) {
    const int in_h = src.height;
    const int in_w = src.width;
    if (in_h == out_height && in_w == out_width) {
        return src.data;
    }

    const AaFilter fx = aa_filter(in_w, out_width);
    const AaFilter fy = aa_filter(in_h, out_height);

    // Horizontal pass. Torchvision resizes 8-bit images in two passes and
    // rounds to 8 bit AFTER EACH PASS; reproducing that rounding (rather than
    // rounding once at the end) is what makes the native output bit-identical
    // to the HF processor.
    std::vector<uint8_t> tmp(static_cast<size_t>(in_h) * out_width * 3, 0);
    for (int y = 0; y < in_h; ++y) {
        for (int x = 0; x < out_width; ++x) {
            double acc[3] = {0.0, 0.0, 0.0};
            const auto & idx = fx.idx[static_cast<size_t>(x)];
            const auto & wts = fx.w[static_cast<size_t>(x)];
            for (size_t t = 0; t < idx.size(); ++t) {
                const uint8_t * p = src.pixel(y, idx[t]);
                const double w = wts[t];
                acc[0] += w * p[0];
                acc[1] += w * p[1];
                acc[2] += w * p[2];
            }
            uint8_t * dst = tmp.data() + (static_cast<size_t>(y) * out_width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const double rounded = std::floor(acc[c] + 0.5);
                dst[c] = static_cast<uint8_t>(rounded < 0.0 ? 0.0 : (rounded > 255.0 ? 255.0 : rounded));
            }
        }
    }

    // Vertical pass, then round to 8 bit.
    std::vector<uint8_t> out(static_cast<size_t>(out_height) * out_width * 3, 0);
    for (int y = 0; y < out_height; ++y) {
        const auto & idx = fy.idx[static_cast<size_t>(y)];
        const auto & wts = fy.w[static_cast<size_t>(y)];
        for (int x = 0; x < out_width; ++x) {
            double acc[3] = {0.0, 0.0, 0.0};
            for (size_t t = 0; t < idx.size(); ++t) {
                const uint8_t * p = tmp.data() + (static_cast<size_t>(idx[t]) * out_width + x) * 3;
                const double w = wts[t];
                acc[0] += w * p[0];
                acc[1] += w * p[1];
                acc[2] += w * p[2];
            }
            uint8_t * dst = out.data() + (static_cast<size_t>(y) * out_width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const double rounded = std::floor(acc[c] + 0.5);
                dst[c] = static_cast<uint8_t>(rounded < 0.0 ? 0.0 : (rounded > 255.0 ? 255.0 : rounded));
            }
        }
    }
    return out;
}

std::vector<float> patchify_normalize(const std::vector<uint8_t> & resized,
                                       int height, int width, int patch_size, int max_patches,
                                       std::vector<int32_t> & mask,
                                       int & out_grid_h, int & out_grid_w) {
    out_grid_h = height / patch_size;
    out_grid_w = width / patch_size;
    const int patch_dim = 3 * patch_size * patch_size;

    std::vector<float> values(static_cast<size_t>(max_patches) * patch_dim, 0.0f);
    mask.assign(static_cast<size_t>(max_patches), 0);

    for (int py = 0; py < out_grid_h; ++py) {
        for (int px = 0; px < out_grid_w; ++px) {
            const int p = py * out_grid_w + px;
            float * dst = values.data() + static_cast<size_t>(p) * patch_dim;
            size_t k = 0;
            for (int iy = 0; iy < patch_size; ++iy) {
                const uint8_t * row = resized.data() +
                    (static_cast<size_t>(py * patch_size + iy) * width + px * patch_size) * 3;
                for (int ix = 0; ix < patch_size; ++ix) {
                    for (int c = 0; c < 3; ++c) {
                        const float x = static_cast<float>(row[ix * 3 + c]) / 255.0f;
                        dst[k++] = (x - 0.5f) / 0.5f;
                    }
                }
            }
            mask[static_cast<size_t>(p)] = 1;
        }
    }
    return values;
}

std::vector<float> interpolate_positions(const std::vector<float> & pos_table, int embed_dim,
                                         int grid_h, int grid_w, int max_pos) {
    const int src = static_cast<int>(std::lround(std::sqrt(static_cast<double>(max_pos))));
    if (src * src != max_pos) {
        throw std::runtime_error("position table must be square");
    }
    const AaFilter fx = aa_filter(src, grid_w);
    const AaFilter fy = aa_filter(src, grid_h);

    const int n_out = grid_h * grid_w;
    std::vector<float> out(static_cast<size_t>(max_pos) * embed_dim, 0.0f);

    // Horizontal pass of the (src x src) position grid, per embedding dim.
    std::vector<double> horiz(static_cast<size_t>(src) * grid_w * embed_dim, 0.0);
    for (int y = 0; y < src; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            double * acc = horiz.data() + (static_cast<size_t>(y) * grid_w + x) * embed_dim;
            const auto & idx = fx.idx[static_cast<size_t>(x)];
            const auto & wts = fx.w[static_cast<size_t>(x)];
            for (size_t t = 0; t < idx.size(); ++t) {
                const float * src_row = pos_table.data() +
                    (static_cast<size_t>(y) * src + idx[t]) * embed_dim;
                const double w = wts[t];
                for (int c = 0; c < embed_dim; ++c) {
                    acc[c] += w * src_row[c];
                }
            }
        }
    }

    // Vertical pass.
    for (int y = 0; y < grid_h; ++y) {
        const auto & idx = fy.idx[static_cast<size_t>(y)];
        const auto & wts = fy.w[static_cast<size_t>(y)];
        for (int x = 0; x < grid_w; ++x) {
            float * dst = out.data() + (static_cast<size_t>(y * grid_w + x) * embed_dim);
            for (size_t t = 0; t < idx.size(); ++t) {
                const double * src_row = horiz.data() +
                    (static_cast<size_t>(idx[t]) * grid_w + x) * embed_dim;
                const double w = wts[t];
                for (int c = 0; c < embed_dim; ++c) {
                    dst[c] += static_cast<float>(w * src_row[c]);
                }
            }
        }
    }

    // Padded slots repeat the first position (HF `resize_positional_embeddings`).
    for (int p = n_out; p < max_pos; ++p) {
        std::memcpy(out.data() + static_cast<size_t>(p) * embed_dim,
                    out.data(), static_cast<size_t>(embed_dim) * sizeof(float));
    }
    return out;
}

} // namespace qvl
