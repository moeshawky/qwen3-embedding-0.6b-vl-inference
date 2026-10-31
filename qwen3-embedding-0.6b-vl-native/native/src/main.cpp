// qwen3vl-embed — native (GGML) image embedding for
// moeshawky/qwen3-embedding-0.6b-vl.
//
// The program loads the published Q8_0 mmproj sidecar, runs the SigLIP2 naflex
// vision tower + attention-pooling head + vision projection entirely in ggml
// (CPU), and writes the 1024-D L2-normalised image embedding plus the
// intermediate tensors needed for a layer-by-layer parity comparison.
//
// Usage:
//   qwen3vl-embed --mmproj <sidecar.gguf> --image <file.png> [--image ...]
//                 --out <dir> [--threads N]
//   qwen3vl-embed --mmproj <sidecar.gguf> --input <file.bin> --out <dir>
//
// `--input` consumes a preprocessed tensor bundle (magic "QVL1") so the graph
// can be compared against an oracle without involving the image pipeline.

#include "npy.hpp"
#include "preprocess.hpp"
#include "sidecar.hpp"
#include "text_embed.hpp"
#include "vision_graph.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string mmproj;
    std::string out_dir;
    std::vector<std::string> images;
    std::string input_bin;
    std::string text_model;
    std::string compat_head;
    std::vector<std::string> texts;
    std::string text_file;
    std::string stem = "text";
    int threads = 0;
    bool use_cuda = false;
};

void usage() {
    std::fprintf(stderr,
        "usage: qwen3vl-embed --mmproj <sidecar.gguf> (--image <file.png> | --input <file.bin>)\n"
        "                     --out <dir> [--threads N]\n");
}

/// Reads a preprocessed bundle: "QVL1", n_patches, n_dims, grid_h, grid_w,
/// float32 pixel_values[n_patches*n_dims], int32 mask[n_patches].
bool read_bundle(const std::string & path, qvl::VisionInputs & in) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "QVL1", 4) != 0) {
        std::fprintf(stderr, "error: %s is not a QVL1 bundle\n", path.c_str());
        return false;
    }
    int32_t n_patches = 0, n_dims = 0, gh = 0, gw = 0;
    f.read(reinterpret_cast<char *>(&n_patches), 4);
    f.read(reinterpret_cast<char *>(&n_dims), 4);
    f.read(reinterpret_cast<char *>(&gh), 4);
    f.read(reinterpret_cast<char *>(&gw), 4);
    if (!f) {
        return false;
    }
    in.pixel_values.resize(static_cast<size_t>(n_patches) * static_cast<size_t>(n_dims));
    f.read(reinterpret_cast<char *>(in.pixel_values.data()),
           static_cast<std::streamsize>(in.pixel_values.size() * sizeof(float)));
    in.mask.resize(static_cast<size_t>(n_patches));
    f.read(reinterpret_cast<char *>(in.mask.data()),
           static_cast<std::streamsize>(in.mask.size() * sizeof(int32_t)));
    in.grid_h = gh;
    in.grid_w = gw;
    return true;
}

void write_outputs(const std::string & dir, const std::string & stem, const qvl::VisionOutputs & o,
                   int n_tokens, int n_embd, int out_dim) {
    auto p = [&](const char * name) { return dir + "/" + stem + "_" + name + ".npy"; };
    qvl::write_npy_f32(p("patch_plus_position"), { n_tokens, n_embd }, o.patch_plus_position.data());
    qvl::write_npy_f32(p("encoder_layer_0"), { n_tokens, n_embd }, o.layer_0.data());
    qvl::write_npy_f32(p("encoder_layer_1"), { n_tokens, n_embd }, o.layer_1.data());
    qvl::write_npy_f32(p("encoder_layer_13"), { n_tokens, n_embd }, o.layer_13.data());
    qvl::write_npy_f32(p("encoder_layer_26"), { n_tokens, n_embd }, o.layer_26.data());
    qvl::write_npy_f32(p("post_layernorm"), { n_tokens, n_embd }, o.post_layernorm.data());
    qvl::write_npy_f32(p("map_pooler"), { n_embd }, o.map_pooler.data());
    qvl::write_npy_f32(p("map_pooler_norm"), { n_embd }, o.map_pooler_norm.data());
    qvl::write_npy_f32(p("linear_projection"), { out_dim }, o.linear_projection.data());
    qvl::write_npy_f32(p("image_embeds"), { out_dim }, o.image_embeds.data());
}

/// Writes the preprocessed inputs so the native preprocessing can be diffed
/// against the oracle's processor output.
void write_inputs(const std::string & dir, const std::string & stem, const qvl::VisionInputs & in,
                  int patch_dim) {
    auto p = [&](const char * name) { return dir + "/" + stem + "_" + name + ".npy"; };
    qvl::write_npy_f32(p("pixel_values"),
                       { static_cast<int64_t>(in.pixel_values.size()) / patch_dim, patch_dim },
                       in.pixel_values.data());
    qvl::write_npy_i32(p("pixel_attention_mask"),
                       { static_cast<int64_t>(in.mask.size()) }, in.mask.data());
}

} // namespace

int main(int argc, char ** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--mmproj") {
            opt.mmproj = next("--mmproj");
        } else if (a == "--out") {
            opt.out_dir = next("--out");
        } else if (a == "--image") {
            opt.images.push_back(next("--image"));
        } else if (a == "--input") {
            opt.input_bin = next("--input");
        } else if (a == "--text-model") {
            opt.text_model = next("--text-model");
        } else if (a == "--compat-head") {
            opt.compat_head = next("--compat-head");
        } else if (a == "--text") {
            opt.texts.push_back(next("--text"));
        } else if (a == "--text-file") {
            opt.text_file = next("--text-file");
        } else if (a == "--stem") {
            opt.stem = next("--stem");
        } else if (a == "--threads") {
            opt.threads = std::stoi(next("--threads"));
        } else if (a == "--cuda") {
            opt.use_cuda = true;
        } else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (opt.out_dir.empty()) {
        usage();
        return 2;
    }
    const bool want_text = !opt.text_model.empty();
    const bool want_vision = !opt.mmproj.empty() && (!opt.images.empty() || !opt.input_bin.empty());
    if (!want_text && !want_vision) {
        usage();
        return 2;
    }
    if (want_text && opt.compat_head.empty()) {
        std::fprintf(stderr, "error: --compat-head is required for the text branch\n");
        return 2;
    }

    try {
        const qvl::VisionConfig cfg;
        cfg.use_cuda = opt.use_cuda;

        // The vision branch needs the sidecar; the text branch does not.
        std::unique_ptr<qvl::Sidecar> sidecar;
        std::unique_ptr<qvl::VisionRunner> runner;
        const qvl::RawWeight * pos_w = nullptr;
        if (want_vision) {
            sidecar = std::make_unique<qvl::Sidecar>(opt.mmproj, opt.use_cuda);
            std::fprintf(stderr, "sidecar: %lld tensors loaded from %s (backend=%s)\n",
                         static_cast<long long>(sidecar->tensor_count()), opt.mmproj.c_str(),
                         opt.use_cuda ? "cuda" : "cpu");
            runner = std::make_unique<qvl::VisionRunner>(*sidecar, cfg);
            runner->set_threads(opt.threads);

            // The position table is needed on the host for the interpolation.
            const qvl::RawWeight & pos = sidecar->raw("v.position_embd.weight");
            if (pos.ne[0] != cfg.n_embd || pos.ne[1] != cfg.max_patches) {
                std::fprintf(stderr, "error: unexpected position table shape\n");
                return 3;
            }
            pos_w = &pos;
        }

        std::vector<std::string> stems;
        std::vector<qvl::VisionInputs> inputs;

        if (!opt.input_bin.empty()) {
            qvl::VisionInputs in;
            if (!read_bundle(opt.input_bin, in)) {
                return 3;
            }
            stems.push_back("input");
            inputs.push_back(std::move(in));
        }

        for (const std::string & path : opt.images) {
            const qvl::ImageU8 img = qvl::load_image(path);
            int th = 0, tw = 0;
            qvl::compute_target_size(img.height, img.width, cfg.patch_size, cfg.max_patches, th, tw);
            const std::vector<uint8_t> resized = qvl::resize_bilinear_aa(img, th, tw);
            qvl::VisionInputs in;
            std::vector<int32_t> mask;
            in.pixel_values = qvl::patchify_normalize(resized, th, tw, cfg.patch_size,
                                                       cfg.max_patches, mask, in.grid_h, in.grid_w);
            in.mask = mask;
            // Stem: the file name without directories and extension.
            std::string stem = path.substr(path.find_last_of('/') + 1);
            const size_t dot = stem.find_last_of('.');
            if (dot != std::string::npos) {
                stem = stem.substr(0, dot);
            }
            stems.push_back(stem);
            inputs.push_back(std::move(in));
        }

        for (size_t k = 0; k < inputs.size(); ++k) {
            const qvl::VisionInputs & in = inputs[k];
            const std::vector<float> positions = qvl::interpolate_positions(
                pos_w->data, cfg.n_embd, in.grid_h, in.grid_w, cfg.max_patches);

            const auto t0 = std::chrono::steady_clock::now();
            const qvl::VisionOutputs out = runner->run(in, positions);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            write_outputs(opt.out_dir, stems[k], out, cfg.max_patches, cfg.n_embd, cfg.out_dim);
            write_inputs(opt.out_dir, stems[k], in, cfg.patch_dim);

            int valid = 0;
            for (int32_t m : in.mask) {
                valid += (m != 0) ? 1 : 0;
            }
            double norm = 0.0;
            for (float v : out.image_embeds) {
                norm += static_cast<double>(v) * static_cast<double>(v);
            }
            norm = std::sqrt(norm);

            std::printf("{\"stem\": \"%s\", \"grid\": [%d, %d], \"valid_patches\": %d, "
                        "\"embedding_norm\": %.9f, \"compute_ms\": %.1f}\n",
                        stems[k].c_str(), in.grid_h, in.grid_w, valid, norm, ms);
            std::fflush(stdout);
        }
        if (want_text) {
            std::vector<std::string> prompts = opt.texts;
            if (!opt.text_file.empty()) {
                std::ifstream f(opt.text_file);
                if (!f) {
                    std::fprintf(stderr, "error: could not read %s\n", opt.text_file.c_str());
                    return 3;
                }
                std::string line;
                while (std::getline(f, line)) {
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    prompts.push_back(line);
                }
            }
            if (prompts.empty()) {
                std::fprintf(stderr, "error: no text to embed\n");
                return 2;
            }

            qvl::TextEmbedder embedder(opt.text_model, opt.compat_head, opt.threads, opt.use_cuda);
            std::vector<float> all;
            all.reserve(prompts.size() * 1024);
            for (const std::string & prompt : prompts) {
                const std::vector<float> e = embedder.embed(prompt);
                all.insert(all.end(), e.begin(), e.end());
                double norm = 0.0;
                for (float x : e) {
                    norm += static_cast<double>(x) * static_cast<double>(x);
                }
                std::printf("{\"stem\": \"%s\", \"tokens\": %d, \"embedding_norm\": %.9f}\n",
                            opt.stem.c_str(), embedder.last_token_count(), std::sqrt(norm));
                std::fflush(stdout);
            }
            qvl::write_npy_f32(opt.out_dir + "/" + opt.stem + "_text_embeds.npy",
                               { static_cast<int64_t>(prompts.size()), 1024 }, all.data());
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
