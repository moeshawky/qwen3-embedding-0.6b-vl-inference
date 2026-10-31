#pragma once
// Native (C++) preprocessing for the SigLIP2 naflex vision path.
//
// This reproduces, without PyTorch, the exact tensor pipeline that the
// HuggingFace `Siglip2ImageProcessor` + `Siglip2VisionEmbeddings` perform:
//
//   1. target size   : `get_image_size_for_max_num_patches` binary search
//   2. resize        : bilinear with antialias (torchvision semantics)
//   3. rescale+norm  : (x/255 - 0.5) / 0.5
//   4. patchify      : 16x16 patches, channel-LAST inside the patch
//   5. pad           : zero rows up to 256 patches + 0/1 attention mask
//   6. position emb. : antialiased bilinear interpolation of the 16x16
//                      position table onto the (h, w) patch grid
//
// Steps 1-5 mirror the HF image processor; step 6 mirrors
// `torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False,
// antialias=True)` applied to the 256x1152 position table.

#include <cstdint>
#include <string>
#include <vector>

namespace qvl {

/// An 8-bit RGB image, row-major, 3 channels interleaved.
struct ImageU8 {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data; ///< width*height*3 bytes

    const uint8_t * pixel(int y, int x) const { return data.data() + (static_cast<size_t>(y) * width + x) * 3; }
};

/// Decodes a PNG/JPEG file into an RGB8 image. Throws on failure.
ImageU8 load_image(const std::string & path);

/// Target pixel size chosen by `get_image_size_for_max_num_patches`.
void compute_target_size(int height, int width, int patch_size, int max_num_patches,
                         int & out_height, int & out_width);

/// The (h, w) patch grid that results from a target pixel size.
inline int grid_rows(int target_height, int patch_size) { return target_height / patch_size; }
inline int grid_cols(int target_width, int patch_size) { return target_width / patch_size; }

/// Bilinear resize with an antialias filter (torchvision semantics).
/// The result is rounded back to 8-bit, exactly like the torchvision path.
std::vector<uint8_t> resize_bilinear_aa(const ImageU8 & src, int out_height, int out_width);

/// Patchifies + normalises into `max_patches` rows of
/// `3*patch_size*patch_size` floats, row-major over the patch grid.
/// Padded rows are zero. `mask[i]` is 1 for real patches, 0 for padding.
std::vector<float> patchify_normalize(const std::vector<uint8_t> & resized,
                                       int height, int width, int patch_size, int max_patches,
                                       std::vector<int32_t> & mask,
                                       int & out_grid_h, int & out_grid_w);

/// Interpolates the 16x16 position table onto a (grid_h, grid_w) patch grid
/// using the antialiased bilinear rule, then repeats the first position for
/// the padded slots (matching `resize_positional_embeddings`).
///
/// @param pos_table 256 * embed_dim floats (row-major, position-major).
/// @param embed_dim embedding width (1152).
/// @param grid_h    patch rows of the target grid.
/// @param grid_w    patch columns of the target grid.
/// @param max_pos   number of position slots (256).
std::vector<float> interpolate_positions(const std::vector<float> & pos_table, int embed_dim,
                                         int grid_h, int grid_w, int max_pos);

} // namespace qvl
