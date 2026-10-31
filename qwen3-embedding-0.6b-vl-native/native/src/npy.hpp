#pragma once
// Minimal NumPy `.npy` writer (float32, C order) so the native binary can
// emit artifacts that the Python comparator reads without any conversion.

#include <cstdint>
#include <string>
#include <vector>

namespace qvl {

/// Writes `data` as a float32 `.npy` file with the given shape.
void write_npy_f32(const std::string & path, const std::vector<int64_t> & shape, const float * data);

/// Writes `data` as an int32 `.npy` file with the given shape.
void write_npy_i32(const std::string & path, const std::vector<int64_t> & shape, const int32_t * data);

} // namespace qvl
