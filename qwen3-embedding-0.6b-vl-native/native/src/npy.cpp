#include "npy.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace qvl {
namespace {

/// Builds the `.npy` header dict, padded so that the total header size is a
/// multiple of 64 bytes (the format's alignment requirement).
std::string npy_header(const char * descr, const std::vector<int64_t> & shape) {
    std::string dict = "{'descr': '";
    dict += descr;
    dict += "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        dict += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            dict += ", ";
        }
    }
    if (shape.size() == 1) {
        // A 1-element tuple needs the trailing comma to stay a tuple.
        dict += ",";
    }
    dict += "), }";

    // 6 (magic) + 2 (version) + 2 (length) + header must be a multiple of 64.
    size_t total = 10 + dict.size() + 1;
    const size_t pad = (64 - (total % 64)) % 64;
    dict.append(pad, ' ');
    dict += '\n';
    return dict;
}

void write_npy(const std::string & path, const char * descr, const std::vector<int64_t> & shape,
               const void * data, size_t item_size) {
    const std::string header = npy_header(descr, shape);
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        throw std::runtime_error("could not open " + path + " for writing");
    }
    const unsigned char magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
    const unsigned char version[2] = { 0x01, 0x00 };
    const uint16_t hlen = static_cast<uint16_t>(header.size());
    std::fwrite(magic, 1, sizeof(magic), f);
    std::fwrite(version, 1, sizeof(version), f);
    std::fwrite(&hlen, sizeof(hlen), 1, f);
    std::fwrite(header.data(), 1, header.size(), f);

    size_t count = 1;
    for (int64_t d : shape) {
        count *= static_cast<size_t>(d);
    }
    std::fwrite(data, item_size, count, f);
    std::fclose(f);
}

} // namespace

void write_npy_f32(const std::string & path, const std::vector<int64_t> & shape, const float * data) {
    write_npy(path, "<f4", shape, data, sizeof(float));
}

void write_npy_i32(const std::string & path, const std::vector<int64_t> & shape, const int32_t * data) {
    write_npy(path, "<i4", shape, data, sizeof(int32_t));
}

} // namespace qvl
