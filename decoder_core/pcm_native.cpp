// Optional I_PCM byte serializer. Python supplies the authoritative slice
// header/parameter sets and checks the completed access unit's level limits.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
bool add(std::size_t a, std::size_t b, std::size_t& result) {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    result = a + b;
    return true;
}

bool multiply(std::size_t a, std::size_t b, std::size_t& result) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) return false;
    result = a * b;
    return true;
}

struct Escaper {
    std::uint8_t* output;
    unsigned zeros = 0;

    void put(std::uint8_t value) {
        if (zeros == 2 && value <= 3) {
            *output++ = 3;
            zeros = 0;
        }
        *output++ = value;
        zeros = value == 0 ? zeros + 1 : 0;
    }
};
}  // namespace

extern "C" std::uint32_t pcm_abi_version() noexcept { return 1; }

// Returns 0 on success, 1 for invalid arguments, 2 for size overflow, or 3
// for insufficient output capacity. No input/output byte is accessed until
// every size and pointer has been checked. Caller must provide real buffers
// of the declared sizes; input, templates and output must not overlap.
extern "C" int pcm_encode(
    const std::uint8_t* plane, std::size_t plane_size,
    std::uint32_t width, std::uint32_t height,
    const std::uint8_t* header, std::size_t header_size,
    const std::uint8_t* prefix, std::size_t prefix_size,
    std::uint8_t nal_header, std::uint8_t* output, std::size_t capacity,
    std::size_t* output_size) noexcept {
    if (!output_size) return 1;
    *output_size = 0;
    if (!plane || !header || !header_size || (!prefix && prefix_size) || !output ||
        width < 16 || height < 16 || width % 16 || height % 16 ||
        (nal_header != 0x65 && nal_header != 0x61)) return 1;
    std::size_t pixels, macroblocks, raw_size, required;
    const std::size_t columns = static_cast<std::size_t>(width) / 16 + 2;
    const std::size_t rows = static_cast<std::size_t>(height) / 16 + 2;
    if (!multiply(width, height, pixels) || !multiply(columns, rows, macroblocks) ||
        !multiply(macroblocks, 386, raw_size) || !add(raw_size, header_size - 1, raw_size)) return 2;
    if (plane_size != pixels) return 1;
    // Each MB contributes 384 PCM bytes and all but the first contribute
    // two syntax bytes. The slice ends with one RBSP trailing byte.
    if (!add(raw_size, raw_size / 2, required) || !add(required, 5, required) ||
        !add(required, prefix_size, required)) return 2;
    if (capacity < required) {
        *output_size = required;
        return 3;
    }
    if (prefix_size) std::memcpy(output, prefix, prefix_size);
    auto* next = output + prefix_size;
    *next++ = 0; *next++ = 0; *next++ = 0; *next++ = 1; *next++ = nal_header;
    Escaper escaped{next};
    for (std::size_t index = 0; index < header_size; ++index) escaped.put(header[index]);
    for (std::size_t my = 0; my < rows; ++my) {
        for (std::size_t mx = 0; mx < columns; ++mx) {
            if (my || mx) { escaped.put(0x0d); escaped.put(0); }
            if (my == 0 || mx == 0 || my + 1 == rows || mx + 1 == columns) {
                for (unsigned index = 0; index < 256; ++index) escaped.put(0);
            } else {
                const auto* source = plane + ((my - 1) * 16) * width + (mx - 1) * 16;
                for (unsigned y = 0; y < 16; ++y) {
                    const auto* source_row = source + y * static_cast<std::size_t>(width);
                    for (unsigned x = 0; x < 16; ++x) escaped.put(source_row[x]);
                }
            }
            for (unsigned index = 0; index < 128; ++index) escaped.put(128);
        }
    }
    escaped.put(128);
    *output_size = static_cast<std::size_t>(escaped.output - output);
    return 0;
}
