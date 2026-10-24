// Exact active-image CPU baseline. Fixed 16x16 fields; zero extension.
// Contexts own reusable scratch and are single-threaded. No pixel-result cache.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int tile = 16;
constexpr int halo = 32;
thread_local std::string last_error;

inline int horizontal(const uint8_t* p) {
    return int(p[-2]) - 5 * int(p[-1]) + 20 * int(p[0]) +
           20 * int(p[1]) - 5 * int(p[2]) + int(p[3]);
}
inline int vertical(const uint8_t* p, int stride) {
    return int(p[-2 * stride]) - 5 * int(p[-stride]) + 20 * int(p[0]) +
           20 * int(p[stride]) - 5 * int(p[2 * stride]) + int(p[3 * stride]);
}
template<int shift> inline uint8_t rounded_clip(int value) {
    // Negative pre-rounded values always clip to zero, so do not rely on
    // implementation-defined right shifts of negative signed integers.
    const int rounded = value <= 0 ? 0 : (value + (1 << (shift - 1))) >> shift;
    return static_cast<uint8_t>(std::min(255, rounded));
}
inline uint8_t average(uint8_t a, uint8_t b) {
    return static_cast<uint8_t>((unsigned(a) + unsigned(b) + 1) >> 1);
}

// Each specialization has a fixed phase: no phase switches or pixel-boundary
// tests inside the vectorizable x loop. Diagonals retain signed six-tap results
// in int16_t (range -2550..10710) until their int32_t vertical pass.
template<int fx, int fy>
void fractional_block(const uint8_t* source, int stride, uint8_t* out, int16_t* raw) {
    constexpr bool diagonal = (fx == 2 && fy != 0) || (fy == 2 && fx != 0);
    if constexpr (diagonal) {
        for (int y = -2; y < tile + 3; ++y) {
            const auto* row = source + y * stride;
            auto* target = raw + (y + 2) * tile;
            for (int x = 0; x < tile; ++x)
                target[x] = static_cast<int16_t>(horizontal(row + x));
        }
    }
    for (int y = 0; y < tile; ++y) {
        const auto* row = source + y * stride;
        auto* target = out + y * tile;
        if constexpr (fx == 0 && fy == 0) {
            std::memcpy(target, row, tile);
        } else {
            for (int x = 0; x < tile; ++x) {
                const auto* p = row + x;
                uint8_t d = 0;
                if constexpr (diagonal) {
                    const auto* h = raw + y * tile + x;
                    d = rounded_clip<10>(int(h[0]) - 5 * int(h[tile]) + 20 * int(h[2 * tile]) +
                                         20 * int(h[3 * tile]) - 5 * int(h[4 * tile]) + int(h[5 * tile]));
                }
                // H.264 quarter positions use already-clipped half samples;
                // only the diagonal's two six-tap passes share raw precision.
                if constexpr (fx == 2 && fy == 2) target[x] = d;
                else if constexpr (fx == 2 && fy == 0) target[x] = rounded_clip<5>(horizontal(p));
                else if constexpr (fx == 0 && fy == 2) target[x] = rounded_clip<5>(vertical(p, stride));
                else if constexpr ((fx & 1) && (fy & 1)) {
                    const auto h = rounded_clip<5>(horizontal(p + (fy == 3 ? stride : 0)));
                    const auto v = rounded_clip<5>(vertical(p + (fx == 3 ? 1 : 0), stride));
                    target[x] = average(h, v);
                } else if constexpr (fy == 0) {
                    target[x] = average(rounded_clip<5>(horizontal(p)), p[fx == 3 ? 1 : 0]);
                } else if constexpr (fx == 0) {
                    target[x] = average(rounded_clip<5>(vertical(p, stride)), p[fy == 3 ? stride : 0]);
                } else if constexpr (fy == 2) {
                    target[x] = average(d, rounded_clip<5>(vertical(p + (fx == 3 ? 1 : 0), stride)));
                } else { // fx == 2, fy == 1 or 3
                    target[x] = average(d, rounded_clip<5>(horizontal(p + (fy == 3 ? stride : 0))));
                }
            }
        }
    }
}
using BlockFunction = void (*)(const uint8_t*, int, uint8_t*, int16_t*);
constexpr BlockFunction phases[] = {
    fractional_block<0,0>, fractional_block<1,0>, fractional_block<2,0>, fractional_block<3,0>,
    fractional_block<0,1>, fractional_block<1,1>, fractional_block<2,1>, fractional_block<3,1>,
    fractional_block<0,2>, fractional_block<1,2>, fractional_block<2,2>, fractional_block<3,2>,
    fractional_block<0,3>, fractional_block<1,3>, fractional_block<2,3>, fractional_block<3,3>
};
struct Read {
    size_t offset = 0;
    BlockFunction function = phases[0];
    bool zero = true;
};
struct Block {
    size_t destination;
    Read a, b;
};

class Context {
public:
    int size, kind, rounds, stride;
    size_t pixels;
    std::vector<Block> blocks;
    std::vector<uint8_t> padded_a, padded_b, current_a, current_b;
    alignas(64) std::array<uint8_t, tile * tile> block_a{}, block_b{};
    alignas(64) std::array<int16_t, (tile + 5) * tile> horizontal_scratch{};

    Read plan(int bx, int by, const int32_t* field, size_t index, bool quarter) const {
        int64_t dx = field[2 * index], dy = field[2 * index + 1];
        int fx = 0, fy = 0;
        if (quarter) {
            // C++ division truncates; H.264 coordinates use floor division.
            fx = static_cast<int>((dx % 4 + 4) % 4);
            fy = static_cast<int>((dy % 4 + 4) % 4);
            dx = (dx - fx) / 4;
            dy = (dy - fy) / 4;
        }
        const int64_t x = bx + dx, y = by + dy;
        Read read;
        read.function = phases[fy * 4 + fx];
        // Conservative six-tap support for every phase. Anything wholly outside
        // is constant zero. Remaining source rectangles fit the fixed halo.
        if (x + tile + 2 < 0 || y + tile + 2 < 0 || x - 2 >= size || y - 2 >= size)
            return read;
        read.zero = false;
        read.offset = static_cast<size_t>(y + halo) * stride + static_cast<size_t>(x + halo);
        return read;
    }

    Context(int side, int operation, int iterations, const int32_t* a, const int32_t* b)
        : size(side), kind(operation), rounds(iterations), stride(0), pixels(0) {
        if (size < tile || size > 4096 || size % tile)
            throw std::invalid_argument("size must be a multiple of 16 in 16..4096");
        stride = size + 2 * halo;
        pixels = static_cast<size_t>(size) * size;
        if (kind < 0 || kind > 4) throw std::invalid_argument("kind must be 0..4");
        if (kind == 4 && (rounds < 1 || rounds > 4096))
            throw std::invalid_argument("iterative rounds must be 1..4096");
        if (kind >= 2 && !a) throw std::invalid_argument("field_a is required");
        if (kind >= 3 && !b) throw std::invalid_argument("field_b is required");
        if (kind >= 2) {
            padded_a.resize(static_cast<size_t>(stride) * stride, 0);
            if (kind == 3) padded_b.resize(static_cast<size_t>(stride) * stride, 0);
            if (kind == 4) { current_a.resize(pixels); current_b.resize(pixels); }
            blocks.reserve(pixels / (tile * tile));
            for (int y = 0; y < size; y += tile) {
                for (int x = 0; x < size; x += tile) {
                    const size_t index = blocks.size();
                    Block block{static_cast<size_t>(y) * size + x, plan(x, y, a, index, kind >= 3), {}};
                    if (kind >= 3) block.b = plan(x, y, b, index, true);
                    blocks.push_back(block);
                }
            }
        }
    }

    void pad(const uint8_t* input, std::vector<uint8_t>& output) {
        // Border bytes are initialized once and never modified. Every active
        // byte is replaced on every call/round; no previous result is reused.
        for (int y = 0; y < size; ++y)
            std::memcpy(output.data() + (y + halo) * stride + halo, input + y * size, size);
    }

    void warp(const Read& read, const uint8_t* padded, uint8_t* out) {
        if (read.zero) std::memset(out, 0, tile * tile);
        else read.function(padded + read.offset, stride, out, horizontal_scratch.data());
    }

    void blend(const uint8_t* a, const uint8_t* b, uint8_t* out) {
        for (const auto& block : blocks) {
            warp(block.a, a, block_a.data());
            warp(block.b, b, block_b.data());
            for (int y = 0; y < tile; ++y) {
                auto* destination = out + block.destination + y * size;
                const auto* pa = block_a.data() + y * tile;
                const auto* pb = block_b.data() + y * tile;
                for (int x = 0; x < tile; ++x) destination[x] = average(pa[x], pb[x]);
            }
        }
    }

    void execute(const uint8_t* a, const uint8_t* b, uint8_t* out) {
        if (!a || !out || ((kind == 1 || kind == 3) && !b))
            throw std::invalid_argument("required image pointer is null");
        if (kind == 0) { std::memmove(out, a, pixels); return; }
        if (kind == 1) {
            for (size_t i = 0; i < pixels; ++i) out[i] = average(a[i], b[i]);
            return;
        }
        if (kind == 2) {
            pad(a, padded_a);
            for (const auto& block : blocks) {
                for (int y = 0; y < tile; ++y) {
                    auto* destination = out + block.destination + y * size;
                    if (block.a.zero) std::memset(destination, 0, tile);
                    else std::memcpy(destination, padded_a.data() + block.a.offset + y * stride, tile);
                }
            }
            return;
        }
        if (kind == 3) {
            pad(a, padded_a); pad(b, padded_b);
            blend(padded_a.data(), padded_b.data(), out);
            return;
        }
        std::memcpy(current_a.data(), a, pixels);
        auto* current = current_a.data();
        auto* next = current_b.data();
        for (int i = 0; i < rounds; ++i) {
            pad(current, padded_a);
            blend(padded_a.data(), padded_a.data(), next);
            std::swap(current, next);
        }
        std::memmove(out, current, pixels);
    }
};
} // namespace

extern "C" {
// The C ABI requires size*size readable/writable image bytes, and exactly
// (size/16)^2 consecutive dx,dy int32 pairs for each required field. Array
// lengths cannot be inferred from pointers; frontends must validate lengths.
// Fields are copied into plans during create and may then be released.
void* cpu_create(int size, int kind, int rounds, const int32_t* field_a, const int32_t* field_b) {
    try {
        last_error.clear();
        return new Context(size, kind, rounds, field_a, field_b);
    } catch (const std::exception& error) { last_error = error.what(); return nullptr; }
    catch (...) { last_error = "unknown CPU context allocation failure"; return nullptr; }
}
int cpu_execute(void* context, const uint8_t* a, const uint8_t* b, uint8_t* out) {
    try {
        last_error.clear();
        if (!context) throw std::invalid_argument("context is null");
        static_cast<Context*>(context)->execute(a, b, out);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
    catch (...) { last_error = "unknown CPU execution failure"; return -1; }
}
void cpu_destroy(void* context) { delete static_cast<Context*>(context); }
const char* cpu_error() { return last_error.c_str(); }
}
