// Specialized portable CPU baseline for the fixed (1,2)/(-1,-2) recurrence.
// Each round shares signed horizontal and clipped diagonal/vertical half
// samples over the complete image. Six horizontal rows and one preceding
// blended row suffice; no full intermediate half-image needs to be stored.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

template<int shift> inline std::uint8_t rounded_clip(int value) {
    // Negative sums always clip to zero. Shift a nonnegative value so the
    // result does not depend on implementation-defined signed right shift.
    return static_cast<std::uint8_t>(std::min(255, (std::max(0, value) + (1 << (shift-1))) >> shift));
}

inline std::uint8_t average(std::uint8_t a, std::uint8_t b) {
    return static_cast<std::uint8_t>((unsigned(a)+unsigned(b)+1) >> 1);
}

class Context {
    int size, rounds, stride, raw_stride;
    std::vector<std::uint8_t> image_a, image_b, diagonal, vertical, previous;
    std::vector<std::int16_t> horizontal;

    static void horizontal_row(const std::uint8_t* row, std::int16_t* output, int width) {
        // Columns -1..size-1 are required: the preceding diagonal sample
        // D(x-1,y-1) can be nonzero even when its coordinates are outside.
        const auto* source = row-1;
        for (int x = 0; x <= width; ++x)
            output[x] = static_cast<std::int16_t>(int(source[x-2])-5*int(source[x-1])+
                          20*int(source[x])+20*int(source[x+1])-5*int(source[x+2])+int(source[x+3]));
        // Raw range -2550..10710 fits int16_t; there is no rounding/clipping.
    }

    static void diagonal_row(const std::array<std::int16_t*, 6>& rows, std::uint8_t* output, int width) {
        const auto *r0 = rows[0], *r1 = rows[1], *r2 = rows[2];
        const auto *r3 = rows[3], *r4 = rows[4], *r5 = rows[5];
        for (int x = 0; x <= width; ++x)
            output[x] = rounded_clip<10>(int(r0[x])-5*int(r1[x])+20*int(r2[x])+
                                        20*int(r3[x])-5*int(r4[x])+int(r5[x]));
    }

    static void vertical_row(const std::uint8_t* row, std::uint8_t* output, int width, int pitch) {
        for (int x = 0; x < width; ++x)
            output[x] = rounded_clip<5>(int(row[x-2*pitch])-5*int(row[x-pitch])+20*int(row[x])+
                                        20*int(row[x+pitch])-5*int(row[x+2*pitch])+int(row[x+3*pitch]));
    }

    void round(const std::uint8_t* current, std::uint8_t* next) {
        const int width = size, pitch = stride, raw_pitch = raw_stride;
        auto* raw = horizontal.data();
        auto* d = diagonal.data();
        auto* v = vertical.data();
        auto* prior = previous.data();
        std::array<std::int16_t*, 6> rows{};
        for (int i = 0; i < 6; ++i) {
            rows[i] = raw+i*raw_pitch;
            horizontal_row(current+(i-3)*pitch, rows[i], width);
        }
        // Initialize B(x,0)=avg(D(x-1,-1), V(x,-1)). Filter integer zeros
        // outside the active image; do not replace outside half samples by 0.
        diagonal_row(rows, d, width);
        vertical_row(current-pitch, v, width, pitch);
        for (int x = 0; x < width; ++x) prior[x] = average(d[x], v[x]);
        for (int y = 0; y < width; ++y) {
            auto* oldest = rows[0];
            for (int i = 0; i < 5; ++i) rows[i] = rows[i+1];
            rows[5] = oldest;
            horizontal_row(current+(y+3)*pitch, rows[5], width);
            diagonal_row(rows, d, width);
            // Separate row passes keep portable compiler alias checks small
            // enough for automatic vectorization; no ISA intrinsics are used.
            vertical_row(current+y*pitch, v, width, pitch);
            auto* output = next+y*pitch;
            for (int x = 0; x < width; ++x) {
                // A=avg(D(x,y),V(x,y)); B=avg(D(x-1,y-1),V(x,y-1)).
                // Preserve both nested half-up roundings before final average.
                output[x] = average(average(d[x+1], v[x]), prior[x]);
                prior[x] = average(d[x], v[x]);
            }
        }
    }

public:
    Context(int side, int kind, int iterations, const std::int32_t* a, const std::int32_t* b)
        : size(side), rounds(iterations), stride(0), raw_stride(0) {
        if (size < 16 || size > 4096 || size % 16)
            throw std::invalid_argument("size must be a multiple of 16 in 16..4096");
        if (kind != 4) throw std::invalid_argument("plane specialization supports only kind 4");
        if (rounds < 1 || rounds > 4096) throw std::invalid_argument("iterative rounds must be 1..4096");
        if (!a || !b) throw std::invalid_argument("both fixed motion fields are required");
        const int blocks = (size/16)*(size/16);
        for (int i = 0; i < blocks; ++i)
            if (a[2*i] != 1 || a[2*i+1] != 2 || b[2*i] != -1 || b[2*i+1] != -2)
                throw std::invalid_argument("plane specialization requires fields (1,2) and (-1,-2) in every block");
        stride = raw_stride = size+16;
        const auto padded = static_cast<std::size_t>(stride)*(size+8);
        image_a.resize(padded, 0);
        image_b.resize(padded, 0);
        horizontal.resize(static_cast<std::size_t>(raw_stride)*6);
        diagonal.resize(size+1);
        vertical.resize(size);
        previous.resize(size);
    }

    void execute(const std::uint8_t* input, std::uint8_t* output) {
        if (!input || !output) throw std::invalid_argument("required image pointer is null");
        auto* current = image_a.data()+4*stride+4;
        auto* next = image_b.data()+4*stride+4;
        // Padded borders are initialized once and never written again. Active
        // bytes and every used scratch value are replaced on every invocation.
        for (int y = 0; y < size; ++y)
            std::memcpy(current+y*stride, input+static_cast<std::size_t>(y)*size, size);
        for (int i = 0; i < rounds; ++i) {
            round(current, next);
            std::swap(current, next);
        }
        for (int y = 0; y < size; ++y)
            std::memmove(output+static_cast<std::size_t>(y)*size, current+y*stride, size);
    }
};
}  // namespace

extern "C" {
// Same ABI as cpu.cpp. Callers provide size*size input/output bytes and exactly
// (size/16)^2 signed dx,dy pairs in each field. Pointer lengths cannot be inferred.
// Contexts are single-threaded, reusable, and retain no caller-owned pointers.
void* cpu_create(int size, int kind, int rounds, const std::int32_t* a, const std::int32_t* b) {
    try {
        last_error.clear();
        return new Context(size, kind, rounds, a, b);
    } catch (const std::exception& error) { last_error = error.what(); return nullptr; }
    catch (...) { last_error = "unknown plane context allocation failure"; return nullptr; }
}
int cpu_execute(void* context, const std::uint8_t* a, const std::uint8_t*, std::uint8_t* output) {
    try {
        last_error.clear();
        if (!context) throw std::invalid_argument("context is null");
        static_cast<Context*>(context)->execute(a, output);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
    catch (...) { last_error = "unknown plane execution failure"; return -1; }
}
void cpu_destroy(void* context) { delete static_cast<Context*>(context); }
const char* cpu_error() { return last_error.c_str(); }
}
