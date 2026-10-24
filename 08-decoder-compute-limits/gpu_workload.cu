// CUDA kernels and a persistent, toolkit-free CUDA Driver API host.
// build_gpu_workload.sh emits PTX with clang, and builds this host with c++.
#ifdef GPU_KERNELS

__attribute__((device)) static unsigned index_x() {
    unsigned thread, block, size;
    asm("mov.u32 %0, %%tid.x;" : "=r"(thread));
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(block));
    asm("mov.u32 %0, %%ntid.x;" : "=r"(size));
    return block * size + thread;
}

extern "C" __attribute__((global)) void initialize(unsigned *values, unsigned count) {
    unsigned i = index_x();
    if (i < count) values[i] = i;
}

// Independent integer arithmetic chains, observable global output each unit.
extern "C" __attribute__((global)) void arithmetic(unsigned *output, unsigned count,
                                                    unsigned cycles) {
    unsigned i = index_x();
    if (i >= count) return;
    unsigned a = i + 1, b = i + 17, c = i + 101, d = i + 1009;
    for (unsigned j = 0; j < cycles; ++j) {
        a = a * 1664525u + 1013904223u;
        b = b * 22695477u + 1u;
        c = c * 1103515245u + 12345u;
        d = (d ^ (d >> 13)) * 1597334677u;
    }
    output[i] = a ^ b ^ c ^ d;
}

// Coalesced streaming read/write; buffers are swapped after each work unit.
extern "C" __attribute__((global)) void memory_stream(const unsigned *input,
                                                       unsigned *output, unsigned count) {
    unsigned i = index_x();
    if (i < count) output[i] = input[i] * 1664525u + 1013904223u;
}

// Frozen 1->8 binary 3x3 convolutions, zero outside, then binary 8->1 mix.
// Weights and thresholds are data loaded from the supplied model, never outputs.
extern "C" __attribute__((global)) void denoise(const unsigned char *input,
                                                const int *weights, unsigned char *output) {
    unsigned i = index_x();
    if (i >= 8 * 32 * 32) return;
    int pixel = i % 1024, x = pixel % 32, y = pixel / 32;
    int base = (i / 1024) * 1024, mix = 0;
    for (int channel = 0; channel < 8; ++channel) {
        int sum = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx >= 0 && xx < 32 && yy >= 0 && yy < 32)
                    sum += input[base + yy * 32 + xx] * weights[channel * 9 + (dy + 1) * 3 + dx + 1];
            }
        mix += (sum >= weights[72 + channel]) * weights[80 + channel];
    }
    output[i] = mix >= weights[88];
}

#else

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using CUdeviceptr = unsigned long long;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CUevent = void *;

struct Driver {
    void *library;
#define API(name, ...) int (*name)(__VA_ARGS__)
    API(cuInit, unsigned);
    API(cuDeviceGet, int *, int);
    API(cuCtxCreate_v2, CUcontext *, unsigned, int);
    API(cuCtxSetCurrent, CUcontext);
    API(cuCtxDestroy_v2, CUcontext);
    API(cuModuleLoad, CUmodule *, const char *);
    API(cuModuleGetFunction, CUfunction *, CUmodule, const char *);
    API(cuMemAlloc_v2, CUdeviceptr *, size_t);
    API(cuMemFree_v2, CUdeviceptr);
    API(cuMemHostAlloc, void **, size_t, unsigned);
    API(cuMemFreeHost, void *);
    API(cuMemcpyHtoD_v2, CUdeviceptr, const void *, size_t);
    API(cuMemcpyDtoH_v2, void *, CUdeviceptr, size_t);
    API(cuMemcpyHtoDAsync_v2, CUdeviceptr, const void *, size_t, CUstream);
    API(cuMemcpyDtoHAsync_v2, void *, CUdeviceptr, size_t, CUstream);
    API(cuStreamCreate, CUstream *, unsigned);
    API(cuStreamDestroy_v2, CUstream);
    API(cuStreamSynchronize, CUstream);
    API(cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned, unsigned,
        unsigned, unsigned, unsigned, CUstream, void **, void **);
    API(cuEventCreate, CUevent *, unsigned);
    API(cuEventRecord, CUevent, CUstream);
    API(cuEventSynchronize, CUevent);
    API(cuEventElapsedTime, float *, CUevent, CUevent);
    API(cuEventDestroy_v2, CUevent);
#undef API
    Driver() {
        library = dlopen("libcuda.so.1", RTLD_NOW);
        if (!library) throw std::runtime_error("CUDA driver library unavailable");
#define LOAD(name) name = reinterpret_cast<decltype(name)>(dlsym(library, #name)); \
        if (!name) throw std::runtime_error("CUDA driver entry point unavailable: " #name)
        LOAD(cuInit); LOAD(cuDeviceGet); LOAD(cuCtxCreate_v2); LOAD(cuCtxSetCurrent);
        LOAD(cuCtxDestroy_v2); LOAD(cuModuleLoad); LOAD(cuModuleGetFunction);
        LOAD(cuMemAlloc_v2); LOAD(cuMemFree_v2); LOAD(cuMemHostAlloc); LOAD(cuMemFreeHost);
        LOAD(cuMemcpyHtoD_v2); LOAD(cuMemcpyDtoH_v2); LOAD(cuMemcpyHtoDAsync_v2);
        LOAD(cuMemcpyDtoHAsync_v2); LOAD(cuStreamCreate); LOAD(cuStreamDestroy_v2);
        LOAD(cuStreamSynchronize); LOAD(cuLaunchKernel); LOAD(cuEventCreate);
        LOAD(cuEventRecord); LOAD(cuEventSynchronize); LOAD(cuEventElapsedTime);
        LOAD(cuEventDestroy_v2);
#undef LOAD
    }
};

static void checked(int result) {
    if (result) throw std::runtime_error("CUDA driver error " + std::to_string(result));
}
static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::vector<unsigned char> read_bytes(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read input file");
    return {std::istreambuf_iterator<char>(input), {}};
}
static void emit(const std::string &json) { std::cout << json << std::endl; }

struct Unit { uint64_t start_ns, end_ns; float gpu_ms; };

struct Service {
    Driver api;
    CUcontext context{};
    CUmodule module{};
    CUfunction init{}, compute{}, memory{}, inference{};
    CUstream work_stream{}, infer_stream{};
    CUdeviceptr first{}, second{}, device_input{}, device_output{}, device_weights{};
    unsigned char *host_input{}, *host_output{};
    CUevent work_begin{}, work_end{}, infer_events[4]{};
    std::thread worker;
    std::exception_ptr worker_error;
    std::vector<Unit> units;
    unsigned elements, cycles;
    bool stress_active = false;
    uint64_t stress_start{}, stress_end{};
    uint64_t checksum{};
    unsigned expected_samples[16]{};
    std::string output_path, active_mode;

    Service(const std::string &ptx, const std::string &input_path,
            const std::string &weight_path, const std::string &output,
            unsigned count, unsigned iterations) : elements(count), cycles(iterations), output_path(output) {
        if (elements < 256 || elements > 64u * 1024u * 1024u || cycles < 1 || cycles > 4096)
            throw std::runtime_error("Workload size outside bounded limits");
        auto data = read_bytes(input_path);
        if (data.size() != 8192 || std::any_of(data.begin(), data.end(), [](unsigned v) { return v > 1; }))
            throw std::runtime_error("Expected eight binary 32x32 inputs");
        std::ifstream source(weight_path);
        std::vector<int> weights;
        int value;
        while (source >> value) weights.push_back(value);
        if (weights.size() != 89 || !source.eof()) throw std::runtime_error("Expected 89 integer weights/thresholds");
        checked(api.cuInit(0));
        int device; checked(api.cuDeviceGet(&device, 0));
        // Blocking event synchronization avoids a CPU busy-wait stressor.
        checked(api.cuCtxCreate_v2(&context, 4, device));
        checked(api.cuModuleLoad(&module, ptx.c_str()));
        checked(api.cuModuleGetFunction(&init, module, "initialize"));
        checked(api.cuModuleGetFunction(&compute, module, "arithmetic"));
        checked(api.cuModuleGetFunction(&memory, module, "memory_stream"));
        checked(api.cuModuleGetFunction(&inference, module, "denoise"));
        checked(api.cuStreamCreate(&work_stream, 1));
        checked(api.cuStreamCreate(&infer_stream, 1));
        checked(api.cuMemAlloc_v2(&first, size_t(elements) * 4));
        checked(api.cuMemAlloc_v2(&second, size_t(elements) * 4));
        checked(api.cuMemAlloc_v2(&device_input, 8192));
        checked(api.cuMemAlloc_v2(&device_output, 8192));
        checked(api.cuMemAlloc_v2(&device_weights, weights.size() * sizeof(int)));
        checked(api.cuMemHostAlloc(reinterpret_cast<void **>(&host_input), 8192, 0));
        checked(api.cuMemHostAlloc(reinterpret_cast<void **>(&host_output), 8192, 0));
        std::memcpy(host_input, data.data(), 8192);
        checked(api.cuMemcpyHtoD_v2(device_weights, weights.data(), weights.size() * sizeof(int)));
        checked(api.cuEventCreate(&work_begin, 1));
        checked(api.cuEventCreate(&work_end, 1));
        for (auto &event : infer_events) checked(api.cuEventCreate(&event, 1));
    }
    ~Service() {
        if (worker.joinable()) worker.join();
        if (context) {
            api.cuCtxSetCurrent(context);
            for (auto event : infer_events) if (event) api.cuEventDestroy_v2(event);
            if (work_begin) api.cuEventDestroy_v2(work_begin);
            if (work_end) api.cuEventDestroy_v2(work_end);
            if (host_input) api.cuMemFreeHost(host_input);
            if (host_output) api.cuMemFreeHost(host_output);
            if (first) api.cuMemFree_v2(first);
            if (second) api.cuMemFree_v2(second);
            if (device_input) api.cuMemFree_v2(device_input);
            if (device_output) api.cuMemFree_v2(device_output);
            if (device_weights) api.cuMemFree_v2(device_weights);
            if (work_stream) api.cuStreamDestroy_v2(work_stream);
            if (infer_stream) api.cuStreamDestroy_v2(infer_stream);
            api.cuCtxDestroy_v2(context);
        }
    }
    void launch(CUfunction kernel, unsigned count, CUstream stream, void **args) {
        checked(api.cuLaunchKernel(kernel, (count + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr));
    }
    void infer() {
        auto start = now_ns();
        checked(api.cuEventRecord(infer_events[0], infer_stream));
        checked(api.cuMemcpyHtoDAsync_v2(device_input, host_input, 8192, infer_stream));
        checked(api.cuEventRecord(infer_events[1], infer_stream));
        void *args[] = {&device_input, &device_weights, &device_output};
        launch(inference, 8192, infer_stream, args);
        checked(api.cuEventRecord(infer_events[2], infer_stream));
        checked(api.cuMemcpyDtoHAsync_v2(host_output, device_output, 8192, infer_stream));
        checked(api.cuEventRecord(infer_events[3], infer_stream));
        checked(api.cuEventSynchronize(infer_events[3]));
        float timings[3];
        for (int i = 0; i < 3; ++i)
            checked(api.cuEventElapsedTime(&timings[i], infer_events[i], infer_events[i + 1]));
        auto completed = now_ns();
        std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<char *>(host_output), 8192);
        file.close();
        if (!file) throw std::runtime_error("Cannot write CUDA result");
        std::ostringstream out;
        out << std::setprecision(9) << "{\"event\":\"inference\",\"start_ns\":" << start
            << ",\"gpu_completed_ns\":" << completed << ",\"end_ns\":" << now_ns()
            << ",\"h2d_ms\":" << timings[0] << ",\"kernel_ms\":" << timings[1]
            << ",\"d2h_ms\":" << timings[2] << ",\"copied_bytes_each_way\":8192}";
        emit(out.str());
    }
    void start(const std::string &mode, unsigned count) {
        if (stress_active || count < 1 || count > 100000 || (mode != "compute" && mode != "memory"))
            throw std::runtime_error("Invalid stress command or previous run not collected");
        units.clear(); units.reserve(count); worker_error = nullptr; active_mode = mode;
        // CPU oracle preparation cannot contend with the timed inference.
        for (unsigned i = 0; i < 16; ++i) {
            unsigned expected = i;
            if (mode == "memory") {
                for (unsigned j = 0; j < count; ++j) expected = expected * 1664525u + 1013904223u;
            } else {
                unsigned a = i + 1, b = i + 17, c = i + 101, d = i + 1009;
                for (unsigned j = 0; j < cycles; ++j) {
                    a = a * 1664525u + 1013904223u; b = b * 22695477u + 1u;
                    c = c * 1103515245u + 12345u; d = (d ^ (d >> 13)) * 1597334677u;
                }
                expected = a ^ b ^ c ^ d;
            }
            expected_samples[i] = expected;
        }
        void *init_args[] = {&first, &elements};
        launch(init, elements, work_stream, init_args);
        checked(api.cuStreamSynchronize(work_stream));
        stress_active = true;
        // Launch and finish one unit before announcing readiness. That first
        // unit belongs to this fixed workload and remains in recorded timings.
        worker = std::thread([this, mode, count] {
            try {
                checked(api.cuCtxSetCurrent(context));
                stress_start = now_ns();
                for (unsigned iteration = 0; iteration < count; ++iteration) {
                    Unit unit{now_ns(), 0, 0};
                    checked(api.cuEventRecord(work_begin, work_stream));
                    if (mode == "compute") {
                        unsigned compute_count = std::min(elements, 262144u);
                        void *args[] = {&second, &compute_count, &cycles};
                        launch(compute, compute_count, work_stream, args);
                    } else {
                        void *args[] = {&first, &second, &elements};
                        launch(memory, elements, work_stream, args);
                        std::swap(first, second);
                    }
                    checked(api.cuEventRecord(work_end, work_stream));
                    checked(api.cuEventSynchronize(work_end));
                    checked(api.cuEventElapsedTime(&unit.gpu_ms, work_begin, work_end));
                    unit.end_ns = now_ns();
                    units.push_back(unit);
                    if (iteration == 0) emit("{\"event\":\"started\",\"start_ns\":" + std::to_string(stress_start)
                                             + ",\"ready_ns\":" + std::to_string(now_ns()) + "}");
                }
                stress_end = now_ns();
            } catch (...) {
                worker_error = std::current_exception();
                emit("{\"event\":\"error\",\"message\":\"CUDA stress execution failed\"}");
            }
        });
    }
    void collect() {
        if (!stress_active) throw std::runtime_error("No active stress run");
        worker.join(); stress_active = false;
        if (worker_error) std::rethrow_exception(worker_error);
        // The caller sends WAIT after inference returns. Readback and sample
        // validation therefore cannot inflate the overlapping inference time.
        unsigned sample[16];
        checked(api.cuMemcpyDtoH_v2(sample, active_mode == "memory" ? first : second, sizeof(sample)));
        checksum = 0;
        for (unsigned i = 0; i < 16; ++i) {
            if (sample[i] != expected_samples[i]) throw std::runtime_error("GPU stress output differs from CPU oracle");
            checksum = checksum * 131u + sample[i];
        }
        std::ostringstream out;
        out << std::setprecision(9) << "{\"event\":\"stress\",\"mode\":\"" << active_mode
            << "\",\"start_ns\":" << stress_start << ",\"end_ns\":" << stress_end
            << ",\"completed_units\":" << units.size() << ",\"sample_validation\":\"pass\",\"sample_checksum\":"
            << checksum << ",\"units\":[";
        for (size_t i = 0; i < units.size(); ++i) {
            if (i) out << ',';
            out << '[' << units[i].start_ns << ',' << units[i].end_ns << ',' << units[i].gpu_ms << ']';
        }
        out << "]}";
        emit(out.str());
    }
};

int main(int argc, char **argv) {
    try {
        if (argc != 8) throw std::runtime_error("Usage: gpu_workload PTX INPUT_U8 WEIGHTS_TXT OUTPUT_U8 ELEMENTS CYCLES protocol-v1");
        if (std::string(argv[7]) != "protocol-v1") throw std::runtime_error("Unsupported protocol");
        Service service(argv[1], argv[2], argv[3], argv[4], std::stoul(argv[5]), std::stoul(argv[6]));
        emit("{\"event\":\"ready\",\"clock\":\"CLOCK_MONOTONIC\",\"streams\":2,\"context\":\"shared\"}");
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream command(line);
            std::string op; command >> op;
            if (op == "INFER") service.infer();
            else if (op == "STRESS") {
                std::string mode; unsigned count = 0; command >> mode >> count;
                service.start(mode, count);
            } else if (op == "WAIT") service.collect();
            else if (op == "QUIT") break;
            else throw std::runtime_error("Unknown command");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
