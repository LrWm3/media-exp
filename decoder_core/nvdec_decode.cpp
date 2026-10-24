// Decode with NVIDIA's unmodified plugin; copy surfaces without range conversion.
// NVDEC surface readback with NvBufSurface / NvBufSurfTransform APIs.
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include <chrono>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <cerrno>
#include <cstdint>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

struct Options {
    bool selective = false;
    std::vector<size_t> selected;
    size_t expected_frames = 0;
    unsigned int width = 0, height = 0;
};

static size_t parse_index(const std::string &value) {
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
            "Picture indices and dimensions must be unsigned decimal integers");
    const auto parsed = std::stoull(value);
    require(parsed <= std::numeric_limits<size_t>::max(), "Integer exceeds supported range");
    return static_cast<size_t>(parsed);
}

static Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 3; index < argc; index += 2) {
        require(index + 1 < argc, "Missing option value");
        const std::string flag = argv[index];
        const std::string value = argv[index + 1];
        if (flag == "--frames") {
            require(!options.selective, "Duplicate --frames option");
            options.selective = true;
            std::ifstream input(value);
            require(input.is_open(), "Cannot open picture selection file");
            std::string item;
            while (input >> item) {
                const size_t picture = parse_index(item);
                require(options.selected.empty() || picture > options.selected.back(),
                        "Picture selection must be strictly increasing without duplicates");
                options.selected.push_back(picture);
            }
            require(input.eof(), "Cannot read picture selection file");
            require(!options.selected.empty(), "Picture selection must be nonempty");
        } else if (flag == "--expected-frames") {
            require(!options.expected_frames, "Duplicate --expected-frames option");
            options.expected_frames = parse_index(value);
            require(options.expected_frames > 0, "Expected picture count must be positive");
        } else if (flag == "--width" || flag == "--height") {
            const auto dimension = parse_index(value);
            require(dimension > 0 && dimension <= std::numeric_limits<unsigned int>::max() && !(dimension % 2),
                    "Expected dimensions must be positive even integers");
            auto &target = flag == "--width" ? options.width : options.height;
            require(!target, "Duplicate dimension option");
            target = static_cast<unsigned int>(dimension);
        } else {
            throw std::runtime_error("Unknown option: " + flag);
        }
    }
    require(bool(options.width) == bool(options.height), "Expected width and height must be supplied together");
    if (options.selective && options.expected_frames)
        require(options.selected.back() < options.expected_frames, "Selected picture index out of range");
    return options;
}

struct Counts {
    size_t decoded = 0, copied = 0;
};

// Private, versioned local socket protocol; stdout/stderr remain diagnostic channels.
// All integers are unsigned 64-bit network order. See decoder_core/session.py.
static constexpr uint64_t SESSION_MAGIC = 0x4e564453;
static constexpr uint64_t SESSION_LIMIT = 128 * 1024 * 1024;
class Channel {
    int fd;
    std::chrono::steady_clock::time_point deadline;
public:
    explicit Channel(int value) : fd(value) { reset(); }
    void reset() { deadline = std::chrono::steady_clock::now() + std::chrono::seconds(170); }
    void wait_request() {
        // Idle sessions have no lease. Once any request byte arrives, a bounded
        // deadline applies to the complete request; peer death also wakes poll.
        pollfd pfd{fd, POLLIN, 0};
        int ready;
        do { ready = poll(&pfd, 1, -1); } while (ready < 0 && errno == EINTR);
        require(ready > 0, "Cannot wait for session request");
        reset();
    }
    void transfer(void *data, size_t size, bool writing) {
        auto *bytes = static_cast<unsigned char *>(data);
        while (size) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            require(left > 0, "Session socket timed out");
            pollfd pfd{fd, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
            const int ready = poll(&pfd, 1, static_cast<int>(left));
            if (ready < 0 && errno == EINTR) continue;
            require(ready > 0, "Session socket timed out or failed");
            const auto count = writing ? send(fd, bytes, size, MSG_NOSIGNAL | MSG_DONTWAIT)
                                       : recv(fd, bytes, size, MSG_DONTWAIT);
            if (count < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            require(count > 0, "Session socket closed or failed");
            bytes += count;
            size -= static_cast<size_t>(count);
        }
    }
    uint64_t read() {
        unsigned char bytes[8]; transfer(bytes, 8, false);
        uint64_t value = 0;
        for (auto byte : bytes) value = (value << 8) | byte;
        return value;
    }
    void write(uint64_t value) {
        unsigned char bytes[8];
        for (int i = 7; i >= 0; --i) { bytes[i] = value & 255; value >>= 8; }
        transfer(bytes, 8, true);
    }
};

static void check_unit(const std::vector<unsigned char> &unit, bool idr, unsigned int idr_id) {
    require(unit.size() >= 6 && unit[0] == 0 && unit[1] == 0 && unit[2] == 0 && unit[3] == 1,
            "Expected compiler Annex-B access unit with four-byte start codes");
    unsigned int pictures = 0, sps = 0, pps = 0, slices = 0;
    std::vector<size_t> starts;
    for (size_t offset = 0; offset + 3 < unit.size(); ++offset) {
        if (unit[offset] || unit[offset + 1] || unit[offset + 2] || unit[offset + 3] != 1) continue;
        starts.push_back(offset);
    }
    starts.push_back(unit.size());
    for (size_t nal_index = 0; nal_index + 1 < starts.size(); ++nal_index) {
        const size_t offset = starts[nal_index], end = starts[nal_index + 1];
        require(end - offset >= 6, "Truncated NAL unit");
        const auto header = unit[offset + 4];
        require(!(header & 128), "Invalid NAL forbidden bit");
        const unsigned int kind = header & 31;
        if (kind == 7) { require(idr && nal_index == 0, "Unexpected SPS order"); ++sps; }
        else if (kind == 8) { require(idr && nal_index == 1 && sps == 1, "Unexpected PPS order"); ++pps; }
        else {
            require(kind == (idr ? 5u : 1u), "Unexpected NAL type or IDR boundary");
            require(offset + 5 < unit.size(), "Truncated slice NAL");
            // ue(0) is a leading one bit: exactly one first_mb_in_slice=0.
            if (unit[offset + 5] & 128) ++pictures;
            ++slices;
            if (idr) {
                std::vector<unsigned char> rbsp;
                // Only the short slice header is needed, not the PCM payload.
                unsigned int zeros = 0;
                for (size_t j = offset + 5; j < end && rbsp.size() < 64; ++j) {
                    const auto byte = unit[j];
                    if (zeros == 2 && byte == 3) { zeros = 0; continue; }
                    rbsp.push_back(byte); zeros = byte == 0 ? zeros + 1 : 0;
                }
                size_t position = 0;
                auto bit = [&]() {
                    require(position < rbsp.size() * 8, "Truncated IDR slice header");
                    const auto value = (rbsp[position / 8] >> (7 - position % 8)) & 1;
                    ++position; return value;
                };
                auto ue = [&]() {
                    unsigned int leading = 0;
                    while (!bit()) { require(++leading <= 31, "Invalid IDR slice header"); }
                    uint64_t value = 1;
                    for (unsigned int j = 0; j < leading; ++j) value = 2 * value + bit();
                    return value - 1;
                };
                ue(); ue(); ue();
                for (unsigned int j = 0; j < 16; ++j) bit();
                require(ue() == idr_id, "Compiler input IDR id must be 0 and fence IDR id must be 1");
            }
        }
    }
    require(slices && pictures == 1, "Access unit must contain exactly one picture");
    require(idr ? (sps == 1 && pps == 1) : (!sps && !pps), "Unexpected parameter-set boundary");
}

struct BufferMap {
    GstBuffer *buffer;
    GstMapInfo info = GST_MAP_INFO_INIT;
    explicit BufferMap(GstBuffer *value) : buffer(value) {
        require(buffer && gst_buffer_map(buffer, &info, GST_MAP_READ), "Cannot map decoder GstBuffer");
    }
    ~BufferMap() { gst_buffer_unmap(buffer, &info); }
};

struct SurfaceMap {
    NvBufSurface *surface;
    explicit SurfaceMap(NvBufSurface *value) : surface(value) {
        require(NvBufSurfaceMap(surface, 0, -1, NVBUF_MAP_READ) == 0, "Cannot map pitch-linear surface");
    }
    ~SurfaceMap() { NvBufSurfaceUnMap(surface, 0, -1); }
};

class Decoder {
    GstElement *pipeline = nullptr;
    GstAppSink *sink = nullptr;  // Owned by pipeline.
    GstBus *bus = nullptr;
    NvBufSurface *linear = nullptr;
    std::vector<unsigned char> packed;
    std::ofstream output;
    unsigned int width = 0, height = 0;

    void check_bus() {
        GstMessage *message = gst_bus_pop_filtered(bus,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
        if (!message) return;
        GError *error = nullptr;
        gchar *debug = nullptr;
        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR)
            gst_message_parse_error(message, &error, &debug);
        else
            gst_message_parse_warning(message, &error, &debug);
        std::string detail = "GST ERROR: ";
        detail += error ? error->message : "Unknown pipeline error";
        if (debug) detail += std::string("; ") + debug;
        if (error) g_error_free(error);
        g_free(debug);
        gst_message_unref(message);
        throw std::runtime_error(detail);
    }

    void check_frame(GstSample *sample, const Options &options) {
        GstVideoInfo video;
        gst_video_info_init(&video);
        require(gst_video_info_from_caps(&video, gst_sample_get_caps(sample)), "Missing video dimensions");
        require(GST_VIDEO_INFO_FORMAT(&video) == GST_VIDEO_FORMAT_NV12, "Expected eight-bit NV12 decoder output");
        const unsigned int frame_width = GST_VIDEO_INFO_WIDTH(&video);
        const unsigned int frame_height = GST_VIDEO_INFO_HEIGHT(&video);
        require(frame_width && frame_height && !(frame_width % 2) && !(frame_height % 2),
                "Invalid decoded dimensions");
        if (!width) {
            width = frame_width;
            height = frame_height;
        }
        require(width == frame_width && height == frame_height, "Decoded dimensions changed within stream");
        require(!options.width || (width == options.width && height == options.height),
                "Decoded dimensions do not match expected width and height");
    }

    void pack_frame(GstSample *sample) {
        BufferMap mapped(gst_sample_get_buffer(sample));
        require(mapped.info.size >= sizeof(NvBufSurface), "Decoder buffer does not contain NvBufSurface");
        auto *source = reinterpret_cast<NvBufSurface *>(mapped.info.data);
        require(source->surfaceList && source->batchSize >= 1 && source->numFilled == 1,
                "Expected one decoded surface per buffer");
        const NvBufSurfaceParams &input = source->surfaceList[0];
        const auto format = input.colorFormat;
        require(format == NVBUF_COLOR_FORMAT_NV12 || format == NVBUF_COLOR_FORMAT_NV12_ER ||
                format == NVBUF_COLOR_FORMAT_NV12_709 || format == NVBUF_COLOR_FORMAT_NV12_709_ER ||
                format == NVBUF_COLOR_FORMAT_NV12_2020, "Unsupported decoder surface format");
        require(input.width >= width && input.height >= height, "Invalid decoded surface dimensions");
        if (!linear) {
            NvBufSurfaceCreateParams allocation{};
            allocation.gpuId = source->gpuId;
            allocation.width = input.width;
            allocation.height = input.height;
            allocation.layout = NVBUF_LAYOUT_PITCH;
            allocation.colorFormat = input.colorFormat;  // Preserve *_ER full range.
            allocation.memType = NVBUF_MEM_SURFACE_ARRAY;
            require(NvBufSurfaceCreate(&linear, 1, &allocation) == 0, "Cannot allocate pitch-linear surface");
            NvBufSurfTransformConfigParams session{};
            session.compute_mode = NvBufSurfTransformCompute_VIC;
            session.gpu_id = source->gpuId;
            require(NvBufSurfTransformSetSessionParams(&session) == NvBufSurfTransformError_Success,
                    "Cannot configure surface copy");
            packed.resize(static_cast<size_t>(width) * height * 3 / 2);
        }
        require(linear->surfaceList[0].width == input.width && linear->surfaceList[0].height == input.height &&
                linear->surfaceList[0].colorFormat == input.colorFormat, "Decoded format changed within stream");
        NvBufSurfTransformParams copy{};  // Identical dimensions/format: layout copy only.
        copy.transform_flip = NvBufSurfTransform_None;
        copy.transform_filter = NvBufSurfTransformInter_Nearest;
        require(NvBufSurfTransform(source, linear, &copy) == NvBufSurfTransformError_Success,
                "Cannot copy decoder surface to pitch-linear layout");
        SurfaceMap cpu(linear);
        require(NvBufSurfaceSyncForCpu(linear, 0, -1) == 0, "Cannot synchronize copied surface");
        const auto &plane = linear->surfaceList[0];
        require(plane.planeParams.num_planes == 2 && plane.mappedAddr.addr[0] && plane.mappedAddr.addr[1] &&
                plane.planeParams.pitch[0] >= width && plane.planeParams.pitch[1] >= width &&
                plane.planeParams.height[0] >= height && plane.planeParams.height[1] >= height / 2,
                "Invalid mapped NV12 plane layout");
        const auto *luma = static_cast<const unsigned char *>(plane.mappedAddr.addr[0]);
        const auto *chroma = static_cast<const unsigned char *>(plane.mappedAddr.addr[1]);
        const size_t luma_bytes = static_cast<size_t>(width) * height;
        for (unsigned int y = 0; y < height; ++y)
            std::memcpy(packed.data() + static_cast<size_t>(y) * width,
                        luma + static_cast<size_t>(y) * plane.planeParams.pitch[0], width);
        for (unsigned int y = 0; y < height / 2; ++y) {
            const auto *row = chroma + static_cast<size_t>(y) * plane.planeParams.pitch[1];
            for (unsigned int x = 0; x < width / 2; ++x) {
                const size_t index = static_cast<size_t>(y) * (width / 2) + x;
                packed[luma_bytes + index] = row[2 * x];
                packed[luma_bytes + luma_bytes / 4 + index] = row[2 * x + 1];
            }
        }
    }

    void write_frame(GstSample *sample) {
        pack_frame(sample);
        output.write(reinterpret_cast<const char *>(packed.data()), packed.size());
        require(output.good(), "Cannot write decoded I420 output");
    }

public:
    ~Decoder() {
        if (pipeline) gst_element_set_state(pipeline, GST_STATE_NULL);
        if (bus) gst_object_unref(bus);
        if (pipeline) gst_object_unref(pipeline);
        if (linear) NvBufSurfaceDestroy(linear);
    }

    Counts run(const char *input, const char *destination, const Options &options) {
        pipeline = gst_pipeline_new("exact-nvdec-decode");
        require(pipeline, "Cannot create pipeline");
        GstElement *elements[4]{};
        const char *factories[] = {"filesrc", "h264parse", "nvv4l2decoder", "appsink"};
        for (unsigned int i = 0; i < 4; ++i) {
            elements[i] = gst_element_factory_make(factories[i], nullptr);
            require(elements[i], "Missing required GStreamer element");
            if (!gst_bin_add(GST_BIN(pipeline), elements[i])) {
                gst_object_unref(elements[i]);
                throw std::runtime_error("Cannot add GStreamer element");
            }
        }
        g_object_set(elements[0], "location", input, nullptr);
        g_object_set(elements[2], "enable-error-check", TRUE, nullptr);
        sink = GST_APP_SINK(elements[3]);
        g_object_set(sink, "sync", FALSE, "max-buffers", 2u, "drop", FALSE,
                     "enable-last-sample", FALSE, nullptr);
        GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
        gst_app_sink_set_caps(sink, caps);
        gst_caps_unref(caps);
        require(gst_element_link_many(elements[0], elements[1], elements[2], elements[3], nullptr),
                "Cannot link decoder pipeline");
        bus = gst_element_get_bus(pipeline);
        output.open(destination, std::ios::binary | std::ios::trunc);
        require(output.is_open(), "Cannot open decoded output file");
        require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
                "Cannot start decoder pipeline");
        Counts counts;
        size_t selection_cursor = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(170);
        while (true) {
            check_bus();
            require(std::chrono::steady_clock::now() < deadline, "Decoder timed out before EOS");
            std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample(
                gst_app_sink_try_pull_sample(sink, GST_SECOND), gst_sample_unref);
            if (sample) {
                // Caps inspection needs no NvBufSurface mapping, VIC copy or CPU readback.
                check_frame(sample.get(), options);
                require(!options.expected_frames || counts.decoded < options.expected_frames,
                        "Decoded picture count exceeds expected count");
                if (!options.selective || (selection_cursor < options.selected.size() &&
                                           options.selected[selection_cursor] == counts.decoded)) {
                    write_frame(sample.get());
                    ++counts.copied;
                    if (options.selective) ++selection_cursor;
                }
                ++counts.decoded;
                // Every sample is unreferenced here, including unselected pictures. The
                // hardware decoder owns its internal decoded-picture reference buffers.
            } else {
                check_bus();
                if (gst_app_sink_is_eos(sink)) break;
            }
        }
        require(counts.decoded != 0, "Decoder produced no pictures");
        require(!options.expected_frames || counts.decoded == options.expected_frames,
                "Decoded picture count does not match expected count");
        require(!options.selective || selection_cursor == options.selected.size(),
                "Selected picture index out of range at EOS");
        gst_element_set_state(pipeline, GST_STATE_NULL);
        check_bus();
        output.close();
        require(!output.fail(), "Cannot finish decoded output file");
        return counts;
    }

    // Experimental continuous stream: append acknowledges queueing, collect
    // acknowledges output samples. Only close sends EOS; no request resets DPB.
    Counts stream(int fd, const Options &options) {
        constexpr uint64_t magic = 0x4e565354;  // NVST, separate from NVDS jobs.
        constexpr uint64_t max_pictures = 32767, max_wait_ms = 60000;
        require(options.width && options.height && !options.selective && !options.expected_frames,
                "Stream mode requires fixed dimensions and no file-mode selection");
        const uint64_t pixels = uint64_t(options.width) * options.height;
        require(pixels <= SESSION_LIMIT / 3 * 2, "Stream frame exceeds byte limit");
        const uint64_t frame_bytes = pixels * 3 / 2;
        Channel channel(fd);
        pipeline = gst_pipeline_new("exact-nvdec-stream");
        require(pipeline, "Cannot create stream pipeline");
        GstElement *source = gst_element_factory_make("appsrc", nullptr);
        GstElement *decoder = gst_element_factory_make("nvv4l2decoder", nullptr);
        GstElement *destination = gst_element_factory_make("appsink", nullptr);
        require(source && decoder && destination, "Missing stream GStreamer element");
        gst_bin_add_many(GST_BIN(pipeline), source, decoder, destination, nullptr);
        g_object_set(source, "format", GST_FORMAT_TIME, "is-live", TRUE, "block", FALSE,
                     "max-bytes", static_cast<guint64>(SESSION_LIMIT), nullptr);
        GstCaps *input_caps = gst_caps_from_string("video/x-h264,stream-format=byte-stream,alignment=au");
        gst_app_src_set_caps(GST_APP_SRC(source), input_caps); gst_caps_unref(input_caps);
        g_object_set(decoder, "enable-error-check", TRUE, "enable-full-frame", TRUE, nullptr);
        sink = GST_APP_SINK(destination);
        g_object_set(sink, "sync", FALSE, "async", FALSE, "max-buffers", 2u,
                     "drop", FALSE, "enable-last-sample", FALSE, nullptr);
        GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
        gst_app_sink_set_caps(sink, caps); gst_caps_unref(caps);
        require(gst_element_link_many(source, decoder, destination, nullptr), "Cannot link stream pipeline");
        bus = gst_element_get_bus(pipeline);
        require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
                "Cannot start stream pipeline");
        for (const uint64_t value : {magic, uint64_t(1), uint64_t(options.width), uint64_t(options.height),
                                    frame_bytes, SESSION_LIMIT, max_pictures, max_wait_ms}) channel.write(value);
        Counts counts;
        uint64_t submitted = 0, requests = 0, outstanding_bytes = 0;
        std::vector<uint64_t> unit_sizes;
        constexpr uint64_t tick = GST_SECOND / 25;
        auto pull = [&](GstClockTime timeout) {
            check_bus();
            std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample(
                gst_app_sink_try_pull_sample(sink, timeout), gst_sample_unref);
            if (sample) {
                check_frame(sample.get(), options);
                require(counts.decoded < submitted, "Unexpected extra stream picture");
                const auto pts = GST_BUFFER_PTS(gst_sample_get_buffer(sample.get()));
                require(pts == counts.decoded * tick, "Unexpected stream picture timestamp/order");
                require(outstanding_bytes >= unit_sizes[counts.decoded], "Invalid outstanding stream byte count");
                outstanding_bytes -= unit_sizes[counts.decoded];
                ++counts.decoded;
            }
            check_bus();
            return sample;
        };
        auto reply = [&](uint64_t id, uint64_t returned) {
            for (const uint64_t value : {magic, id, submitted, uint64_t(counts.decoded),
                                        uint64_t(counts.copied), submitted - counts.decoded,
                                        outstanding_bytes, returned}) channel.write(value);
        };
        while (true) {
            channel.wait_request();
            const uint64_t opcode = channel.read(), id = channel.read(), target = channel.read();
            const uint64_t selected_count = channel.read(), wait_ms = channel.read();
            const uint64_t reserved1 = channel.read(), reserved2 = channel.read(), reserved3 = channel.read();
            require(id == requests + 1 && !reserved1 && !reserved2 && !reserved3,
                    "Unexpected stream request id or reserved field");
            require(opcode >= 1 && opcode <= 3, "Unknown stream request opcode");
            if (opcode == 1) {
                require(target && target <= max_pictures - submitted && !selected_count && !wait_ms,
                        "Invalid stream append counts");
                uint64_t bytes = 0;
                std::vector<std::vector<unsigned char>> units;
                for (uint64_t i = 0; i < target; ++i) {
                    const uint64_t size = channel.read();
                    require(size && size <= SESSION_LIMIT - outstanding_bytes - bytes,
                            "Stream outstanding input exceeds byte limit");
                    bytes += size;
                    units.emplace_back(size);
                    channel.transfer(units.back().data(), size, false);
                    check_unit(units.back(), !submitted && i == 0, 0);
                }
                // Conservative bound includes every queued/uncollected input AU,
                // even if appsrc or the decoder has already consumed its bytes.
                for (const auto &unit : units) {
                    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, unit.size(), nullptr);
                    require(buffer, "Cannot allocate stream input buffer");
                    gst_buffer_fill(buffer, 0, unit.data(), unit.size());
                    GST_BUFFER_PTS(buffer) = GST_BUFFER_DTS(buffer) = submitted * tick;
                    GST_BUFFER_DURATION(buffer) = tick;
                    GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_MARKER);
                    unit_sizes.push_back(unit.size());
                    outstanding_bytes += unit.size(); ++submitted;
                    require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
                            "Cannot append stream access unit");
                }
                check_bus(); ++requests; reply(id, 0);
                continue;
            }
            const bool closing = opcode == 3;
            require(wait_ms <= max_wait_ms && (closing ? (!target && !wait_ms)
                    : (target >= counts.decoded && target <= submitted)), "Invalid stream collection target or wait");
            const uint64_t end = closing ? submitted : target;
            require(selected_count <= end - counts.decoded && selected_count <= SESSION_LIMIT / frame_bytes,
                    "Stream selection exceeds count or readback limit");
            std::vector<uint64_t> selected;
            for (uint64_t i = 0; i < selected_count; ++i) {
                const uint64_t index = channel.read();
                require(index >= counts.decoded && index < end && (selected.empty() || index > selected.back()),
                        "Invalid stream picture selection");
                selected.push_back(index);
            }
            if (closing)
                require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK, "Cannot finish stream");
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(closing ? 170000 : wait_ms);
            std::vector<unsigned char> result;
            std::vector<uint64_t> returned;
            result.reserve(selected_count * frame_bytes);
            size_t cursor = 0;
            while (closing ? !gst_app_sink_is_eos(sink) : counts.decoded < end) {
                const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    deadline - std::chrono::steady_clock::now()).count();
                const GstClockTime timeout = remaining > 0
                    ? static_cast<GstClockTime>(std::min<int64_t>(remaining, GST_SECOND / 10)) : 0;
                auto sample = pull(timeout);
                if (sample) {
                    const uint64_t index = counts.decoded - 1;
                    if (cursor < selected.size() && index == selected[cursor]) {
                        pack_frame(sample.get());
                        result.insert(result.end(), packed.begin(), packed.end());
                        returned.push_back(index); ++cursor; ++counts.copied;
                    }
                } else if (closing) {
                    require(std::chrono::steady_clock::now() < deadline || gst_app_sink_is_eos(sink),
                            "Stream EOS timed out");
                } else {
                    require(!gst_app_sink_is_eos(sink), "Unexpected EOS within stream");
                }
                // A collection budget limits the entire operation, including
                // sample processing. Expiry returns partial progress, not error.
                if (!closing && std::chrono::steady_clock::now() >= deadline) break;
            }
            check_bus();
            if (closing) {
                require(counts.decoded == submitted && !outstanding_bytes && cursor == selected.size(),
                        "Incomplete stream at EOS");
                gst_element_set_state(pipeline, GST_STATE_NULL); check_bus();
            }
            ++requests; reply(id, returned.size());
            for (size_t i = 0; i < returned.size(); ++i) {
                channel.write(returned[i]);
                channel.transfer(result.data() + i * frame_bytes, frame_bytes, true);
            }
            if (closing) return counts;
        }
    }

    Counts serve(int fd, const Options &options) {
        require(options.width && options.height && !options.selective && !options.expected_frames,
                "Serve mode requires fixed dimensions and no file-mode selection");
        const uint64_t pixels = uint64_t(options.width) * options.height;
        require(pixels <= SESSION_LIMIT / 3 * 2, "Session frame exceeds byte limit");
        const uint64_t frame_bytes = pixels * 3 / 2;
        Channel channel(fd);
        pipeline = gst_pipeline_new("exact-nvdec-session");
        require(pipeline, "Cannot create session pipeline");
        GstElement *source = gst_element_factory_make("appsrc", nullptr);
        GstElement *decoder = gst_element_factory_make("nvv4l2decoder", nullptr);
        GstElement *destination = gst_element_factory_make("appsink", nullptr);
        require(source && decoder && destination, "Missing session GStreamer element");
        gst_bin_add_many(GST_BIN(pipeline), source, decoder, destination, nullptr);
        // The compiler already provides complete AUs. No parser tail buffering,
        // EOS, flushing event or pipeline state transition occurs between jobs.
        g_object_set(source, "format", GST_FORMAT_TIME, "is-live", TRUE, "block", FALSE,
                     "max-bytes", static_cast<guint64>(SESSION_LIMIT), nullptr);
        GstCaps *input_caps = gst_caps_from_string("video/x-h264,stream-format=byte-stream,alignment=au");
        gst_app_src_set_caps(GST_APP_SRC(source), input_caps); gst_caps_unref(input_caps);
        g_object_set(decoder, "enable-error-check", TRUE, "enable-full-frame", TRUE, nullptr);
        sink = GST_APP_SINK(destination);
        g_object_set(sink, "sync", FALSE, "async", FALSE, "max-buffers", 2u,
                     "drop", FALSE, "enable-last-sample", FALSE, nullptr);
        GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
        gst_app_sink_set_caps(sink, caps); gst_caps_unref(caps);
        require(gst_element_link_many(source, decoder, destination, nullptr), "Cannot link session pipeline");
        bus = gst_element_get_bus(pipeline);
        require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
                "Cannot start session pipeline");
        // Protocol v2 permits zero selected pictures for diagnostic completion-only requests.
        for (const uint64_t value : {SESSION_MAGIC, uint64_t(2), uint64_t(options.width),
                                    uint64_t(options.height), frame_bytes, SESSION_LIMIT}) channel.write(value);
        Counts counts;
        uint64_t submitted = 0, requests = 0, fences_decoded = 0;
        bool pending_fence = false;
        constexpr uint64_t tick = GST_SECOND / 25;
        auto pull = [&](GstClockTime timeout) {
            check_bus();
            std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample(
                gst_app_sink_try_pull_sample(sink, timeout), gst_sample_unref);
            if (sample) {
                check_frame(sample.get(), options);
                const auto pts = GST_BUFFER_PTS(gst_sample_get_buffer(sample.get()));
                require(pts == counts.decoded * tick, "Unexpected picture timestamp/order or extra output");
                ++counts.decoded;
            }
            check_bus();
            return sample;
        };
        while (true) {
            channel.wait_request();
            const uint64_t opcode = channel.read(), id = channel.read();
            const uint64_t n = channel.read(), selected_count = channel.read();
            const uint64_t fence_size = channel.read(), reserved = channel.read();
            require(id == requests + 1 && !reserved, "Unexpected session request id or reserved field");
            if (opcode == 2) {
                require(!n && !selected_count && !fence_size, "Invalid session close request");
                require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK, "Cannot finish session");
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(170);
                while (!gst_app_sink_is_eos(sink)) {
                    require(std::chrono::steady_clock::now() < deadline, "Session EOS timed out");
                    auto sample = pull(GST_SECOND / 10);
                    if (sample) {
                        require(pending_fence && counts.decoded == submitted, "Unexpected picture at session EOS");
                        pending_fence = false; ++fences_decoded;
                    }
                }
                require(!pending_fence && counts.decoded == submitted && fences_decoded == requests,
                        "Incomplete session at EOS");
                gst_element_set_state(pipeline, GST_STATE_NULL); check_bus();
                for (const uint64_t value : {SESSION_MAGIC, id, uint64_t(counts.decoded),
                                            uint64_t(counts.copied), fences_decoded, uint64_t(0)}) channel.write(value);
                return counts;
            }
            require(opcode == 1 && n && n < 32768 && selected_count <= n,
                    "Invalid session request counts");
            require(fence_size && fence_size <= SESSION_LIMIT && selected_count <= SESSION_LIMIT / frame_bytes,
                    "Session request exceeds byte limit");
            std::vector<size_t> selected;
            for (uint64_t i = 0; i < selected_count; ++i) {
                const uint64_t index = channel.read();
                require(index < n && (selected.empty() || index > selected.back()), "Invalid session picture selection");
                selected.push_back(index);
            }
            uint64_t bytes = fence_size;
            std::vector<std::vector<unsigned char>> units;
            for (uint64_t i = 0; i <= n; ++i) {
                const uint64_t size = i == n ? fence_size : channel.read();
                require(size && size <= SESSION_LIMIT && (i == n || size <= SESSION_LIMIT - bytes),
                        "Session input exceeds byte limit");
                if (i != n) bytes += size;
                units.emplace_back(size);
                channel.transfer(units.back().data(), size, false);
                check_unit(units.back(), i == 0 || i == n, i == n ? 1 : 0);
            }
            require(submitted <= (GST_CLOCK_TIME_NONE / tick) - n - 1, "Session timestamp limit reached");
            const uint64_t first = submitted;
            for (const auto &unit : units) {
                GstBuffer *buffer = gst_buffer_new_allocate(nullptr, unit.size(), nullptr);
                require(buffer, "Cannot allocate session input buffer");
                gst_buffer_fill(buffer, 0, unit.data(), unit.size());
                GST_BUFFER_PTS(buffer) = GST_BUFFER_DTS(buffer) = submitted * tick;
                GST_BUFFER_DURATION(buffer) = tick;
                GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_MARKER);
                ++submitted;
                require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
                        "Cannot submit session access unit");
            }
            std::vector<unsigned char> result;
            result.reserve(selected_count * frame_bytes);
            size_t cursor = 0;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(170);
            while (counts.decoded < first + n) {
                require(std::chrono::steady_clock::now() < deadline, "Session job timed out before fence");
                auto sample = pull(GST_SECOND / 10);
                if (!sample) { require(!gst_app_sink_is_eos(sink), "Unexpected EOS within session"); continue; }
                const uint64_t index = counts.decoded - 1;
                if (index < first) {
                    require(pending_fence && index + 1 == first, "Unexpected pending picture");
                    pending_fence = false; ++fences_decoded;
                } else if (cursor < selected.size() && index - first == selected[cursor]) {
                    pack_frame(sample.get());
                    result.insert(result.end(), packed.begin(), packed.end());
                    ++cursor; ++counts.copied;
                }
            }
            require(cursor == selected.size() && !pending_fence, "Incomplete session selection");
            pending_fence = true; ++requests;
            // A backend may also release the final fence immediately. Accept exactly
            // that timestamp, never a second output, and report its actual status.
            if (auto sample = pull(0)) {
                require(counts.decoded == submitted, "Unexpected extra session picture");
                pending_fence = false; ++fences_decoded;
            }
            for (const uint64_t value : {SESSION_MAGIC, id, uint64_t(counts.decoded), uint64_t(counts.copied),
                                        fences_decoded, uint64_t(pending_fence)}) channel.write(value);
            channel.transfer(result.data(), result.size(), true);
        }
    }
};

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: nvdec_decode INPUT.h264 OUTPUT.i420 [--frames INDICES.txt] "
                     "[--expected-frames COUNT] [--width WIDTH --height HEIGHT]\n";
        return 2;
    }
    try {
        const Options options = parse_options(argc, argv);
        gst_init(nullptr, nullptr);
        Decoder decoder;
        const Counts counts = std::string(argv[1]) == "--serve"
            ? decoder.serve(static_cast<int>(parse_index(argv[2])), options)
            : std::string(argv[1]) == "--stream"
            ? decoder.stream(static_cast<int>(parse_index(argv[2])), options)
            : decoder.run(argv[1], argv[2], options);
        std::cerr << "NVDEC_DECODE_FRAMES=" << counts.decoded << '\n';
        std::cerr << "NVDEC_COPIED_FRAMES=" << counts.copied << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "NVDEC_DECODE_ERROR: " << error.what() << '\n';
        return 1;
    }
}
