// Decode with NVIDIA's unmodified plugin; copy surfaces without range conversion.
// NVDEC surface readback with NvBufSurface / NvBufSurfTransform APIs.
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

    void write_frame(GstSample *sample) {
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
        const Counts counts = decoder.run(argv[1], argv[2], options);
        std::cerr << "NVDEC_DECODE_FRAMES=" << counts.decoded << '\n';
        std::cerr << "NVDEC_COPIED_FRAMES=" << counts.copied << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "NVDEC_DECODE_ERROR: " << error.what() << '\n';
        return 1;
    }
}
