"""Explicitly loaded optional PCM serializer for contiguous compact u8 planes.

Build separately with ``decoder_core/build_pcm.sh /path/libpcm_native.so``;
``NativePCMWriter(writer, helper=path).input_luma(plane, frame_index)`` never
builds code, evaluates a graph, or synthesizes instructions. Read-only views
over partial/non-byte storage may require a compact copy; bytes and writable
contiguous byte buffers use their existing storage during native serialization.
"""

import ctypes
from pathlib import Path
from threading import Lock


class NativePCMWriter:
    """Reuse one Writer's syntax and level policy with an optional native loop.

    Input is a one-dimensional C-contiguous unsigned-byte buffer containing
    active pixels only, in row order. Configuration must remain unchanged;
    create another adapter after changing the writer. Returned bytes own their
    storage. Calls on this adapter are serialized so scratch storage is safe.
    """

    def __init__(self, writer, *, helper):
        # Do not import bitstream here: older benchmark hosts load its source
        # with postponed annotations, and pass that compatible Writer class.
        methods = ("header", "parameter_sets", "_validate_frame", "_checked_access_unit")
        if not all(callable(getattr(writer, method, None)) for method in methods):
            raise TypeError("NativePCMWriter requires a Writer-compatible object")
        if helper is None:
            raise ValueError("An explicit native PCM helper path is required")
        self.writer = writer
        self.width, self.height = writer.width - 32, writer.height - 32
        self.helper = Path(helper).expanduser().resolve(strict=True)
        self._library = ctypes.CDLL(str(self.helper))
        version = self._library.pcm_abi_version
        version.argtypes, version.restype = [], ctypes.c_uint32
        if version() != 1:
            raise RuntimeError("Unsupported native PCM helper ABI")
        self._encode = self._library.pcm_encode
        self._encode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
                                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.c_uint8, ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_size_t)]
        self._encode.restype = ctypes.c_int
        self._configuration = self._signature()
        self._templates = {}
        self._output, self._capacity = None, 0
        self._lock = Lock()

    def _signature(self):
        writer = self.writer
        return (writer.width, writer.height, writer.mb_width, writer.mb_height,
                writer.fps, writer.motion_bits, writer.level_idc, writer.reference_capacity,
                id(writer.reference_slots))

    def _template(self, frame_index):
        if frame_index not in self._templates:
            bits = self.writer.header(0, frame_index)
            bits.ue(25)
            header = bits.aligned_bytes()
            prefix = self.writer.parameter_sets() if frame_index == 0 else b""
            raw_size = self.writer.mb_width*self.writer.mb_height*386 + len(header)-1
            capacity = len(prefix)+5+raw_size+raw_size//2
            self._templates[frame_index] = (header, prefix, capacity,
                                            ctypes.c_char_p(header), ctypes.c_char_p(prefix))
        return self._templates[frame_index]

    def input_luma(self, compact_plane, frame_index=0):
        """Serialize one input identically to Writer.input_luma and check limits."""
        self.writer._validate_frame(frame_index)
        try:
            view = memoryview(compact_plane)
        except TypeError as error:
            raise ValueError("PCM input must be a contiguous one-dimensional u8 buffer") from error
        if view.ndim != 1 or view.itemsize != 1 or view.format != "B" or not view.c_contiguous:
            raise ValueError("PCM input must be a contiguous one-dimensional u8 buffer")
        if view.nbytes != self.width*self.height:
            raise ValueError("PCM byte count differs from the writer's active dimensions")
        # Keep the source owner alive throughout the C call. Partial read-only
        # views need a copy because ctypes.from_buffer requires writable data.
        if not view.readonly:
            owner = (ctypes.c_ubyte*view.nbytes).from_buffer(view)
        elif isinstance(view.obj, bytes) and view.nbytes == len(view.obj):
            owner = ctypes.c_char_p(view.obj)
        else:
            owner = ctypes.c_char_p(view.tobytes())
        with self._lock:
            if self._signature() != self._configuration:
                raise ValueError("Writer configuration changed; create another NativePCMWriter")
            header, prefix, capacity, header_pointer, prefix_pointer = self._template(frame_index)
            if capacity > self._capacity:
                self._output = ctypes.create_string_buffer(capacity)
                self._capacity = capacity
            written = ctypes.c_size_t()
            status = self._encode(owner, view.nbytes, self.width, self.height,
                                  header_pointer, len(header), prefix_pointer, len(prefix),
                                  0x65 if frame_index == 0 else 0x61,
                                  self._output, self._capacity, ctypes.byref(written))
            if status:
                raise RuntimeError(f"Native PCM serializer failed with status {status}")
            if written.value > self._capacity:
                raise RuntimeError("Native PCM helper returned an invalid output size")
            output = ctypes.string_at(self._output, written.value)
            return self.writer._checked_access_unit(frame_index, output,
                                                    "PCM seed" if frame_index == 0 else "PCM input")
