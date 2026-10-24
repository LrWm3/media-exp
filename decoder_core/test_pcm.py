"""Optional native PCM serialization preserves the authoritative Writer bytes."""

import array
import ast
import builtins
from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode
from decoder_core.pcm import NativePCMWriter


def rows(plane, width):
    return [list(plane[start:start+width]) for start in range(0, len(plane), width)]


def expected_i420(plane, width, height):
    coded_width, coded_height = width+32, height+32
    luma = bytearray(coded_width*coded_height)
    for y in range(height):
        start = (y+16)*coded_width+16
        luma[start:start+width] = plane[y*width:(y+1)*width]
    return bytes(luma)+bytes([128])*(coded_width*coded_height//2)


class NativePCMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which(os.environ.get("CXX", "c++")):
            raise unittest.SkipTest("A C++17 compiler is required for optional PCM tests")
        cls.directory = tempfile.TemporaryDirectory(prefix="native-pcm-test-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.helper = Path(cls.directory.name)/"libpcm_native.so"
        subprocess.run(["bash", str(Path(__file__).with_name("build_pcm.sh")), str(cls.helper)],
                       check=True, capture_output=True, text=True)

    def adapter(self, writer):
        return NativePCMWriter(writer, helper=self.helper)

    def test__given_full_byte_inputs_and_reference_plans__when_serializing_both_geometries__then_every_byte_matches_writer(self):
        for size, level in ((128, 31), (256, 41)):
            randomizer = random.Random(718+size)
            patterns = {"zero": bytes(size*size), "maximum": bytes([255])*(size*size),
                        "full-byte": bytes(range(256))*(size*size//256),
                        "random": bytes(randomizer.randrange(256) for _ in range(size*size)),
                        "escape-heavy": (bytes([0, 0, 0, 1, 0, 0, 2, 3])*(size*size//8))}
            for mode in ("row", "macroblock"):
                for slots in (None, {0: 0, 1: 1, 2: 0}):
                    # Macroblock prediction capacity at 128/3.1 requires <=18fps.
                    writer = Writer(size, size, level_idc=level, slice_mode=mode, fps=15,
                                    reference_slots=slots)
                    adapter = self.adapter(writer)
                    for label, plane in patterns.items():
                        board = rows(plane, size)
                        for frame in (0, 1, 2):
                            with self.subTest(size=size, mode=mode, slots=slots, label=label, frame=frame):
                                expected = writer.input_luma(board, frame)
                                self.assertEqual(adapter.input_luma(plane, frame), expected)
                                self.assertEqual([nal[0] & 31 for nal in expected.split(b"\0\0\0\1")[1:]],
                                                 [7, 8, 5] if frame == 0 else [1])

    def test__given_rectangles_and_large_frame_indices__when_reusing_templates__then_strides_and_header_wraps_match(self):
        for width, height in ((16, 32), (48, 16)):
            writer = Writer(width, height, motion_bits=8)
            adapter = self.adapter(writer)
            plane = bytes((index*71+index//width) % 256 for index in range(width*height))
            for frame in (0, 1, 127, 32767, 32768, 65535, 65536):
                with self.subTest(width=width, height=height, frame=frame):
                    self.assertEqual(adapter.input_luma(plane, frame), writer.input_luma(rows(plane, width), frame))

    def test__given_bytes_and_partial_buffer_views__when_serializing__then_compact_visible_pixels_are_used(self):
        writer = Writer(16, 16)
        adapter = self.adapter(writer)
        plane = bytes(range(256))
        expected = writer.input_luma(rows(plane, 16), 0)
        storage = bytearray(b"prefix"+plane+b"suffix")
        variants = (plane, bytearray(plane), memoryview(plane), memoryview(bytearray(plane)),
                    memoryview(b"prefix"+plane+b"suffix")[6:-6], memoryview(storage)[6:-6],
                    memoryview(storage)[6:-6].toreadonly())
        for variant in variants:
            with self.subTest(kind=type(variant).__name__):
                self.assertEqual(adapter.input_luma(variant), expected)

    def test__given_malformed_or_wrong_sized_buffers__when_serializing__then_native_code_is_not_called(self):
        adapter = self.adapter(Writer(16, 16))
        invalid = (None, [0]*256, b"", bytes(255), bytes(257), array.array("b", [0]*256),
                   array.array("H", [0]*256), memoryview(bytes(512))[::2],
                   memoryview(bytes(256)).cast("B", (16, 16)))
        with patch.object(adapter, "_encode", side_effect=AssertionError("invalid buffer reached native code")):
            for plane in invalid:
                with self.subTest(kind=type(plane).__name__), self.assertRaisesRegex(ValueError, "PCM"):
                    adapter.input_luma(plane)

    def test__given_invalid_or_unplanned_frames__when_serializing__then_writer_validation_is_preserved(self):
        adapter = self.adapter(Writer(16, 16, reference_slots={0: 0, 1: 1}))
        for frame in (True, False, -1, None, "1", 1.0, 2):
            with self.subTest(frame=frame), self.assertRaisesRegex(ValueError, "frame"):
                adapter.input_luma(bytes(256), frame)

    def test__given_prepared_templates_and_changed_inputs__when_serializing_again__then_only_fill_escape_and_limits_run(self):
        writer = Writer(16, 16, reference_slots={0: 0, 1: 1})
        adapter = self.adapter(writer)
        plane = bytes(range(256))
        expected = [writer.input_luma(rows(plane, 16), frame) for frame in (0, 1)]
        for frame in (0, 1):
            adapter.input_luma(bytes(256), frame)
        with patch.object(writer, "header", side_effect=AssertionError("header rebuilt")), \
                patch.object(writer, "parameter_sets", side_effect=AssertionError("SPS/PPS rebuilt")), \
                patch.object(writer, "input_luma", side_effect=AssertionError("Python PCM fallback")), \
                patch.object(writer, "_checked_access_unit", wraps=writer._checked_access_unit) as limits:
            for frame in (0, 1):
                actual = adapter.input_luma(plane, frame)
                self.assertEqual(actual, expected[frame])
                limits.assert_called_with(frame, actual, "PCM seed" if frame == 0 else "PCM input")
            self.assertEqual(limits.call_count, 2)
        with patch.object(writer, "_checked_access_unit", side_effect=ValueError("test level budget")):
            with self.assertRaisesRegex(ValueError, "test level budget"):
                adapter.input_luma(plane)

    def test__given_access_units_over_level_limits__when_serializing_natively__then_same_budget_rejects(self):
        plane = bytes(256*256)
        board = rows(plane, 256)
        for level, fps, message in ((31, 30, "initial access-unit"), (41, 60, "VCL bitrate")):
            writer = Writer(256, 256, level_idc=level, fps=fps, slice_mode="row")
            with self.subTest(level=level, implementation="native"), self.assertRaisesRegex(ValueError, message):
                self.adapter(writer).input_luma(plane)
            with self.subTest(level=level, implementation="Python"), self.assertRaisesRegex(ValueError, message):
                writer.input_luma(board, 0)
        writer = Writer(256, 256, level_idc=31, slice_mode="row")
        self.assertEqual(self.adapter(writer).input_luma(plane, 1), writer.input_luma(board, 1))

    def test__given_changed_writer_configuration__when_reusing_adapter__then_stale_templates_are_rejected(self):
        writer = Writer(16, 16)
        adapter = self.adapter(writer)
        adapter.input_luma(bytes(256))
        writer.motion_bits = 8
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            adapter.input_luma(bytes(256))

    def test__given_concurrent_calls__when_reusing_scratch__then_results_own_correct_immutable_storage(self):
        writer = Writer(16, 16)
        adapter = self.adapter(writer)
        planes = [bytes((value+index*17) % 256 for value in range(256)) for index in range(12)]
        expected = [writer.input_luma(rows(plane, 16), index % 3) for index, plane in enumerate(planes)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = list(executor.map(lambda item: adapter.input_luma(item[1], item[0] % 3), enumerate(planes)))
        self.assertEqual(actual, expected)
        self.assertTrue(all(type(result) is bytes for result in actual))

    def test__given_explicit_helper_contract__when_loading_without_library_or_writer__then_no_build_is_attempted(self):
        with self.assertRaisesRegex(TypeError, "Writer-compatible"):
            NativePCMWriter(object(), helper=self.helper)
        with self.assertRaisesRegex(ValueError, "explicit"):
            NativePCMWriter(Writer(16, 16), helper=None)
        with self.assertRaises(FileNotFoundError):
            NativePCMWriter(Writer(16, 16), helper=Path(self.directory.name)/"missing.so")

    def test__given_older_host_writer_loader__when_importing_adapter__then_no_bitstream_import_or_modern_annotation_is_required(self):
        source = Path(__file__).with_name("pcm.py")
        ast.parse(source.read_text(), feature_version=(3, 8))
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if "bitstream" in name:
                raise AssertionError("Adapter imported host-incompatible bitstream source")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked_import):
            cls = runpy.run_path(str(source))["NativePCMWriter"]
        writer = Writer(16, 16)

        class LoadedWriter:
            def __getattr__(self, name):
                return getattr(writer, name)

        plane = bytes(range(256))
        self.assertEqual(cls(LoadedWriter(), helper=self.helper).input_luma(plane),
                         writer.input_luma(rows(plane, 16), 0))

    def test__given_native_invalid_sizes_and_pointers__when_calling_c_abi__then_buffers_are_untouched(self):
        adapter = self.adapter(Writer(16, 16))
        source = ctypes.create_string_buffer(bytes(256))
        header = ctypes.create_string_buffer(b"\x80")
        output = ctypes.create_string_buffer(b"!"*32)
        written = ctypes.c_size_t(99)
        default = [source, 256, 16, 16, header, 1, None, 0, 0x65, output, 32, ctypes.byref(written)]
        size_max = ctypes.c_size_t(-1).value
        cases = [({0: None}, 1), ({1: 255}, 1), ({1: 257}, 1), ({2: 0}, 1), ({2: 15}, 1),
                 ({2: 17}, 1), ({3: 17}, 1), ({4: None}, 1), ({5: 0}, 1), ({5: size_max}, 2),
                 ({7: 1}, 1), ({6: header, 7: size_max}, 2),
                 ({8: 0x67}, 1), ({9: None}, 1), ({11: None}, 1),
                 ({2: 0xfffffff0, 3: 0xfffffff0}, 2), ({}, 3)]
        for changes, status in cases:
            args = default.copy()
            for index, value in changes.items():
                args[index] = value
            with self.subTest(changes=changes):
                self.assertEqual(adapter._encode(*args), status)
                self.assertEqual(output.raw, b"!"*32+b"\0")
                if status == 3:
                    self.assertGreater(written.value, len(output))

    def test__given_native_pcm_inputs_followed_by_p_and_b__when_decoding__then_all_active_guard_and_chroma_bytes_match(self):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        for size, level in ((128, 31), (256, 41)):
            writer = Writer(size, size, level_idc=level, slice_mode="row",
                            reference_slots={0: 0, 1: 1, 2: 2, 3: 0})
            adapter = self.adapter(writer)
            a = bytes((17*x+31*y) % 256 for y in range(size) for x in range(size))
            b = bytes(255-value for value in a)
            moved = bytes(max(0, a[y*size+x+1]-7) if x+1 < size else 0
                          for y in range(size) for x in range(size))
            average = bytes((left+right+1)//2 for left, right in zip(moved, b))
            stream = adapter.input_luma(a, 0)+adapter.input_luma(b, 1)
            stream += writer.prediction(2, Operation("moved", (Source(0, offset=-7, dx=1),)))
            stream += writer.prediction(3, Operation("average", (Source(2), Source(1))))
            expected = b"".join(expected_i420(plane, size, size) for plane in (a, b, moved, average))
            path = Path(self.directory.name)/f"native-{size}.h264"
            path.write_bytes(stream)
            for backend in backends:
                with self.subTest(size=size, decoder=backend):
                    self.assertEqual(decode(path, backend), expected)


if __name__ == "__main__":
    unittest.main()
