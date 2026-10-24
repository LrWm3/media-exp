"""Selected physical-picture readback against real decoder output."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import h264
from decoder_core.decode import available, decode, decode_batch


BACKENDS = tuple(backend for backend in ("ffmpeg", "vaapi", "openh264", "nvdec")
                 if available(backend))


class SelectiveReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not available("ffmpeg"):
            raise unittest.SkipTest("FFmpeg unavailable")
        cls.image = [[(x+3*y) % 2 for x in range(16)] for y in range(16)]
        cls.program = h264.compile(
            lambda x: {"result": h264.bit_not(h264.bit_or(x, h264.shift(x, 1))),
                       "input": x, "input_alias": x}, 16, 16)
        cls.indices = cls.program.output_picture_indices
        cls.frame_size = cls.program.writer.width*cls.program.writer.height*3//2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"program.h264"
            path.write_bytes(cls.program.encode(cls.image))
            cls.full_raw = decode(path)
        cls.selected_raw = cls.pack(cls.indices)

    @classmethod
    def pack(cls, selection):
        return b"".join(cls.full_raw[index*cls.frame_size:(index+1)*cls.frame_size]
                        for index in selection)

    def test__given_output_aliases_in_reverse_order__when_selecting__then_physical_indices_are_sorted_and_unique(self):
        self.assertEqual(self.program.output_frames, {"result": 3, "input": 1, "input_alias": 1})
        self.assertEqual(self.indices, (1, 3))
        self.assertEqual(len(self.selected_raw), 2*self.frame_size)

    def test__given_named_outputs_and_aliases__when_decoding_selected_pictures__then_all_available_decoders_match_full_readback(self):
        expected = self.program.reference(self.image)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                full = self.program.verify(self.image, backend, readback="all")
                selected = self.program.verify(self.image, backend, readback="outputs")
                self.assertEqual(full, expected)
                self.assertEqual(selected, full)
                self.assertEqual(selected["input_alias"], self.image)

    def test__given_zero_operation_binary_kernel__when_selecting_input_aliases__then_normalized_input_is_returned(self):
        program = h264.compile(lambda x: {"input": x, "alias": x}, 16, 16)
        self.assertEqual(program.costs["arithmetic_pictures"], 0)
        self.assertEqual(program.pictures, 2)
        self.assertEqual(program.output_picture_indices, (1,))
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                expected = {"input": self.image, "alias": self.image}
                self.assertEqual(program.verify(self.image, backend), expected)
                self.assertEqual(program.verify(self.image, backend, readback="outputs"), expected)

    def test__given_zero_operation_named_inputs__when_selecting_input_aliases__then_original_physical_pictures_are_returned(self):
        program = h264.compile(lambda a, b: {"b": b, "a": a, "alias": a}, 16, 16,
                               input_ranges={"a": (0, 255), "b": (0, 255)})
        inputs = {"a": [[x+16*y for x in range(16)] for y in range(16)],
                  "b": [[255-x-16*y for x in range(16)] for y in range(16)]}
        self.assertEqual(program.pictures, 2)
        self.assertEqual(program.output_picture_indices, (0, 1))
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                full = program.verify(inputs, backend)
                self.assertEqual(program.verify(inputs, backend, readback="outputs"), full)
                self.assertEqual(full, {"b": inputs["b"], "a": inputs["a"], "alias": inputs["a"]})

    def test__given_single_literal_input_picture__when_selecting_identity_output__then_ffmpeg_and_vaapi_preserve_seed(self):
        program = h264.compile(lambda x: x, 16, 16, input_range=(0, 255))
        image = [[x+16*y for x in range(16)] for y in range(16)]
        self.assertEqual(program.pictures, 1)
        self.assertEqual(program.output_picture_indices, (0,))
        for backend in (backend for backend in BACKENDS if backend != "openh264"):
            with self.subTest(backend=backend):
                self.assertEqual(program.verify(image, backend), {"output": image})
                self.assertEqual(program.verify(image, backend, readback="outputs"), {"output": image})

    @unittest.skipUnless("openh264" in BACKENDS, "OpenH264 unavailable")
    def test__given_single_literal_seed__when_openh264_cannot_flush_it__then_both_modes_propagate_decoder_failure(self):
        # The current GStreamer OpenH264 plugin cannot flush a lone I_PCM IDR.
        # Preserve the one-picture stream and report its decoder error, without
        # adding a dummy picture or fabricating an output for this edge case.
        program = h264.compile(lambda x: x, 16, 16, input_range=(0, 255))
        self.assertEqual(program.pictures, 1)
        results = []
        for mode in ("all", "outputs"):
            with self.subTest(readback=mode):
                try:
                    results.append(program.verify(self.image, "openh264", readback=mode))
                except subprocess.CalledProcessError as error:
                    self.assertIn(b"No valid frames decoded before end of stream", error.stderr)
                    results.append(error.returncode)
        # Also accepts plugin versions that fix the limitation, provided both
        # modes return the verified single picture instead of dropping it.
        self.assertEqual(results[0], results[1])

    def test__given_packed_selection_and_full_stream__when_reading__then_all_named_outputs_match_reference(self):
        expected = self.program.reference(self.image)
        self.assertEqual(self.program.read(self.full_raw), expected)
        self.assertEqual(self.program.read(self.full_raw, selected_frames=range(self.program.pictures)), expected)
        self.assertEqual(self.program.read(memoryview(self.selected_raw), selected_frames=iter(self.indices)), expected)
        with_extra = (1, 2, 3)
        self.assertEqual(self.program.read(self.pack(with_extra), selected_frames=with_extra), expected)

    def test__given_invalid_or_incomplete_selection__when_reading__then_contract_violation_is_explicit(self):
        cases = (((), "nonempty"), ((3, 1), "strictly increasing"),
                 ((1, 1, 3), "strictly increasing"), ((-1, 1, 3), "out of range"),
                 ((1, 3, self.program.pictures), "out of range"),
                 ((True, 3), "integer picture"), ((1.0, 3), "integer picture"),
                 (("1", 3), "integer picture"), ((1,), "missing: result"),
                 ((3,), "missing: input, input_alias"))
        for selection, message in cases:
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, message):
                self.program.read(self.selected_raw, selected_frames=selection)

    def test__given_wrong_packed_length__when_reading_selected_or_all__then_missing_and_extra_bytes_fail(self):
        for raw in (b"", self.selected_raw[:-1], self.selected_raw+b"\x00", self.full_raw):
            with self.subTest(length=len(raw)), self.assertRaisesRegex(AssertionError, "picture count"):
                self.program.read(raw, selected_frames=self.indices)
        with self.assertRaisesRegex(AssertionError, "picture count"):
            self.program.read(self.selected_raw)

    def test__given_corrupt_guard_in_any_supplied_picture__when_reading__then_physical_frame_is_reported(self):
        width, height = self.program.writer.width, self.program.writer.height
        # Include the unnamed intermediate: explicitly supplied frames must all
        # pass validation, even when no named output reads their active image.
        selection = (1, 2, 3)
        offsets = (0, (height-1)*width, 16*width, 17*width-1)
        for position, frame in enumerate(selection):
            for offset in offsets:
                raw = bytearray(self.pack(selection))
                raw[position*self.frame_size+offset] = 1
                with self.subTest(frame=frame, offset=offset), self.assertRaisesRegex(
                        AssertionError, f"Guard changed at frame {frame}"):
                    self.program.read(raw, selected_frames=selection)

    def test__given_corrupt_chroma_in_any_supplied_picture__when_reading__then_physical_frame_is_reported(self):
        luma_size = self.program.writer.width*self.program.writer.height
        selection = (1, 2, 3)
        for position, frame in enumerate(selection):
            for offset in (luma_size, self.frame_size-1):
                raw = bytearray(self.pack(selection))
                raw[position*self.frame_size+offset] = 127
                with self.subTest(frame=frame, offset=offset), self.assertRaisesRegex(
                        AssertionError, f"Chroma changed at frame {frame}"):
                    self.program.read(raw, selected_frames=selection)

    def test__given_out_of_range_input_or_result__when_reading_selected__then_each_named_output_range_is_checked(self):
        active = 16*self.program.writer.width+16
        for position, name in ((0, "input"), (1, "result")):
            raw = bytearray(self.selected_raw)
            raw[position*self.frame_size+active] = 2
            with self.subTest(name=name), self.assertRaisesRegex(
                    AssertionError, f"Decoded output {name} violates its proven range"):
                self.program.read(raw, selected_frames=self.indices)

    def test__given_unknown_readback_mode__when_decoding_or_verifying__then_failure_precedes_encoding(self):
        with patch.object(self.program, "encode") as encode:
            for method in (self.program.decode, self.program.verify):
                for mode in ("output", "ALL", "", None):
                    with self.subTest(method=method.__name__, mode=mode), self.assertRaisesRegex(ValueError, "readback"):
                        method(self.image, readback=mode)
            encode.assert_not_called()

    def test__given_two_real_jobs__when_batch_selecting_outputs__then_pixels_and_actual_copy_counts_match_backend(self):
        other = [[1-value for value in row] for row in self.image]
        images = (self.image, other)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/f"job-{index}.h264" for index in range(2)]
            originals = [self.program.encode(image) for image in images]
            for path, stream in zip(paths, originals):
                path.write_bytes(stream)
            for backend in BACKENDS:
                with self.subTest(backend=backend):
                    result = decode_batch(paths, backend, width=self.program.writer.width,
                                          height=self.program.writer.height,
                                          pictures_per_job=self.program.pictures,
                                          selected_frames=self.indices)
                    self.assertEqual(result.selected_frames, self.indices)
                    self.assertEqual(result.decoded_frames, 2*self.program.pictures)
                    copied_per_job = len(self.indices) if backend == "nvdec" else self.program.pictures
                    self.assertEqual(result.copied_frames, 2*copied_per_job)
                    self.assertEqual(result.job_count, 2)
                    for image, frames in zip(images, result.frames):
                        self.assertEqual(len(frames), len(self.indices))
                        self.assertTrue(all(len(frame) == self.frame_size for frame in frames))
                        actual = self.program.read(b"".join(frames), selected_frames=result.selected_frames)
                        self.assertEqual(actual, self.program.reference(image))
            self.assertEqual([path.read_bytes() for path in paths], originals)


if __name__ == "__main__":
    unittest.main(verbosity=2)
