"""The public resident API serializes inputs and consumes actual decoder bytes."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import codec_compute as media
from decoder_core.decode import decode


class RuntimeTests(unittest.TestCase):
    def test__given_explicit_or_invalid_levels__when_compiling__then_level_choice_is_preserved_or_rejected(self):
        for level in (31, 41):
            p = media.compile(lambda x: x, h264_level=level)
            self.assertEqual(p.manifest()["level_idc"], level)
        for level in (True, "41", 32, 40, 41.0):
            with self.subTest(level=level), self.assertRaisesRegex(media.CompileError, "h264_level"):
                media.compile(lambda x: x, h264_level=level)
        with self.assertRaisesRegex(media.CompileError, "only"):
            media.compile(lambda x: x, target="hevc", h264_level=41)

    def test__given_large_extreme_inputs__when_compiling_level41__then_fused_output_and_access_units_remain_exact(self):
        program = media.compile(lambda a, b: media.average(media.sample_h264(a, 2), media.sample_h264(b, -2)),
                                width=256, height=256, h264_level=41,
                                input_ranges={"a": (0, 255), "b": (0, 255)})
        inputs = {"a": [[0]*256 for _ in range(256)], "b": [[255]*256 for _ in range(256)]}
        self.assertEqual(program.manifest()["costs"]["computation_pictures"], 1)
        self.assertEqual(b"".join(program.native.access_units(inputs)), program.encode(inputs))
        self.assertEqual(program.verify(inputs)["output"][128][128], 128)
        legacy_level = media.compile(lambda x: x, width=256, height=256)
        with self.assertRaisesRegex(ValueError, "initial access-unit"):
            legacy_level.encode({"image": inputs["a"]})

    def test__given_changed_inputs_and_named_aliases__when_using_resident_api__then_only_inputs_are_encoded_and_selected_frames_are_mapped(self):
        def kernel(image):
            result = media.average(image, media.sample_h264(image, 2, -2))
            return {"input": image, "output": result, "alias": result}
        program = media.compile(kernel)
        jobs = [{"image": [[(x*11+y*17+offset) % 256 for x in range(32)] for y in range(32)]}
                for offset in (0, 91)]
        expected = [program.reference(inputs) for inputs in jobs]
        frozen = {}
        with tempfile.TemporaryDirectory() as folder:
            for inputs in jobs:
                units = program.native.access_units(inputs)
                path = Path(folder)/"program.h264"
                path.write_bytes(b"".join(units)+program.native.fence_unit)
                raw = decode(path)
                step = program.native.writer.width*program.native.writer.height*3//2
                frozen[units] = tuple(raw[i:i+step] for i in range(0, len(raw)-step, step))
        captures = []

        class FakeSession:
            def __init__(self, **kwargs):
                self.stats = {"requests": 0}
                captures.append(kwargs)
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
            def close(self): self.closed = True
            def submit(self, units, selected_frames=None, *, fence):
                captures.append((selected_frames, fence))
                self.stats["requests"] += 1
                return tuple(frozen[units][index] for index in selected_frames)

        with patch("decoder_core.session.NVDECSession", FakeSession), \
             patch("codec_compute.ir.evaluate", side_effect=AssertionError("runtime graph evaluation")), \
             patch.object(program.native.writer, "prediction", side_effect=AssertionError("runtime instruction emission")), \
             patch.object(program, "reference", side_effect=AssertionError("runtime reference")):
            with program.session(helper="fake") as active:
                for inputs, want in zip(jobs, expected):
                    self.assertEqual(active.run(inputs), want)
                self.assertEqual(active.stats["requests"], 2)
            self.assertTrue(active.decoder.closed)
        self.assertEqual(captures[0], {"width": 64, "height": 64, "helper": "fake", "timeout": 30})
        self.assertTrue(all(item[0] == (0, 1) for item in captures[1:]))
        self.assertTrue(all(item[1] == program.native.fence_unit for item in captures[1:]))

    def test__given_invalid_input__when_submitting__then_graph_validation_precedes_decoder_io(self):
        program = media.compile(lambda x: x)
        with patch("decoder_core.session.NVDECSession") as factory:
            active = program.session()
            with self.assertRaisesRegex(ValueError, "Input image"):
                active.run({"image": [[999]*32 for _ in range(32)]})
            factory.return_value.submit.assert_not_called()
            active.close()
            factory.return_value.close.assert_called_once()

    def test__given_non_nvdec_decoder__when_opening_session__then_unsupported_path_is_explicit(self):
        program = media.compile(lambda x: x)
        with self.assertRaisesRegex(media.UnsupportedOperation, "NVDEC"):
            program.session(decoder="ffmpeg")

    def test__given_corrupt_guard_or_chroma__when_reading_selected_output__then_runtime_closes_and_rejects(self):
        program = media.compile(lambda x: x)
        inputs = {"image": [[0]*32 for _ in range(32)]}
        width, height = program.native.writer.width, program.native.writer.height
        good = bytes(width*height)+bytes([128])*(width*height//2)
        for location, message in ((0, "Guard"), (width*height, "Chroma")):
            broken = bytearray(good)
            broken[location] ^= 1
            with self.subTest(location=location), patch("decoder_core.session.NVDECSession") as factory:
                factory.return_value.submit.return_value = (bytes(broken),)
                active = program.session()
                with self.assertRaisesRegex(AssertionError, message):
                    active.run(inputs)
                factory.return_value.close.assert_called_once()

    def test__given_delayed_diagnostic_failure__when_closing_context__then_api_does_not_hide_it(self):
        program = media.compile(lambda x: x)
        with patch("decoder_core.session.NVDECSession") as factory:
            factory.return_value.__exit__.side_effect = RuntimeError("concealed macroblocks")
            with self.assertRaisesRegex(RuntimeError, "concealed"):
                with program.session():
                    pass


if __name__ == "__main__":
    unittest.main()
