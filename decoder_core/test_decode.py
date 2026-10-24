"""Decoder selection and NVDEC's explicit error/concealment diagnostics."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from decoder_core.decode import _validate_nvdec_diagnostics, available, decode, decode_batch


NVDEC_STARTUP = (
    b"NvMMLiteOpen : Block : BlockType = 261 \n"
    b"NvMMLiteBlockCreate : Block : BlockType = 261 \n"
    b"NVMMLITE_NVVIDEODEC, <NvMMLiteNvVideoDecSetAttribute:9304> Error status reporting set to 1\n"
    b"Opening in BLOCKING MODE \n"
)
NVDEC_FRAME = b"ErrorType= 0  Decoded MBs= 36  Concealed MBs= 0  FrameDecodeTime 123\n"
NVDEC_DONE = b"NVDEC_DECODE_FRAMES=1\n"


class DecodeTests(unittest.TestCase):
    def test__given_nvdec_status_messages__when_decoding__then_explicit_helper_returns_i420(self):
        captured = []

        def run(command, **kwargs):
            captured.append(command)
            self.assertEqual(kwargs, {"capture_output": True, "check": True, "timeout": 180})
            Path(command[-1]).write_bytes(b"decoded i420")
            return subprocess.CompletedProcess(command, 0, NVDEC_STARTUP, NVDEC_FRAME + NVDEC_DONE)

        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "nvdec_decode"
            helper.write_text("unused mock helper")
            helper.chmod(0o700)
            with patch.dict(os.environ, {"H264_NVDEC_DECODER": str(helper)}):
                with patch("decoder_core.decode.subprocess.run", side_effect=run):
                    self.assertEqual(decode("input.h264", "nvdec"), b"decoded i420")
        command = captured[0]
        self.assertEqual(command[:2], [str(helper), str(Path("input.h264").resolve())])
        self.assertEqual(len(command), 3)

    def test__given_whitelisted_stdout_and_stderr__when_validating__then_decimal_and_hex_zero_status_are_accepted(self):
        hexadecimal = b"ErrorType= 0x0  Decoded MBs= 0x24  Concealed MBs= 0X00  FrameDecodeTime= 123\n"
        for stdout, stderr in ((NVDEC_STARTUP + NVDEC_FRAME, b""), (b"", NVDEC_STARTUP + NVDEC_FRAME),
                               (NVDEC_STARTUP, hexadecimal), (b"\n", b"")):
            with self.subTest(stdout=stdout, stderr=stderr):
                self.assertEqual(_validate_nvdec_diagnostics(stdout, stderr + NVDEC_DONE), 1)

    def test__given_missing_zero_or_duplicate_picture_count__when_validating__then_helper_cannot_succeed_silently(self):
        for marker in (b"", b"NVDEC_DECODE_FRAMES=0\n", NVDEC_DONE + NVDEC_DONE):
            with self.subTest(marker=marker), self.assertRaisesRegex(RuntimeError, "picture count"):
                _validate_nvdec_diagnostics(NVDEC_STARTUP + NVDEC_FRAME, marker)

    def test__given_empty_output_or_missing_helper__when_decoding__then_failure_is_explicit(self):
        with patch("decoder_core.decode.Path.is_file", return_value=False), self.assertRaisesRegex(RuntimeError, "Build"):
            decode("input.h264", "nvdec")

        def run(command, **kwargs):
            Path(command[-1]).write_bytes(b"")
            return subprocess.CompletedProcess(command, 0, NVDEC_STARTUP, NVDEC_FRAME + NVDEC_DONE)

        with patch("decoder_core.decode.Path.is_file", return_value=True), patch("decoder_core.decode.os.access", return_value=True):
            with patch("decoder_core.decode.subprocess.run", side_effect=run), self.assertRaisesRegex(RuntimeError, "empty"):
                decode("input.h264", "nvdec")

    def test__given_nonzero_hardware_error_or_concealment__when_validating_either_stream__then_decode_fails(self):
        bad_frames = (
            NVDEC_FRAME.replace(b"ErrorType= 0", b"ErrorType= 1"),
            NVDEC_FRAME.replace(b"ErrorType= 0", b"ErrorType= 0x80"),
            NVDEC_FRAME.replace(b"Concealed MBs= 0", b"Concealed MBs= 3"),
            NVDEC_FRAME.replace(b"Concealed MBs= 0", b"Concealed MBs= 0x01"),
        )
        for frame in bad_frames:
            for stdout, stderr in ((NVDEC_STARTUP + frame, b""), (NVDEC_STARTUP, frame)):
                with self.subTest(frame=frame, stderr=bool(stderr)), self.assertRaisesRegex(RuntimeError, "error or concealed"):
                    _validate_nvdec_diagnostics(stdout, stderr)

    def test__given_gstreamer_errors_or_unknown_diagnostics__when_validating__then_no_line_is_silently_ignored(self):
        diagnostics = (
            b"ERROR: from element /GstPipeline:pipeline0/nvv4l2decoder: decoder failed\n",
            b"0:00:00.123 ERROR v4l2 decoder could not decode frame\n",
            b"NvMMLiteOpen : Block : BlockType = 261 ERROR\n",
            b"NVMMLITE_NVVIDEODEC, <NvMMLiteNvVideoDecSetAttribute:9304> Error status reporting set to 0\n",
            b"ErrorType= unknown  Decoded MBs= 36  Concealed MBs= 0  FrameDecodeTime 123\n",
            b"WARNING: unexpected hardware warning\n",
        )
        for diagnostic in diagnostics:
            for stdout, stderr in ((diagnostic, b""), (b"", diagnostic)):
                with self.subTest(diagnostic=diagnostic, stderr=bool(stderr)), self.assertRaises(RuntimeError):
                    _validate_nvdec_diagnostics(stdout, stderr)

    def test__given_nvdec_process_failure__when_decoding__then_subprocess_error_propagates(self):
        failure = subprocess.CalledProcessError(1, ["gst-launch-1.0"], stderr=b"ERROR")
        with patch("decoder_core.decode.Path.is_file", return_value=True), patch("decoder_core.decode.os.access", return_value=True):
            with patch("decoder_core.decode.subprocess.run", side_effect=failure), self.assertRaises(subprocess.CalledProcessError):
                decode("input.h264", "nvdec")

    def test__given_existing_backends__when_stderr_is_nonempty__then_strict_rejection_is_preserved(self):
        for backend in ("ffmpeg", "vaapi", "openh264"):
            result = subprocess.CompletedProcess([], 0, b"", b"unexpected stderr")
            with self.subTest(backend=backend), patch("decoder_core.decode.subprocess.run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "unexpected stderr"):
                    decode("input.h264", backend)

    def test__given_nvdec_installation__when_checking_availability__then_executable_helper_and_decoder_are_required(self):
        for executable, plugin_code in ((True, 0), (True, 1), (False, 0)):
            result = subprocess.CompletedProcess([], plugin_code)
            with self.subTest(executable=executable, plugin_code=plugin_code):
                with patch("decoder_core.decode.Path.is_file", return_value=True), patch("decoder_core.decode.os.access", return_value=executable):
                    with patch("decoder_core.decode.shutil.which", return_value="/usr/bin/gst-inspect-1.0"):
                        with patch("decoder_core.decode.subprocess.run", return_value=result) as run:
                            self.assertEqual(available("nvdec"), executable and plugin_code == 0)
                            if executable:
                                run.assert_called_once_with(["gst-inspect-1.0", "nvv4l2decoder"], capture_output=True)
                            else:
                                run.assert_not_called()

    def test__given_missing_inspector_or_unknown_backend__when_checking_availability__then_no_plugin_process_runs(self):
        with patch("decoder_core.decode.shutil.which", return_value=None), patch("decoder_core.decode.subprocess.run") as run:
            self.assertFalse(available("nvdec"))
            run.assert_not_called()
        with patch("decoder_core.decode.shutil.which") as which, patch("decoder_core.decode.subprocess.run") as run:
            self.assertFalse(available("unknown"))
            which.assert_not_called()
            run.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Unknown decoder"):
            decode("input.h264", "unknown")


class BatchDecodeTests(unittest.TestCase):
    def test__given_invalid_selection_or_shape__when_batching__then_no_decoder_process_starts(self):
        cases = (
            ({"selected_frames": []}, "nonempty"),
            ({"selected_frames": [1, 1]}, "strictly increasing"),
            ({"selected_frames": [2, 0]}, "strictly increasing"),
            ({"selected_frames": [-1]}, "out of range"),
            ({"selected_frames": [3]}, "out of range"),
            ({"selected_frames": [True]}, "integer picture"),
            ({"selected_frames": [1.0]}, "integer picture"),
            ({"width": 3}, "even"),
            ({"height": 0}, "positive integer"),
            ({"pictures_per_job": False}, "positive integer"),
        )
        with patch("decoder_core.decode.subprocess.run") as run:
            for changes, message in cases:
                options = dict(width=2, height=2, pictures_per_job=3)
                options.update(changes)
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                    decode_batch(["unused.h264"], **options)
            for paths in ([], "unused.h264", Path("unused.h264")):
                with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "sequence"):
                    decode_batch(paths, width=2, height=2, pictures_per_job=3)
            run.assert_not_called()

    def test__given_two_jobs_and_selective_copy__when_batching__then_one_helper_maps_output_back_to_each_job(self):
        captured = []

        def run(command, **kwargs):
            self.assertEqual(kwargs, {"capture_output": True, "check": True, "timeout": 180})
            captured.append(command)
            self.assertEqual(Path(command[1]).read_bytes(), b"first streamsecond stream")
            self.assertEqual(Path(command[command.index("--frames") + 1]).read_text(), "2\n5\n")
            self.assertEqual(command[command.index("--expected-frames") + 1], "6")
            self.assertEqual(command[command.index("--width") + 1], "2")
            self.assertEqual(command[command.index("--height") + 1], "2")
            Path(command[2]).write_bytes(b"111111222222")
            return subprocess.CompletedProcess(command, 0, NVDEC_STARTUP,
                                               b"NVDEC_DECODE_FRAMES=6\nNVDEC_COPIED_FRAMES=2\n")

        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("one.h264", "two.h264")]
            for path, raw in zip(paths, (b"first stream", b"second stream")):
                path.write_bytes(raw)
            with patch("decoder_core.decode.os.access", return_value=True), \
                    patch("decoder_core.decode._nvdec_helper", return_value=paths[0]), \
                    patch("decoder_core.decode.subprocess.run", side_effect=run):
                result = decode_batch(paths, width=2, height=2, pictures_per_job=3, selected_frames=(2,))
            self.assertEqual([path.read_bytes() for path in paths], [b"first stream", b"second stream"])
        self.assertEqual(len(captured), 1)
        self.assertEqual(result.frames, ((b"111111",), (b"222222",)))
        self.assertEqual(result.selected_frames, (2,))
        self.assertEqual((result.job_count, result.pictures_per_job, result.frame_bytes), (2, 3, 6))
        self.assertEqual((result.decoded_frames, result.copied_frames), (6, 2))

    def test__given_copy_count_diagnostics__when_validating__then_missing_duplicate_or_wrong_counts_fail(self):
        decoded = b"NVDEC_DECODE_FRAMES=6\n"
        copied = b"NVDEC_COPIED_FRAMES=2\n"
        self.assertEqual(_validate_nvdec_diagnostics(
            decoded, copied, expected_decoded=6, expected_copied=2, require_copied=True), 6)
        cases = ((decoded, b"", "copied picture count"),
                 (decoded, copied + copied, "copied picture count"),
                 (decoded, b"NVDEC_COPIED_FRAMES=3\n", "copied picture count mismatch"),
                 (decoded, b"NVDEC_COPIED_FRAMES=0\n", "copied picture count mismatch"),
                 (b"NVDEC_DECODE_FRAMES=5\n", copied, "decoded picture count mismatch"))
        for stdout, stderr, message in cases:
            with self.subTest(stdout=stdout, stderr=stderr), self.assertRaisesRegex(RuntimeError, message):
                _validate_nvdec_diagnostics(stdout, stderr, expected_decoded=6,
                                             expected_copied=2, require_copied=True)
        with self.assertRaisesRegex(RuntimeError, "copied picture count mismatch"):
            _validate_nvdec_diagnostics(decoded, copied)

    def test__given_wrong_output_size__when_helper_claims_success__then_batch_fails(self):
        for raw in (b"", b"12345", b"1234567"):
            def run(command, **kwargs):
                Path(command[2]).write_bytes(raw)
                return subprocess.CompletedProcess(command, 0, b"",
                                                   b"NVDEC_DECODE_FRAMES=3\nNVDEC_COPIED_FRAMES=1\n")

            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "one.h264"
                path.write_bytes(b"input")
                with patch("decoder_core.decode.os.access", return_value=True), \
                        patch("decoder_core.decode._nvdec_helper", return_value=path), \
                        patch("decoder_core.decode.subprocess.run", side_effect=run), \
                        self.assertRaisesRegex(RuntimeError, "byte count mismatch"):
                    decode_batch([path], width=2, height=2, pictures_per_job=3, selected_frames=(2,))

    def test__given_empty_input__when_batching__then_failure_precedes_decoder_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.h264"
            path.touch()
            with patch("decoder_core.decode.subprocess.run") as run, \
                    self.assertRaisesRegex(ValueError, "Empty H.264"):
                decode_batch([path], width=2, height=2, pictures_per_job=3)
            run.assert_not_called()

    def test__given_distinct_independent_jobs__when_full_and_selected_batch_decode__then_all_bytes_match_standalone(self):
        backends = [backend for backend in ("ffmpeg", "nvdec") if available(backend)]
        if not backends:
            self.skipTest("No batch decoder available")
        streams = Path(__file__).resolve().parents[1] / "05-h264-compiler" / "native_ops" / "streams"
        paths = [streams / "saturating-add.h264", streams / "affine.h264", streams / "saturating-add.h264"]
        original = [path.read_bytes() for path in paths]
        for backend in backends:
            with self.subTest(backend=backend):
                standalone = [decode(path, backend) for path in paths]
                full = decode_batch(paths, backend, width=96, height=96, pictures_per_job=3)
                selected = decode_batch(paths, backend, width=96, height=96,
                                        pictures_per_job=3, selected_frames=(0, 2))
                self.assertEqual((full.decoded_frames, full.copied_frames), (9, 9))
                self.assertEqual((selected.decoded_frames, selected.copied_frames),
                                 (9, 6 if backend == "nvdec" else 9))
                self.assertEqual(selected.selected_frames, (0, 2))
                self.assertNotEqual(standalone[0], standalone[1])
                for index, raw in enumerate(standalone):
                    self.assertEqual(len(raw), 3 * full.frame_bytes)
                    self.assertEqual(b"".join(full.frames[index]), raw)
                    self.assertEqual(selected.frames[index], (full.frames[index][0], full.frames[index][2]))
        self.assertEqual([path.read_bytes() for path in paths], original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
