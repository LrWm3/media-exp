"""Explicit DPB declarations preserve pictures and bound reference use."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode
from decoder_core.pcm import NativePCMWriter
from decoder_core.test_block_predictions import i420_picture
from decoder_core.test_guard_fusion import predict


class Reader:
    def __init__(self, data):
        self.bits = [(byte >> bit) & 1 for byte in data for bit in reversed(range(8))]
        self.index = 0

    def u(self, width=1):
        result = 0
        for _ in range(width):
            result = 2*result+self.bits[self.index]
            self.index += 1
        return result

    def ue(self):
        zeros = 0
        while self.u() == 0:
            zeros += 1
        return (1 << zeros)-1+self.u(zeros)


def sps_capacities(writer):
    escaped = writer.parameter_sets().split(b"\0\0\0\1")[1][1:]
    rbsp, zeros = bytearray(), 0
    for byte in escaped:
        if zeros == 2 and byte == 3:
            zeros = 0
            continue
        rbsp.append(byte)
        zeros = zeros+1 if byte == 0 else 0
    r = Reader(rbsp)
    r.u(24)  # profile, constraints, level
    for _ in range(4):  # SPS id, chroma format, luma/chroma bit depths
        r.ue()
    r.u(2)  # bypass and scaling flags
    for _ in range(3):  # frame_num, POC type and POC width
        r.ue()
    references = r.ue()
    r.u(); r.ue(); r.ue()  # gaps, coded width/height
    r.u(4)  # frame-only, direct inference, no crop, VUI present
    r.u(3)  # no aspect ratio, no overscan, signal type present
    r.u(5)  # format, full range, no colour description
    r.u(2)  # no chroma location, timing present
    r.u(65)  # timing numerator, denominator and fixed-rate flag
    r.u(5)  # no HRDs/pic_struct, restrictions and boundary motion enabled
    for _ in range(4):
        r.ue()  # bytes/picture, bits/MB and horizontal/vertical motion bounds
    reorder = r.ue()
    return references, reorder, r.ue()


class ReferenceCapacityTests(unittest.TestCase):
    def assert_decoded(self, streams, pictures):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        expected = b"".join(i420_picture(board) for board in pictures)
        with tempfile.TemporaryDirectory() as directory:
            for label, stream in streams.items():
                path = Path(directory)/(str(label)+".h264")
                path.write_bytes(stream)
                for backend in backends:
                    with self.subTest(capacity=label, decoder=backend):
                        self.assertEqual(decode(path, backend), expected)

    def test__given_default_or_explicit_sixteen__when_encoding__then_historical_bytes_remain_identical(self):
        board = [[16*y+x for x in range(16)] for y in range(16)]
        default = Writer(16, 16)
        self.assertEqual(hashlib.sha256(default.seed_luma(board)).hexdigest(),
                         "bc8f01c02af464b79488ad7dc87d3f874bbbf0aa2fcea15eafa63326c0a276ec")
        operations = (Operation("p", (Source(0, qx=1, qy=-2),)),
                      Operation("b", (Source(0, weight=3), Source(1, weight=-1)), 1))
        for slots in (None, {0: 0, 1: 1, 2: 0}):
            baseline = Writer(16, 16, reference_slots=slots, slice_mode="row")
            explicit = Writer(16, 16, reference_slots=slots, slice_mode="row", reference_capacity=16)
            self.assertEqual(baseline.seed_luma(board), explicit.seed_luma(board))
            for frame, op in enumerate(operations, 1):
                self.assertEqual(baseline.prediction(frame, op), explicit.prediction(frame, op))
        self.assertEqual(Writer.MAX_REFS, 16)

    def test__given_each_legal_capacity__when_serializing_sps__then_both_dpb_fields_match(self):
        for capacity in range(1, 17):
            with self.subTest(capacity=capacity):
                writer = Writer(16, 16, reference_capacity=capacity)
                self.assertEqual(sps_capacities(writer), (capacity, 0, capacity))

    def test__given_invalid_capacity_or_undersized_plan__when_constructing__then_writer_rejects(self):
        for capacity in (None, True, False, 0, -1, 17, 1.0, "2", [], {}):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, "reference_capacity"):
                Writer(16, 16, reference_capacity=capacity)
        for capacity, slots in ((1, {0: 0, 1: 1}), (2, {0: 0, 1: 0, 2: 2}),
                                (3, {0: 0, 1: 7})):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, "reference_capacity"):
                Writer(16, 16, reference_capacity=capacity, reference_slots=slots)

    def test__given_smaller_capacity__when_checking_levels_and_sliding_age__then_actual_capacity_applies(self):
        with self.assertRaisesRegex(ValueError, "decoded-picture buffer"):
            Writer(512, 512, level_idc=31, slice_mode="row", fps=15)
        writer = Writer(512, 512, level_idc=31, slice_mode="row", fps=15, reference_capacity=1)
        # Smaller DPB does not waive the independent initial PCM access-unit cap.
        with self.assertRaisesRegex(ValueError, "initial access-unit"):
            writer.seed_luma([[0]*512 for _ in range(512)])
        for capacity in (1, 2, 3, 16):
            writer = Writer(16, 16, reference_capacity=capacity)
            writer.prediction(capacity, Operation("oldest legal", (Source(0),)))
            with self.assertRaisesRegex(ValueError, "outside DPB"):
                writer.prediction(capacity+1, Operation("expired", (Source(0),)))

    def test__given_one_slot_recurrence__when_decoding_thirty_two_predictions__then_p_and_b_replacement_stays_exact(self):
        pictures = [[[((37*x+71*y) % 256) for x in range(16)] for y in range(16)]]
        operations = []
        for frame in range(1, 33):
            sources = (Source(frame-1, qx=(frame % 4)-2, qy=(frame//4 % 4)-2),)
            if frame % 2 == 0:
                sources += (Source(frame-1, qx=1, qy=-1),)
            op = Operation("recurrence", sources)
            operations.append(op)
            pictures.append(predict(pictures, op))
        streams = {}
        for slots in (None, {frame: 0 for frame in range(33)}):
            for capacity in (1, 16):
                writer = Writer(16, 16, reference_capacity=capacity, reference_slots=slots,
                                slice_mode="row", merge_zero_guards=True, prediction_slice_rows=0)
                streams[str(slots is None)+"-"+str(capacity)] = writer.seed_luma(pictures[0])+b"".join(
                    writer.prediction(frame, op) for frame, op in enumerate(operations, 1))
        self.assert_decoded(streams, pictures)

    def test__given_old_input_retained_beyond_sixteen_frames__when_replacing_two_slots__then_complete_p_b_pictures_match(self):
        pictures = [[[((37*x+71*y+seed) % 256) for x in range(16)] for y in range(16)]
                    for seed in (0, 19)]
        operations = []
        for frame in range(2, 34):
            op = (Operation("retained average", (Source(0, qx=1), Source(frame-1, qy=-2)))
                  if frame % 2 else Operation("increment", (Source(frame-1, offset=1),)))
            operations.append(op)
            pictures.append(predict(pictures, op))
        slots = {0: 0, **{frame: 1 for frame in range(1, 34)}}
        streams = {}
        for capacity in (2, 3, 16):
            writer = Writer(16, 16, reference_slots=slots, reference_capacity=capacity, slice_mode="row")
            streams[capacity] = b"".join(writer.input_luma(board, frame)
                                        for frame, board in enumerate(pictures[:2]))+b"".join(
                writer.prediction(frame, op) for frame, op in enumerate(operations, 2))
        self.assert_decoded(streams, pictures)

    def test__given_small_capacity_native_pcm__when_preparing_reused_templates__then_prefixes_equal_python_and_mutation_rejects(self):
        if not shutil.which(os.environ.get("CXX", "c++")):
            self.skipTest("A C++17 compiler is required for optional PCM tests")
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory)/"libpcm.so"
            subprocess.run(["bash", str(Path(__file__).with_name("build_pcm.sh")), str(helper)],
                           check=True, capture_output=True, text=True)
            for capacity in (1, 2, 3, 16):
                writer = Writer(16, 16, reference_capacity=capacity,
                                reference_slots={frame: frame % capacity for frame in range(34)})
                native = NativePCMWriter(writer, helper=helper)
                for frame in (0, 1, 17, 33):
                    for plane in (bytes(256), bytes(range(256)), bytes([255])*256):
                        board = [list(plane[y*16:(y+1)*16]) for y in range(16)]
                        self.assertEqual(native.input_luma(plane, frame), writer.input_luma(board, frame))
                writer.reference_capacity = 1 if capacity != 1 else 2
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    native.input_luma(bytes(256), 0)


if __name__ == "__main__":
    unittest.main()
