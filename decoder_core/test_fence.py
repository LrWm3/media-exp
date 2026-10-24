"""Constant IDR grammar and exact decoder output, without live-timing claims.

H.264 (08/2024) 7.3.5, Table 7-11, 8.3.3.3 and Table 9-5 define the
DC/no-residual construction. Consecutive IDR identifiers differ per 7.4.3.
Two pictures also avoid the observed generic single-AU drain failure in
GStreamer OpenH264 1.24.2 / libopenh264 2.4.1; ordinary PCM shares that failure.
"""

from pathlib import Path
import tempfile
import unittest

from decoder_core.bitstream import Writer
from decoder_core.decode import available, decode
from decoder_core.fence import neutral_idr


def units(stream):
    return stream.split(b"\x00\x00\x00\x01")[1:]


class BitReader:
    def __init__(self, ebsp):
        rbsp, zeros = bytearray(), 0
        for value in ebsp:
            if zeros == 2 and value == 3:
                zeros = 0
                continue
            rbsp.append(value)
            zeros = zeros+1 if value == 0 else 0
        self.bits = "".join(f"{value:08b}" for value in rbsp)
        self.position = 0

    def u(self, width=1):
        bits = self.bits[self.position:self.position+width]
        if len(bits) != width:
            raise ValueError("Truncated test bitstream")
        self.position += width
        return int(bits, 2)

    def ue(self):
        zeros = 0
        while self.u() == 0:
            zeros += 1
        return (1 << zeros)-1+(self.u(zeros) if zeros else 0)

    def se(self):
        value = self.ue()
        return (value+1)//2 if value % 2 else -(value//2)


def idr_fields(unit):
    bits = BitReader(unit[1:])
    fields = {"first_mb": bits.ue(), "slice_type": bits.ue(), "pps": bits.ue(),
              "frame_num": bits.u(16), "idr_pic_id": bits.ue(), "poc": bits.u(16),
              "no_output_prior": bits.u(), "long_term": bits.u(),
              "qp_delta": bits.se(), "disable_deblocking": bits.ue()}
    return bits, fields


def ordinary_picture(writer, board):
    """Independent expected I420, including zero luma guards and neutral chroma."""
    luma = bytearray(writer.width*writer.height)
    for y, row in enumerate(board, 16):
        luma[y*writer.width+16:y*writer.width+16+len(row)] = bytes(row)
    return bytes(luma)+bytes([128])*(writer.width*writer.height//2)


class FenceSyntaxTests(unittest.TestCase):
    def test__given_fixed_geometry_and_reference_modes__when_emitting_fence__then_idr_flags_and_zero_coefficient_grammar_are_exact(self):
        for size, level in ((128, 31), (256, 41)):
            for slots in (None, {0: 0}):
                writer = Writer(size, size, slice_mode="row", reference_slots=slots, level_idc=level)
                for identifier in (1, 2, 65535):
                    with self.subTest(size=size, slots=slots, identifier=identifier):
                        stream = neutral_idr(writer, idr_pic_id=identifier)
                        nals = units(stream)
                        self.assertEqual([unit[0] & 31 for unit in nals], [7, 8, 5])
                        self.assertEqual(nals[0][3], level)
                        self.assertEqual(nals[-1][0] >> 5, 3)
                        bits, fields = idr_fields(nals[-1])
                        self.assertEqual(fields, {"first_mb": 0, "slice_type": 2, "pps": 0,
                                                 "frame_num": 0, "idr_pic_id": identifier, "poc": 0,
                                                 "no_output_prior": 0, "long_term": int(slots is not None),
                                                 "qp_delta": 0, "disable_deblocking": 1})
                        for _ in range(writer.mb_width*writer.mb_height):
                            self.assertEqual(bits.ue(), 3)  # Intra16 DC, both coded patterns zero.
                            self.assertEqual(bits.ue(), 0)  # Chroma DC.
                            self.assertEqual(bits.se(), 0)  # Mandatory QP delta.
                            self.assertEqual(bits.u(), 1)   # TotalCoeff=0, nC=0.
                        self.assertEqual(bits.u(), 1)       # RBSP stop, then only alignment zeros.
                        self.assertEqual(set(bits.bits[bits.position:]) - {"0"}, set())
                        self.assertLessEqual(len(bits.bits)-bits.position, 7)
                        self.assertLess(len(stream), writer.mb_width*writer.mb_height+80)
                        self.assertEqual(stream, neutral_idr(writer, idr_pic_id=identifier))
                self.assertEqual(neutral_idr(writer), neutral_idr(writer, idr_pic_id=1))

    def test__given_invalid_idr_identifier__when_emitting_fence__then_integer_range_is_enforced(self):
        writer = Writer(128, 128, slice_mode="row", reference_slots={0: 0})
        for identifier in (True, False, -1, 65536, 1.5, "1", None):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "idr_pic_id"):
                neutral_idr(writer, idr_pic_id=identifier)


class FenceDecoderTests(unittest.TestCase):
    def backends(self):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        return backends

    def test__given_two_fences_with_distinct_idr_identifiers__when_decoding_both_levels__then_every_i420_byte_is_128(self):
        backends = self.backends()
        with tempfile.TemporaryDirectory(prefix="h264-fences-") as directory:
            for size, level in ((128, 31), (256, 41)):
                writer = Writer(size, size, slice_mode="row", reference_slots={0: 0}, level_idc=level)
                first, second = neutral_idr(writer), neutral_idr(writer, idr_pic_id=2)
                self.assertNotEqual(idr_fields(units(first)[-1])[1]["idr_pic_id"],
                                    idr_fields(units(second)[-1])[1]["idr_pic_id"])
                expected = bytes([128])*(writer.width*writer.height*3)
                path = Path(directory)/f"fences-{size}.h264"
                path.write_bytes(first+second)
                for backend in backends:
                    with self.subTest(size=size, backend=backend):
                        self.assertEqual(decode(path, backend), expected)

    def test__given_mixed_pcm_p_b_jobs_with_changed_inputs__when_idr_fences_separate_jobs__then_named_results_and_guards_survive(self):
        import codec_compute as media

        def kernel(a, b):
            moved = media.affine(media.shift(a, 1, -1), bias=-7, saturating=True)
            mixed = media.average(moved, b)
            return {"a": a, "b": b, "moved": moved, "mixed": mixed, "alias": mixed}

        backends = self.backends()
        with tempfile.TemporaryDirectory(prefix="h264-job-fences-") as directory:
            for size, level in ((128, 31), (256, 41)):
                program = media.compile(kernel, width=size, height=size,
                                        input_ranges={"a": (0, 255), "b": (0, 255)}, h264_level=level)
                writer = program.native.writer
                self.assertEqual([len(op.sources) for op in program.native.operations], [1, 2])
                self.assertEqual(program.manifest()["costs"]["computation_pictures"], 2)
                frames = program.manifest()["output_frames"]
                self.assertEqual(frames["mixed"], frames["alias"])
                fence = neutral_idr(writer)
                self.assertEqual(program.native.fence_unit, fence)
                frame_bytes = writer.width*writer.height*3//2
                neutral = bytes([128])*frame_bytes
                stream, expected_frames, named_jobs = bytearray(), [], []
                tail = program.tail
                for variant in (0, 1):
                    a = [[(17*x+31*y+47*variant) % 256 for x in range(size)] for y in range(size)]
                    b = [[(13*x+11*y+83*variant+7) % 256 for x in range(size)] for y in range(size)]
                    moved = [[max(0, a[y-1][x+1]-7) if y > 0 and x+1 < size else 0
                              for x in range(size)] for y in range(size)]
                    mixed = [[(x+y+1)//2 for x, y in zip(ra, rb)] for ra, rb in zip(moved, b)]
                    expected = {"a": a, "b": b, "moved": moved, "mixed": mixed, "alias": mixed}
                    encoded = program.encode({"a": a, "b": b})
                    first_idr = next(unit for unit in units(encoded) if unit[0] & 31 == 5)
                    self.assertEqual(idr_fields(first_idr)[1]["idr_pic_id"], 0)
                    self.assertEqual(idr_fields(units(fence)[-1])[1]["idr_pic_id"], 1)
                    self.assertTrue(encoded.endswith(tail))
                    stream.extend(encoded+fence)
                    expected_frames.extend(ordinary_picture(writer, board) for board in (a, b, moved, mixed))
                    expected_frames.append(neutral)
                    named_jobs.append(expected)
                self.assertEqual(program.tail, tail)
                self.assertNotEqual(named_jobs[0]["mixed"], named_jobs[1]["mixed"])
                path = Path(directory)/f"jobs-{size}.h264"
                path.write_bytes(stream)
                for backend in backends:
                    with self.subTest(size=size, backend=backend):
                        raw = decode(path, backend)
                        self.assertEqual(raw, b"".join(expected_frames))
                        for job, expected in enumerate(named_jobs):
                            for name, board in expected.items():
                                offset = (job*5+frames[name])*frame_bytes
                                actual = [list(raw[offset+(y+16)*writer.width+16:
                                                   offset+(y+16)*writer.width+16+size]) for y in range(size)]
                                self.assertEqual(actual, board, (job, name))


if __name__ == "__main__":
    unittest.main()
