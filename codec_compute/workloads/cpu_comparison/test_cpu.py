"""Exact native-kernel tests; semantic oracle does not call target lowering.

Run: python3 -m unittest codec_compute.workloads.cpu_comparison.test_cpu -v
The shared library is compiled once into a temporary directory. No decoder,
hardware helper, external package, or benchmark measurement is required.
"""

import ctypes as C
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from codec_compute.sampling import interpolate_h264


HERE = Path(__file__).resolve().parent


def gather(plane, size, field, quarter=False):
    """Scalar zero-extension oracle with independent coordinate arithmetic."""
    result = bytearray(size*size)
    for y in range(size):
        for x in range(size):
            dx, dy = field[(y//16)*(size//16)+x//16]
            if quarter:
                value = interpolate_h264(plane, size, size, 4*x+dx, 4*y+dy)
            else:
                sx, sy = x+dx, y+dy
                value = plane[sy*size+sx] if 0 <= sx < size and 0 <= sy < size else 0
            result[y*size+x] = value
    return bytes(result)


def blend(a, b):
    return bytes((x+y+1)//2 for x, y in zip(a, b))


def random_plane(size, seed):
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size*size))


class CPUKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="media-exp-cpu-test-")
        library = Path(cls.temporary.name)/"libcpu.so"
        subprocess.run([str(HERE/"build_cpu.sh"), str(library)], check=True, capture_output=True)
        cls.library = C.CDLL(str(library))
        cls.library.cpu_create.argtypes = [C.c_int, C.c_int, C.c_int,
                                          C.POINTER(C.c_int32), C.POINTER(C.c_int32)]
        cls.library.cpu_create.restype = C.c_void_p
        cls.library.cpu_execute.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        cls.library.cpu_execute.restype = C.c_int
        cls.library.cpu_destroy.argtypes = [C.c_void_p]
        cls.library.cpu_destroy.restype = None
        cls.library.cpu_error.argtypes = []
        cls.library.cpu_error.restype = C.c_char_p

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def create(self, size, kind, a=(), b=(), rounds=1):
        def field(values, needed):
            self.assertEqual(len(values), (size//16)**2 if needed else 0)
            return (C.c_int32*(2*len(values)))(*(v for pair in values for v in pair)) if values else None
        context = self.library.cpu_create(size, kind, rounds, field(a, kind >= 2), field(b, kind >= 3))
        self.assertTrue(context, self.library.cpu_error().decode())
        self.addCleanup(self.library.cpu_destroy, context)
        return context

    def execute(self, context, a, b=None):
        # Sentinel bytes check the compact output contract, including blocks
        # whose source rectangle lies entirely beyond the padded image.
        out = C.create_string_buffer(b"\xA5"*(len(a)+32), len(a)+32)
        status = self.library.cpu_execute(context, a, b, C.byref(out, 16))
        self.assertEqual(status, 0, self.library.cpu_error().decode())
        self.assertEqual(out.raw[:16]+out.raw[-16:], b"\xA5"*32)
        return out.raw[16:-16]

    def test__given_independent_inputs__when_copying_and_averaging__then_exact_bytes(self):
        size = 64
        copy, average = self.create(size, 0), self.create(size, 1)
        fixtures = [(random_plane(size, 1), random_plane(size, 2)),
                    (bytes(range(256))*16, bytes(reversed(range(256)))*16),
                    (bytes(size*size), b"\xff"*(size*size))]
        for a, b in fixtures:
            self.assertEqual(self.execute(copy, a), a)
            self.assertEqual(self.execute(average, a, b), blend(a, b))

    def test__given_integer_fields__when_gathering__then_zero_extended_block_oracle(self):
        size = 64
        fields = [(0, 0), (-3, 2), (5, -7), (16, 16), (-17, -17),
                  (63, 0), (-64, 0), (0, 65), (0, -66),
                  (2**31-1, 0), (0, -(2**31)), (-18, 1),
                  (1, -18), (size+1, size+1), (-19, -19), (31, -31)]
        context = self.create(size, 2, fields)
        for seed in (0, 98):
            plane = random_plane(size, seed)
            self.assertEqual(self.execute(context, plane), gather(plane, size, fields))

    def test__given_all_fractional_phases__when_blending__then_independent_h264_oracle(self):
        size = 32
        a, b = random_plane(size, 400), random_plane(size, 701)
        for fy in range(4):
            for fx in range(4):
                # Distinct sources and phases; cross both active-image borders
                # and block boundaries, with negative floor division.
                field_a = [(-8+fx, -4+fy), (4+fx, -8+fy),
                           (-4+fx, 8+fy), (8+fx, 4+fy)]
                field_b = [(7-fx, 11-fy), (-9-fx, 7-fy),
                           (11-fx, -5-fy), (-5-fx, -9-fy)]
                context = self.create(size, 3, field_a, field_b)
                with self.subTest(phase=(fx, fy)):
                    expected = blend(gather(a, size, field_a, True), gather(b, size, field_b, True))
                    self.assertEqual(self.execute(context, a, b), expected)

    def test__given_extreme_samples_and_support_edges__when_filtering__then_signed_precision(self):
        size = 16
        checker = bytes(255*((x+y) % 2) for y in range(size) for x in range(size))
        edge = bytes(255*(x >= 7 and y >= 7) for y in range(size) for x in range(size))
        impulse = bytes(255*(x == 7 and y == 7) for y in range(size) for x in range(size))
        for fy in range(4):
            for fx in range(4):
                field_a, field_b = [(fx-4, fy-4)], [(fx, fy)]
                context = self.create(size, 3, field_a, field_b)
                for a, b in ((checker, edge), (impulse, b"\xff"*(size*size))):
                    with self.subTest(phase=(fx, fy), impulse=a == impulse):
                        expected = blend(gather(a, size, field_a, True), gather(b, size, field_b, True))
                        self.assertEqual(self.execute(context, a, b), expected)
        # Last partially overlapping six-tap support and completely absent
        # support, including int32 extremes, exercise halo classification.
        for offset in (-2**31, -77, -74, -73, -72, 62, 65, 67, 68, 2**31-1):
            for field in ([(offset, 2)], [(2, offset)], [(offset, offset)]):
                context = self.create(size, 3, field, field)
                self.assertEqual(self.execute(context, edge, edge), gather(edge, size, field, True))

    def test__given_iterative_fields__when_running_native_chain__then_every_round_matches(self):
        for size, rounds in ((32, 1), (32, 3), (32, 8), (16, 64)):
            field_a = [(-3, 2), (7, -1), (-6, 5), (2, -7)][:((size//16)**2)]
            field_b = [(2, -1), (-5, 6), (3, -2), (-1, 3)][:((size//16)**2)]
            context = self.create(size, 4, field_a, field_b, rounds)
            for seed in (81, 9):
                original = random_plane(size, seed)
                expected = original
                for _ in range(rounds):
                    expected = blend(gather(expected, size, field_a, True),
                                     gather(expected, size, field_b, True))
                with self.subTest(size=size, rounds=rounds, seed=seed):
                    self.assertEqual(self.execute(context, original), expected)

    def test__given_fixed_fields__when_reusing_context__then_inputs_refresh_and_fields_stay_cached(self):
        size = 16
        field = (C.c_int32*2)(2, -3)
        context = self.library.cpu_create(size, 3, 1, field, field)
        self.assertTrue(context)
        self.addCleanup(self.library.cpu_destroy, context)
        field[0], field[1] = 2**31-1, -(2**31)
        for a, b in ((b"\xff"*256, bytes(256)), (random_plane(size, 4), random_plane(size, 5)),
                     (bytes(256), bytes(256)), (b"\xff"*256, b"\xff"*256)):
            expected = blend(gather(a, size, [(2, -3)], True), gather(b, size, [(2, -3)], True))
            self.assertEqual(self.execute(context, a, b), expected)
            in_place = C.create_string_buffer(a, len(a))
            self.assertEqual(self.library.cpu_execute(context, in_place, b, in_place), 0)
            self.assertEqual(in_place.raw, expected)

    def test__given_invalid_native_arguments__when_called__then_error_without_memory_access(self):
        for size, kind, rounds in ((0, 0, 1), (-16, 0, 1), (15, 0, 1), (17, 0, 1),
                                   (2**31-1, 0, 1), (16, -1, 1), (16, 5, 1),
                                   (16, 4, 0), (16, 4, 4097), (16, 2, 1), (16, 3, 1)):
            with self.subTest(size=size, kind=kind, rounds=rounds):
                self.assertFalse(self.library.cpu_create(size, kind, rounds, None, None))
                self.assertTrue(self.library.cpu_error())
        self.assertEqual(self.library.cpu_execute(None, None, None, None), -1)
        copy, average = self.create(16, 0), self.create(16, 1)
        out = C.create_string_buffer(256)
        for context, a, b, target in ((copy, None, None, out), (copy, bytes(256), None, None),
                                      (average, bytes(256), None, out)):
            self.assertEqual(self.library.cpu_execute(context, a, b, target), -1)
            self.assertIn(b"pointer", self.library.cpu_error())
        self.assertEqual(self.execute(copy, bytes(256)), bytes(256))
        self.assertEqual(self.library.cpu_error(), b"")
        self.library.cpu_destroy(None)


if __name__ == "__main__":
    unittest.main()
