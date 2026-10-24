"""Exact tests for the fixed-field plane-filter recurrence CPU specialization."""

import ctypes as C
import importlib.util
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from codec_compute.sampling import interpolate_h264
from .native import CPU


HERE = Path(__file__).resolve().parent


def specification(size, rounds=1):
    count = (size//16)**2
    return {"kind": 4, "rounds": rounds, "field_a": [(1, 2)]*count, "field_b": [(-1, -2)]*count}


def oracle(plane, size, rounds, sample=interpolate_h264):
    for _ in range(rounds):
        plane = bytes((sample(plane, size, size, 4*x+1, 4*y+2)+
                       sample(plane, size, size, 4*x-1, 4*y-2)+1)//2
                      for y in range(size) for x in range(size))
    return plane


def random_plane(size, seed):
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size*size))


class PlaneKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get("CXX", "c++")
        if not shutil.which(compiler):
            raise unittest.SkipTest("A C++17 compiler is required")
        cls.temporary = tempfile.TemporaryDirectory(prefix="media-exp-planes-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.libraries = {}
        for name, source in (("planes", "cpu_planes.cpp"), ("generic", "cpu.cpp")):
            path = Path(cls.temporary.name)/(name+".so")
            subprocess.run([compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-Wpedantic", "-fPIC",
                            "-shared", str(HERE/source), "-o", str(path)], check=True, capture_output=True)
            cls.libraries[name] = path
        legacy_path = HERE.parents[2]/"05-h264-compiler/h264/verify.py"
        module = importlib.util.spec_from_file_location("_planes_legacy_verify", legacy_path)
        legacy = importlib.util.module_from_spec(module)
        module.loader.exec_module(legacy)
        cls.legacy = staticmethod(legacy.interpolate)

    def context(self, size, rounds=1, implementation="planes"):
        result = CPU(self.libraries[implementation], size, specification(size, rounds))
        self.addCleanup(result.close)
        return result

    def test__given_nonzero_boundary__when_filtering_outside_half_samples__then_negative_corner_is_preserved(self):
        size = 16
        plane = bytes([255])*(size*size)
        # Zero integer extension still produces nonzero filtered half samples
        # at (-0.5,-0.5) and (0,-0.5). Clipping the half planes to the active
        # rectangle would incorrectly turn this output corner into 128.
        self.assertEqual(interpolate_h264(plane, size, size, -2, -2), 64)
        self.assertEqual(interpolate_h264(plane, size, size, 0, -2), 128)
        self.assertEqual(interpolate_h264(plane, size, size, -1, -2), 96)
        actual = self.context(size).run(plane)
        self.assertEqual(actual[0], 176)
        self.assertEqual(actual, oracle(plane, size, 1))
        self.assertEqual(actual, oracle(plane, size, 1, self.legacy))

    def test__given_extremes_and_border_impulses__when_running_one_round__then_both_scalar_oracles_match(self):
        size = 16
        fixtures = [bytes(size*size), bytes([255])*(size*size), bytes(range(256)), random_plane(size, 471),
                    bytes(255*((x+y) % 2) for y in range(size) for x in range(size)),
                    bytes(255*(x < 3 or y < 3) for y in range(size) for x in range(size)),
                    bytes(255*(x >= size-3 or y >= size-3) for y in range(size) for x in range(size))]
        for px, py in ((0, 0), (0, size-1), (size-1, 0), (size-1, size-1), (1, 2), (8, 8)):
            fixtures.append(bytes(255*(x == px and y == py) for y in range(size) for x in range(size)))
        cpu = self.context(size)
        for index, plane in enumerate(fixtures):
            with self.subTest(fixture=index):
                expected = oracle(plane, size, 1)
                self.assertEqual(expected, oracle(plane, size, 1, self.legacy))
                self.assertEqual(cpu.run(plane), expected)

    def test__given_odd_half_sample_sums__when_blending_twice__then_single_rounding_cannot_replace_nested_rounding(self):
        plane = random_plane(16, 471)
        half = [interpolate_h264(plane, 16, 16, x, y) for x, y in ((2, 2), (0, 2), (-2, -2), (0, -2))]
        self.assertEqual(half, [146, 127, 0, 0])
        self.assertEqual((sum(half)+2)//4, 68)
        self.assertEqual(self.context(16).run(plane)[0], 69)

    def test__given_multistage_recurrence__when_running_up_to_64_rounds__then_every_nested_rounding_is_retained(self):
        for size, rounds in ((16, 2), (32, 3), (16, 16), (16, 64)):
            cpu = self.context(size, rounds)
            for plane in (random_plane(size, 812), bytes([255])*(size*size)):
                with self.subTest(size=size, rounds=rounds, constant=len(set(plane)) == 1):
                    expected = oracle(plane, size, rounds)
                    self.assertEqual(cpu.run(plane), expected)
                    self.assertEqual(expected, oracle(plane, size, rounds, self.legacy))

    def test__given_benchmark_geometries_and_long_chains__when_compared_with_generic_kernel__then_outputs_are_identical(self):
        for size, rounds in ((16, 1), (32, 3), (128, 16), (256, 64), (512, 64)):
            specialized, generic = self.context(size, rounds), self.context(size, rounds, "generic")
            fixtures = (bytes(size*size), bytes([255])*(size*size),
                        bytes(range(256))*(size*size//256), random_plane(size, 912))
            for index, plane in enumerate(fixtures):
                with self.subTest(size=size, rounds=rounds, fixture=index):
                    self.assertEqual(specialized.run(plane), generic.run(plane))

    def test__given_reused_context_and_mutated_field_storage__when_changing_inputs__then_no_results_or_caller_fields_are_reused(self):
        cpu = self.context(16, 3)
        library = cpu.library
        field_a, field_b = (C.c_int32*2)(1, 2), (C.c_int32*2)(-1, -2)
        context = library.cpu_create(16, 4, 3, field_a, field_b)
        self.assertTrue(context)
        self.addCleanup(library.cpu_destroy, context)
        field_a[0], field_b[1] = 2**31-1, -(2**31)
        for plane in (bytes([255])*256, random_plane(16, 9), bytes(256), bytes(range(256))):
            expected = oracle(plane, 16, 3)
            output = C.create_string_buffer(b"\xa5"*288, 288)
            self.assertEqual(library.cpu_execute(context, plane, None, C.byref(output, 16)), 0)
            self.assertEqual(output.raw[:16]+output.raw[-16:], b"\xa5"*32)
            self.assertEqual(output.raw[16:-16], expected)
            # Input/output may alias because input is copied before any result.
            inplace = C.create_string_buffer(plane, len(plane))
            self.assertEqual(library.cpu_execute(context, inplace, None, inplace), 0)
            self.assertEqual(inplace.raw, expected)

    def test__given_partial_input_output_overlap__when_executing__then_all_input_bytes_are_read_before_output(self):
        cpu = self.context(16, 2)
        plane = random_plane(16, 38)
        storage = C.create_string_buffer(b"\xa5"*16+plane+b"\xa5"*32, 304)
        self.assertEqual(cpu.library.cpu_execute(cpu.context, C.byref(storage, 16), None, C.byref(storage, 24)), 0)
        self.assertEqual(storage.raw[24:280], oracle(plane, 16, 2))
        self.assertEqual(storage.raw[:16]+storage.raw[280:], b"\xa5"*40)

    def test__given_unsupported_kind_geometry_rounds_or_fields__when_creating__then_abi_rejects_without_running(self):
        cpu = self.context(16)
        library = cpu.library
        a, b = (C.c_int32*2)(1, 2), (C.c_int32*2)(-1, -2)
        cases = [(0, 4, 1, a, b), (15, 4, 1, a, b), (17, 4, 1, a, b),
                 (4112, 4, 1, a, b), (2**31-1, 4, 1, a, b), (16, 4, 0, a, b),
                 (16, 4, 4097, a, b), (16, 4, 1, None, b), (16, 4, 1, a, None)]
        cases += [(16, kind, 1, a, b) for kind in (-1, 0, 1, 2, 3, 5)]
        for args in cases:
            with self.subTest(arguments=args[:3]):
                self.assertFalse(library.cpu_create(*args))
                self.assertTrue(library.cpu_error())
        # Every block must match, including the final block and both axes.
        for field_index in (0, 1):
            for component in range(8):
                fields = [(C.c_int32*8)(*[1, 2]*4), (C.c_int32*8)(*[-1, -2]*4)]
                fields[field_index][component] += 1
                with self.subTest(field=field_index, component=component):
                    self.assertFalse(library.cpu_create(32, 4, 1, *fields))
                    self.assertIn(b"every block", library.cpu_error())
        # Upper limits are accepted without executing their expensive product.
        upper_rounds = library.cpu_create(16, 4, 4096, a, b)
        self.assertTrue(upper_rounds)
        library.cpu_destroy(upper_rounds)
        with CPU(self.libraries["planes"], 4096, specification(4096)):
            pass

    def test__given_null_execution_arguments__when_called__then_errors_are_reported_and_later_valid_calls_succeed(self):
        cpu = self.context(16)
        library = cpu.library
        out = C.create_string_buffer(256)
        for context, plane, output in ((None, bytes(256), out), (cpu.context, None, out),
                                        (cpu.context, bytes(256), None)):
            self.assertEqual(library.cpu_execute(context, plane, None, output), -1)
            self.assertTrue(library.cpu_error())
        self.assertEqual(cpu.run(bytes(256)), bytes(256))
        self.assertEqual(library.cpu_error(), b"")
        library.cpu_destroy(None)


if __name__ == "__main__":
    unittest.main()
