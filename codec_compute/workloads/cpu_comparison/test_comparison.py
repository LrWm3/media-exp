import copy
from pathlib import Path
import subprocess
import tempfile
import unittest

import codec_compute as media
from codec_compute.workloads.cpu_comparison.native import CPU
from codec_compute.workloads.cpu_comparison.tasks import tasks
from codec_compute.workloads.resampling.kernels import fixture


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.helper = Path(cls.directory.name)/"cpu.so"
        subprocess.run(["c++", "-std=c++17", "-O3", "-fPIC", "-shared",
                        str(Path(__file__).with_name("cpu.cpp")), "-o", str(cls.helper)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test__given_all_tasks__when_compiled__then_cpu_ir_and_ffmpeg_agree(self):
        for task, specification in tasks(32):
            with self.subTest(task=task.name):
                program = media.compile(task.kernel, width=32, height=32, input_ranges=task.input_ranges)
                images = fixture(task, 1, 32)
                expected = program.reference(images)["output"]
                expected_bytes = bytes(v for row in expected for v in row)
                with CPU(self.helper, 32, specification) as cpu:
                    actual = cpu.run(*(bytes(v for row in images[n] for v in row) for n in task.input_names))
                self.assertEqual(actual, expected_bytes)
                self.assertEqual(program.decode(images)["output"], expected)
                if specification["kind"] == 4:
                    self.assertEqual(program.manifest()["costs"]["computation_pictures"], specification["rounds"])
                    self.assertEqual(len(program.native.output_frames), 1)

    def test__given_short_or_invalid_fields__when_constructing__then_reject_before_native_read(self):
        spec = tasks(32)[3][1]
        for field in ([(0, 0)], [(0, 2**32)]*4, [(True, 0)]*4, [(0,)]*4):
            invalid = copy.deepcopy(spec)
            invalid["field_a"] = field
            with self.subTest(field=field), self.assertRaises(ValueError):
                CPU(self.helper, 32, invalid)

    def test__given_changed_inputs__when_reusing_context__then_no_cached_results(self):
        task, spec = tasks(32)[3]
        program = media.compile(task.kernel, width=32, height=32, input_ranges=task.input_ranges)
        with CPU(self.helper, 32, spec) as cpu:
            results = []
            for index in (0, 1, 2):
                images = fixture(task, index, 32)
                actual = cpu.run(*(bytes(v for row in images[n] for v in row) for n in task.input_names))
                self.assertEqual(actual, bytes(v for row in program.reference(images)["output"] for v in row))
                results.append(actual)
            self.assertEqual(len(set(results)), 3)


if __name__ == "__main__":
    unittest.main()
