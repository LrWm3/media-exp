import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from codec_compute.workloads.retained_inputs.prepare import (
    FIXTURES, WORKLOADS, compact, compile_sequence, fixture, nal_types, prepare, variants,
)


class RetainedInputExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.helper = Path(cls.directory.name) / "cpu.so"
        source = Path(__file__).parents[1] / "cpu_comparison" / "cpu.cpp"
        subprocess.run(["c++", "-std=c++17", "-O3", "-fPIC", "-shared", str(source),
                        "-o", str(cls.helper)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test__given_production_fields__when_generating_32_variants__then_each_field_pair_is_distinct(self):
        for size in (128, 512):
            for workload in WORKLOADS:
                with self.subTest(size=size, workload=workload):
                    specs = variants(size, workload, 32)
                    self.assertEqual(specs, variants(size, workload, 32))
                    self.assertEqual(len({(s["cpu"]["field_a"], s["cpu"]["field_b"])
                                          for s in specs}), 32)
                    self.assertEqual([s["name"] for s in specs], [f"output_{i:03}" for i in range(32)])

    def test__given_32_named_outputs__when_scheduled__then_original_inputs_survive_beyond_16_pictures(self):
        for workload, policy in product(WORKLOADS, ("dpb16", "tight")):
            with self.subTest(workload=workload, policy=policy):
                program, _ = compile_sequence(32, workload, 32, dpb_policy=policy)
                native = program.native
                inputs = len(program.graph.inputs)
                self.assertEqual(len(native.operations), 32)
                self.assertEqual(list(native.output_frames.values()), list(range(inputs, inputs + 32)))
                self.assertEqual(native.refreshes, [])
                for operation in native.operations:
                    self.assertEqual(tuple(s.frame for s in operation.sources), tuple(range(inputs)))
                    self.assertEqual(native.writer.slice_count(operation), 1)
                self.assertEqual(set(nal_types(native.tail)), {1})
                self.assertEqual(native.reference_slots[inputs + 31], 0)
                self.assertEqual(native.writer.reference_capacity, inputs + 1 if policy == "tight" else 16)

    def test__given_four_fixtures_and_named_outputs__when_exporting_smoke__then_cpu_ir_and_complete_ffmpeg_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = prepare(root, cpu_helper=self.helper, sizes=(32,), output_counts=(1, 8),
                             decoders=("ffmpeg",))
            self.assertEqual(report["status"], "pass")
            self.assertFalse(report["production_matrix"])
            self.assertEqual(report["validation"]["streams"], 32)
            self.assertEqual(report["validation"]["local_decoder_runs"], 32)
            self.assertEqual(report["validation"]["named_outputs"], 144)
            self.assertEqual(report["validation"]["paired_policy_outputs"], 72)
            self.assertEqual(json.loads((root / "bundle.json").read_text())["status"], "pass")
            for case in report["cases"].values():
                program, specs = compile_sequence(32, case["workload"], case["output_count"],
                                                  dpb_policy=case["dpb_policy"])
                self.assertEqual(case["variants"], specs)
                self.assertEqual(case["reference_capacity"],
                                 max(program.native.reference_slots.values()) + 1
                                 if case["dpb_policy"] == "tight" else 16)
                self.assertEqual(case["writer"]["prediction_slice_rows"], 0)
                self.assertEqual([job["fixture"] for job in case["jobs"]], list(FIXTURES))
                for job in case["jobs"]:
                    with self.subTest(workload=case["workload"], count=case["output_count"], fixture=job["fixture"]):
                        images = fixture(32, case["workload"], job["fixture"])
                        reference = compact(program.reference(images))
                        self.assertEqual(set(job["outputs"]), set(reference))
                        for name, record in job["outputs"].items():
                            data = (root / record["path"]).read_bytes()
                            self.assertEqual(data, reference[name])
                            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
                        stream = (root / job["stream"]["path"]).read_bytes()
                        self.assertEqual(stream, program.encode(images))
                        self.assertEqual(nal_types(stream).count(5), 1)
                        self.assertEqual(job["expected"]["bytes"], program.native.pictures * case["frame_bytes"])


if __name__ == "__main__":
    unittest.main()
