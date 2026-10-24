"""Prevent mislabeled CPU baselines in the paired measurement adapter."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from .build import build
from .prepare import prepare
from .runtime import Corpus


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        cls.build_root, cls.bundle = root/"build", root/"bundle"
        with contextlib.redirect_stdout(io.StringIO()):
            build(cls.build_root)
            prepare(cls.bundle, cpu_helper=cls.build_root/"libcpu.so", size=32, depths=(1,))
        cls.build_info = json.loads((cls.build_root/"build.json").read_text())

    def corpus(self, record=None):
        path = self.build_root/"candidate.json"
        path.write_text(json.dumps(self.build_info if record is None else record))
        # This suite checks CPU/PCM/framing only. No decoder worker is started;
        # the executable is just a real file for the binary identity record.
        return Corpus(self.bundle, cpu_helper=self.build_root/"libcpu.so",
                      pcm_helper=self.build_root/"libpcm.so", nvdec_helper=sys.executable, build_info=path)

    def test__given_paired_export__when_running_tuned_cpu__then_all_inputs_match_exact_outputs(self):
        corpus = self.corpus()
        try:
            self.assertEqual(len(corpus.groups), 1)
            for key in corpus.workloads:
                for index in range(8):
                    corpus.cpu(key, index)
            corpus.verify_stable()
        finally:
            corpus.close()

    def test__given_generic_cpu_build_record__when_loading__then_reject_mislabeled_baseline(self):
        record = json.loads(json.dumps(self.build_info))
        checksum = record["source_sha256"].pop("codec_compute/workloads/cpu_comparison/cpu_planes.cpp")
        record["source_sha256"]["codec_compute/workloads/cpu_comparison/cpu.cpp"] = checksum
        with self.assertRaisesRegex(AssertionError, "shared-filter CPU"):
            self.corpus(record)

    def test__given_portable_or_overridden_flags__when_loading__then_reject_untuned_baseline(self):
        for alteration in ("portable", "override"):
            record = json.loads(json.dumps(self.build_info))
            flags = record["flags"]["cpu"]
            if alteration == "portable":
                flags.remove("-funroll-loops")
            else:
                flags.append("-O0")
            with self.subTest(alteration=alteration), self.assertRaisesRegex(AssertionError, "tuned native/unrolled"):
                self.corpus(record)


if __name__ == "__main__":
    unittest.main()
