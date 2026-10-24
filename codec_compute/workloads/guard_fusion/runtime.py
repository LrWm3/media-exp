"""Portable exact paired workloads; no IR compiler or host evaluator imports."""

import importlib.util
import json
from pathlib import Path
import platform
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


comparison = module("_guard_comparison", HERE.parent/"cpu_comparison/benchmark.py")
common, resident, native = comparison.common, comparison.resident, comparison.native
Accounting, measurement, summarize, counts = (comparison.Accounting, comparison.measurement,
                                              comparison.summarize, comparison.counts)
from decoder_core.pcm import NativePCMWriter
from decoder_core.session import NVDECSession


class Corpus:
    def __init__(self, root, *, cpu_helper, pcm_helper, nvdec_helper, build_info):
        self.root = Path(root).resolve()
        self.helpers = {key: Path(path).resolve() for key, path in
                        (("cpu", cpu_helper), ("pcm", pcm_helper), ("nvdec", nvdec_helper))}
        data, self.bundle, self.workloads = resident.load(self.root)
        bundle = self.bundle
        if bundle.get("experiment") != "guard-fusion" or bundle["jobs"] != 8:
            raise ValueError("Expected an eight-fixture guard-fusion export")
        self.checked_sources = ("decoder_core/bitstream.py", "decoder_core/pcm.py", "decoder_core/pcm_native.cpp",
                     "decoder_core/session.py", "decoder_core/decode.py", "decoder_core/nvdec_decode.cpp",
                     "codec_compute/workloads/resident/benchmark.py", "codec_compute/workloads/temporal_benchmark.py",
                     "codec_compute/workloads/cpu_comparison/benchmark.py", "codec_compute/workloads/cpu_comparison/native.py",
                     "codec_compute/workloads/cpu_comparison/cpu_planes.cpp",
                     "codec_compute/workloads/guard_fusion/runtime.py")
        for name in self.checked_sources:
            if common.digest((REPOSITORY/name).read_bytes()) != bundle["source_sha256"].get(name):
                raise AssertionError("Frozen runtime source differs: "+name)
        build = json.loads(Path(build_info).read_text())
        expected_sources = {"codec_compute/workloads/cpu_comparison/cpu_planes.cpp", "decoder_core/pcm_native.cpp"}
        if set(build["source_sha256"]) != expected_sources or set(build["helper_sha256"]) != {"cpu", "pcm"}:
            raise AssertionError("This experiment requires the shared-filter CPU and native PCM builds")
        architecture = platform.machine().lower()
        native_flag = "-mcpu=native" if architecture in ("aarch64", "arm64") else "-march=native"
        flags = build["flags"]["cpu"]
        if (not {"-O3", native_flag, "-funroll-loops"}.issubset(flags)
                or any(flag.startswith("-O") and flag != "-O3" for flag in flags)):
            raise AssertionError("This experiment requires the tuned native/unrolled CPU baseline")
        for key in ("cpu", "pcm"):
            if common.digest(self.helpers[key].read_bytes()) != build["helper_sha256"][key]:
                raise AssertionError("Native helper differs from its build record")
        for name, checksum in build["source_sha256"].items():
            if bundle["source_sha256"].get(name) != checksum:
                raise AssertionError("Native build source differs from export")
        self.identity = {"source_commit": bundle["source_commit"], "source_sha256": bundle["source_sha256"],
                         "bundle_sha256": common.digest(data), "native_build": build,
                         "helper_sha256": {key: common.digest(path.read_bytes()) for key, path in self.helpers.items()}}
        Writer = common.writer_class(REPOSITORY/"decoder_core/bitstream.py")
        self.groups = {}
        for key, w in self.workloads.items():
            case = w["case"]
            size, name, variant = key.split("/")
            rounds = int(name.rsplit("-", 1)[1])
            spec = case["cpu"]
            expected_field_count = (case["size"]//16)**2
            if (variant not in ("baseline", "merged") or int(size) != case["size"]
                    or spec["kind"] != 4 or spec["rounds"] != rounds
                    or spec["field_a"] != [[1, 2]]*expected_field_count
                    or spec["field_b"] != [[-1, -2]]*expected_field_count):
                raise AssertionError("CPU recurrence or paired case differs")
            params = dict(case["writer"])
            params["reference_slots"] = {int(k): v for k, v in params["reference_slots"].items()}
            if params["merge_zero_guards"] != (variant == "merged"):
                raise AssertionError("Writer option differs from variant")
            w["encoder"] = NativePCMWriter(Writer(**params), helper=self.helpers["pcm"])
            w["cpu"] = native.CPU(self.helpers["cpu"], case["size"], spec)
            w["selection"] = tuple(sorted(set(case["output_frames"].values())))
            for job, record in zip(w["jobs"], case["jobs"]):
                job["compact"] = common.checked(self.root, record["inputs"]["image"])
                job["output"] = common.checked(self.root, record["outputs"]["output"])
                if common.checked(self.root, record["ordinary"]) != b"".join(job["units"]):
                    raise AssertionError("Ordinary stream differs from framed units")
                if w["cpu"].run(job["compact"]) != job["output"]:
                    raise AssertionError("Native CPU output differs")
                if w["encoder"].input_luma(job["compact"], 0) != job["prefix"][0]:
                    raise AssertionError("Native upload differs from frozen input")
            self.groups.setdefault("/".join((size, name)), {})[variant] = key
        for group, variants in self.groups.items():
            if set(variants) != {"baseline", "merged"}:
                raise AssertionError("Unpaired guard variant")
            a, b = (self.workloads[variants[v]] for v in ("baseline", "merged"))
            if (a["case"]["costs"]["total_pictures"] != b["case"]["costs"]["total_pictures"]
                    or a["case"]["cpu"] != b["case"]["cpu"] or a["fence"] != b["fence"]):
                raise AssertionError("Guard optimization changed program structure")
            for x, y in zip(a["jobs"], b["jobs"]):
                if any(x[k] != y[k] for k in ("compact", "output", "prefix", "frames")):
                    raise AssertionError("Paired inputs or complete I420 references differ")

    def cpu(self, key, index):
        w = self.workloads[key]
        job = w["jobs"][index % len(w["jobs"])]
        if w["cpu"].run(job["compact"]) != job["output"]:
            raise AssertionError("Timed CPU output differs")

    def session(self, key):
        case = self.workloads[key]["case"]
        return NVDECSession(case["coded_width"], case["coded_height"], helper=self.helpers["nvdec"], timeout=60)

    def submit(self, active, key, index, full=False):
        w = self.workloads[key]
        job = w["jobs"][index % len(w["jobs"])]
        prefix = w["encoder"].input_luma(job["compact"], 0)
        if prefix != job["prefix"][0]:
            raise AssertionError("Timed native upload differs")
        selected = None if full else w["selection"]
        actual = active.submit((prefix,)+w["tail"], selected, fence=w["fence"])
        expected = job["frames"] if full else tuple(job["frames"][i] for i in selected)
        if actual != expected:
            raise AssertionError("NVDEC returned I420 differs")

    def close(self):
        for w in self.workloads.values():
            if "cpu" in w:
                w["cpu"].close()

    def verify_stable(self):
        if common.digest((self.root/"bundle.json").read_bytes()) != self.identity["bundle_sha256"]:
            raise AssertionError("Bundle changed during execution")
        for name in self.checked_sources:
            if common.digest((REPOSITORY/name).read_bytes()) != self.bundle["source_sha256"][name]:
                raise AssertionError("Runtime source changed during execution: "+name)
        for name, path in self.helpers.items():
            if common.digest(path.read_bytes()) != self.identity["helper_sha256"][name]:
                raise AssertionError("Native helper changed during execution")
