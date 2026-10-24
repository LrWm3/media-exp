"""Python 3.8 runtime for frozen paired layouts; no compiler/evaluator imports."""

import importlib.util
import json
from pathlib import Path
import platform
import re
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


comparison = module("_interface_comparison", HERE.parent/"cpu_comparison/benchmark.py")
common, resident, native = comparison.common, comparison.resident, comparison.native
Accounting, measurement, summarize, counts = (comparison.Accounting, comparison.measurement,
                                              comparison.summarize, comparison.counts)
from decoder_core.pcm import NativePCMWriter
from decoder_core.session import NVDECSession

LAYOUTS = {"rows-1": 1, "rows-4": 4, "full": 0}
BUILD_SOURCES = {"cpu": "codec_compute/workloads/cpu_comparison/cpu.cpp",
                 "cpu_planes": "codec_compute/workloads/cpu_comparison/cpu_planes.cpp",
                 "pcm": "decoder_core/pcm_native.cpp"}


def audit_cpu(root, key, case):
    """Bind every native CPU operation and its static fields to the semantic DAG."""
    name, spec = key.split("/")[1], case["cpu"]
    kinds = {"control": 0, "patch-atlas": 2, "motion-interpolation": 3}
    kind = 4 if name.startswith("resident-chain-") else kinds.get(name)
    rounds = int(name.rsplit("-", 1)[1]) if kind == 4 else 1
    if (kind is None or type(spec["kind"]) is not int or type(spec["rounds"]) is not int
            or (spec["kind"], spec["rounds"]) != (kind, rounds)):
        raise AssertionError("CPU kind or recurrence depth differs")
    count = (case["size"]//16)**2
    for field, needed in (("field_a", kind >= 2), ("field_b", kind >= 3)):
        if len(spec[field]) != (count if needed else 0):
            raise AssertionError("CPU motion grid size differs")
        if any(len(pair) != 2 or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in pair)
               for pair in spec[field]):
            raise AssertionError("CPU motion field requires signed integer pairs")
    if kind == 4 and (spec["field_a"] != [[1, 2]]*count or spec["field_b"] != [[-1, -2]]*count):
        raise AssertionError("Shared-plane CPU requires the fixed recurrence fields")
    graph = json.loads(common.checked(root, case["manifest"]))["ir"]
    input_names = ["a", "b"] if kind == 3 else ["image"]
    if (case["input_names"] != input_names or graph["dtype"] != "u8"
            or [item["name"] for item in graph["inputs"]] != input_names
            or any(item["range"] != [0, 255] for item in graph["inputs"])
            or any(item["range"] != [0, 255] for item in graph["nodes"] if item["op"] == "input")):
        raise AssertionError("CPU inputs require ordered full-u8 semantic inputs")
    nodes, visited = {node["id"]: node for node in graph["nodes"]}, set()

    def node(identifier, operation, parameters, arity):
        value = nodes[identifier]
        if value["op"] != operation or value["params"] != parameters or len(value["args"]) != arity:
            raise AssertionError("CPU specification differs from semantic graph")
        visited.add(identifier)
        return value["args"]

    def image(identifier, name):
        node(identifier, "input", [name], 0)

    def sample(identifier, field, integer=False):
        parameters = [spec[field], 16]+(["zero"] if integer else [])
        return node(identifier, "gather" if integer else "gather_h264", parameters, 1)[0]

    output = graph["outputs"]["output"]
    if kind == 0:
        image(output, "image")
    elif kind == 2:
        image(sample(output, "field_a", integer=True), "image")
    elif kind == 3:
        a, b = node(output, "average", ["half_up"], 2)
        image(sample(a, "field_a"), "a")
        image(sample(b, "field_b"), "b")
    else:
        for _ in range(rounds):
            a, b = node(output, "average", ["half_up"], 2)
            a, b = sample(a, "field_a"), sample(b, "field_b")
            if a != b:
                raise AssertionError("Recurrence branches use different previous images")
            output = a
        image(output, "image")
    if visited != set(nodes) or set(graph["outputs"]) != {"output"}:
        raise AssertionError("Unaccounted semantic nodes or named outputs")


def audit_pairs(root, bundle, workloads):
    if (bundle.get("experiment") != "interface-limits" or bundle["jobs"] != 4
            or bundle.get("layouts") != LAYOUTS or bundle.get("fixtures") != ["scene", "noise", "zero", "255"]):
        raise ValueError("Expected four-fixture paired interface-limits export")
    expected = {str(size)+"/"+name+"/"+variant for size in bundle["sizes"]
                for name in ("control", "patch-atlas", "motion-interpolation")+tuple(
                    "resident-chain-"+str(depth) for depth in bundle["depths"])
                for variant in LAYOUTS}
    if set(workloads) != expected:
        raise AssertionError("Case set differs from declared sizes, depths and layouts")
    groups = {}
    for key, w in workloads.items():
        case = w["case"]
        size, name, variant = key.split("/")
        if (type(case["size"]) is not int or int(size) != case["size"]
                or case["prediction_slice_rows"] != LAYOUTS[variant]
                or case["writer"]["prediction_slice_rows"] != LAYOUTS[variant]
                or case["writer"]["merge_zero_guards"] is not True):
            raise AssertionError("Writer layout differs from case")
        manifest = json.loads(common.checked(root, case["manifest"]))
        if (manifest["prediction_slice_rows"] != LAYOUTS[variant]
                or manifest["merge_zero_guards"] is not True):
            raise AssertionError("Saved manifest differs from declared slice layout")
        audit_cpu(root, key, case)
        expected_compute = 0 if case["cpu"]["kind"] == 0 else case["cpu"]["rounds"]
        mb_rows = case["coded_height"]//16
        band_rows = min(LAYOUTS[variant] or mb_rows, mb_rows)
        expected_slices = (mb_rows+band_rows-1)//band_rows
        costs = case["costs"]
        if (costs["computation_pictures"] != expected_compute
                or costs["input_pictures"] != len(case["input_names"])
                or costs["total_pictures"] != expected_compute+len(case["input_names"])
                or costs["prediction_slices"] != expected_compute*expected_slices
                or len(w["tail"]) != expected_compute):
            raise AssertionError("Prediction pictures or slice-band costs differ")
        for unit in w["tail"]:
            headers = [unit[match.end()] & 31 for match in re.finditer(b"\x00\x00(?:\x00)?\x01", unit)
                       if match.end() < len(unit)]
            if headers != [1]*expected_slices:
                raise AssertionError("Actual prediction NAL slice layout differs")
        for job, record in zip(w["jobs"], case["jobs"]):
            job["compact"] = {name: common.checked(root, record["inputs"][name]) for name in case["input_names"]}
            job["output"] = common.checked(root, record["outputs"]["output"])
            if common.checked(root, record["ordinary"]) != b"".join(job["units"]):
                raise AssertionError("Ordinary stream differs from framed access units")
        for index, value in ((2, 0), (3, 255)):
            if any(plane != bytes([value])*case["size"]**2 for plane in w["jobs"][index]["compact"].values()):
                raise AssertionError("Extreme fixture differs from its declared constant")
        groups.setdefault(size+"/"+name, {})[variant] = key
    for variants in groups.values():
        a = workloads[variants["rows-1"]]
        for variant in ("rows-4", "full"):
            b = workloads[variants[variant]]
            if (a["case"]["costs"]["total_pictures"] != b["case"]["costs"]["total_pictures"]
                    or a["case"]["output_frames"] != b["case"]["output_frames"]
                    or a["case"]["cpu"] != b["case"]["cpu"] or a["fence"] != b["fence"]):
                raise AssertionError("Slice grouping changed program structure")
            for x, y in zip(a["jobs"], b["jobs"]):
                if any(x[k] != y[k] for k in ("compact", "output", "prefix", "frames")):
                    raise AssertionError("Paired inputs or complete I420 differ")
    return groups


class Corpus:
    def __init__(self, root, *, cpu_helper, planes_helper, pcm_helper, nvdec_helper, build_info):
        self.root = Path(root).resolve()
        self.helpers = {key: Path(path).resolve() for key, path in
                        (("cpu", cpu_helper), ("cpu_planes", planes_helper), ("pcm", pcm_helper), ("nvdec", nvdec_helper))}
        self.build_path = Path(build_info).resolve()
        data, self.bundle, self.workloads = resident.load(self.root)
        self.groups = audit_pairs(self.root, self.bundle, self.workloads)
        # These source files are executed/read at runtime. The full compiler and
        # exporter source set is additionally frozen in the export provenance.
        self.checked_sources = (
            "decoder_core/bitstream.py", "decoder_core/pcm.py", "decoder_core/pcm_native.cpp",
            "decoder_core/session.py", "decoder_core/decode.py", "decoder_core/nvdec_decode.cpp",
            "codec_compute/workloads/resident/benchmark.py", "codec_compute/workloads/temporal_benchmark.py",
            "codec_compute/workloads/cpu_comparison/benchmark.py", "codec_compute/workloads/cpu_comparison/native.py",
            "codec_compute/workloads/cpu_comparison/cpu.cpp", "codec_compute/workloads/cpu_comparison/cpu_planes.cpp",
            "codec_compute/workloads/interface_limits/runtime.py", "codec_compute/workloads/interface_limits/benchmark.py",
            "codec_compute/workloads/interface_limits/build.py")
        for name in self.checked_sources:
            if common.digest((REPOSITORY/name).read_bytes()) != self.bundle["source_sha256"].get(name):
                raise AssertionError("Frozen runtime source differs: "+name)
        build_bytes = self.build_path.read_bytes()
        build = json.loads(build_bytes)
        if set(build["source_sha256"]) != set(BUILD_SOURCES.values()) or set(build["helper_sha256"]) != set(BUILD_SOURCES):
            raise AssertionError("Expected generic CPU, shared-plane CPU and native PCM builds")
        architecture = platform.machine().lower()
        native_flag = "-mcpu=native" if architecture in ("aarch64", "arm64") else "-march=native"
        for key in ("cpu", "cpu_planes"):
            flags = build["flags"][key]
            if (not {"-O3", native_flag, "-funroll-loops"}.issubset(flags)
                    or any(flag.startswith("-O") and flag != "-O3" for flag in flags)):
                raise AssertionError("Both CPU baselines require native/unrolled -O3 builds")
        for key, source in BUILD_SOURCES.items():
            if common.digest(self.helpers[key].read_bytes()) != build["helper_sha256"][key]:
                raise AssertionError("Native helper differs from build record")
            if build["source_sha256"][source] != self.bundle["source_sha256"].get(source):
                raise AssertionError("Native source differs from export")
        self.identity = {"source_commit": self.bundle["source_commit"], "source_sha256": self.bundle["source_sha256"],
                         "bundle_sha256": common.digest(data), "native_build": build,
                         "build_info_sha256": common.digest(build_bytes),
                         "helper_sha256": {key: common.digest(path.read_bytes()) for key, path in self.helpers.items()}}
        Writer = common.writer_class(REPOSITORY/"decoder_core/bitstream.py")
        try:
            for w in self.workloads.values():
                case = w["case"]
                params = dict(case["writer"])
                params["reference_slots"] = {int(k): v for k, v in params["reference_slots"].items()}
                w["encoder"] = NativePCMWriter(Writer(**params), helper=self.helpers["pcm"])
                cpu_helper = self.helpers["cpu_planes" if case["cpu"]["kind"] == 4 else "cpu"]
                w["cpu"] = native.CPU(cpu_helper, case["size"], case["cpu"])
                w["selection"] = tuple(sorted(set(case["output_frames"].values())))
                for job in w["jobs"]:
                    if w["cpu"].run(*(job["compact"][name] for name in case["input_names"])) != job["output"]:
                        raise AssertionError("Native CPU output differs")
                    self.encode(w, job)
        except BaseException:
            self.close()
            raise

    def encode(self, w, job):
        prefix = tuple(w["encoder"].input_luma(job["compact"][name], i)
                       for i, name in enumerate(w["case"]["input_names"]))
        if prefix != job["prefix"]:
            raise AssertionError("Native upload differs from frozen input")
        return prefix+w["tail"]

    def cpu(self, key, index):
        w = self.workloads[key]
        job = w["jobs"][index]
        if w["cpu"].run(*(job["compact"][name] for name in w["case"]["input_names"])) != job["output"]:
            raise AssertionError("Timed CPU output differs")

    def session(self, key):
        case = self.workloads[key]["case"]
        return NVDECSession(case["coded_width"], case["coded_height"], helper=self.helpers["nvdec"], timeout=60)

    def submit(self, active, key, index, *, mode="native-host", full=False):
        if mode not in ("native-host", "prepared-host"):
            raise ValueError("Unknown host-result interface")
        w, job = self.workloads[key], self.workloads[key]["jobs"][index]
        units = self.encode(w, job) if mode == "native-host" else job["units"]
        selected = None if full else w["selection"]
        actual = active.submit(units, selected, fence=w["fence"])
        expected = job["frames"] if full else tuple(job["frames"][i] for i in selected)
        if actual != expected:
            raise AssertionError("NVDEC returned full I420 differs")

    def complete(self, active, key, index):
        w = self.workloads[key]
        if active.protocol_version < 2:
            raise RuntimeError("Completion-only requests require protocol v2")
        if active.complete(w["jobs"][index]["units"], fence=w["fence"]) != ():
            raise AssertionError("Completion-only request returned unexpected host results")

    def close(self):
        for w in self.workloads.values():
            if "cpu" in w:
                w["cpu"].close()

    def verify_stable(self):
        if common.digest((self.root/"bundle.json").read_bytes()) != self.identity["bundle_sha256"]:
            raise AssertionError("Bundle changed during execution")
        if common.digest(self.build_path.read_bytes()) != self.identity["build_info_sha256"]:
            raise AssertionError("Native build record changed during execution")
        for name in self.checked_sources:
            if common.digest((REPOSITORY/name).read_bytes()) != self.bundle["source_sha256"][name]:
                raise AssertionError("Runtime source changed during execution: "+name)
        for name, path in self.helpers.items():
            if common.digest(path.read_bytes()) != self.identity["helper_sha256"][name]:
                raise AssertionError("Native helper changed during execution")
