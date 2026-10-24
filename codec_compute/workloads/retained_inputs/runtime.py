"""Python 3.8 runtime for exact multi-output retained-input experiments."""

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


comparison = module("_retained_comparison", HERE.parent/"cpu_comparison/benchmark.py")
common, native = comparison.common, comparison.native
Accounting, measurement, summarize, counts = (comparison.Accounting, comparison.measurement,
                                              comparison.summarize, comparison.counts)
from decoder_core.pcm import NativePCMWriter
from decoder_core.session import NVDECSession
from decoder_core.stream import NVDECStream


def semantic_audit(case, manifest):
    graph = manifest["ir"]
    names = ["image"] if case["workload"] == "patch-atlas" else ["a", "b"]
    kind = 2 if len(names) == 1 else 3
    if (case["input_names"] != names or graph["dtype"] != "u8"
            or [x["name"] for x in graph["inputs"]] != names
            or any(x["range"] != [0, 255] for x in graph["inputs"])):
        raise AssertionError("Semantic input contract differs")
    nodes = {x["id"]: x for x in graph["nodes"]}
    visited = set()

    def check(identifier, operation, params, arity):
        node = nodes[identifier]
        if (node["op"], node["params"], len(node["args"])) != (operation, params, arity):
            raise AssertionError("CPU fields differ from semantic IR")
        if operation == "input" and node["range"] != [0, 255]:
            raise AssertionError("Semantic input must cover full u8 range")
        visited.add(identifier)
        return node["args"]

    expected_names = ["output_{:03}".format(i) for i in range(case["output_count"])]
    if (list(graph["outputs"]) != expected_names
            or [v["name"] for v in case["variants"]] != expected_names):
        raise AssertionError("Named variant order differs")
    for variant in case["variants"]:
        cpu = variant["cpu"]
        if (cpu["kind"], cpu["rounds"]) != (kind, 1):
            raise AssertionError("Wrong native CPU operation")
        output = graph["outputs"][variant["name"]]
        reads = [output] if kind == 2 else check(output, "average", ["half_up"], 2)
        for index, read in enumerate(reads):
            field = cpu["field_a" if index == 0 else "field_b"]
            if (len(field) != (case["size"]//16)**2
                    or any(len(pair) != 2 or any(type(v) is not int for v in pair) for pair in field)):
                raise AssertionError("Invalid motion field")
            parent = check(read, "gather" if kind == 2 else "gather_h264",
                           [field, 16]+(["zero"] if kind == 2 else []), 1)[0]
            check(parent, "input", [names[index]], 0)
        if kind == 2 and cpu["field_b"]:
            raise AssertionError("Atlas has no second source")
    if visited != set(nodes):
        raise AssertionError("Unaccounted graph nodes")


def load(root, *, sources=True):
    root = Path(root).resolve()
    data = (root/"bundle.json").read_bytes()
    bundle = json.loads(data)
    if (bundle.get("format") != "codec-compute-retained-inputs-1" or bundle.get("status") != "pass"
            or bundle["fixtures"] != ["scene", "noise", "zero", "full"]
            or bundle["dpb_policies"] != ["dpb16", "tight"]):
        raise AssertionError("Expected passing paired four-fixture corpus")
    if sources:
        for name, checksum in bundle["source_sha256"].items():
            if common.digest((REPOSITORY/name).read_bytes()) != checksum:
                raise AssertionError("Source differs: "+name)
    expected_keys = {str(s)+"/"+w+"/"+str(n)+"/"+p for s in bundle["sizes"]
                     for w in ("patch-atlas", "motion-interpolation")
                     for n in bundle["output_counts"] for p in bundle["dpb_policies"]}
    if set(bundle["cases"]) != expected_keys:
        raise AssertionError("Corpus case matrix differs")
    workloads, groups = {}, {}
    for key, case in bundle["cases"].items():
        size, workload, number, policy = key.split("/")
        if ((int(size), workload, int(number), policy) !=
                (case["size"], case["workload"], case["output_count"], case["dpb_policy"])):
            raise AssertionError("Case key differs")
        manifest = json.loads(common.checked(root, case["manifest"]))
        shape = common.geometry(manifest)
        semantic_audit(case, manifest)
        inputs, outputs = len(case["input_names"]), case["output_count"]
        expected_frames = dict((v["name"], inputs+i) for i, v in enumerate(case["variants"]))
        slots = case["writer"]["reference_slots"]
        capacity = 16 if policy == "dpb16" else max(slots.values())+1
        if (case["reference_capacity"] != capacity or case["writer"]["reference_capacity"] != capacity
                or manifest["reference_capacity"] != capacity or manifest["prediction_slice_rows"] != 0
                or slots != manifest["reference_slots"]
                or manifest["optimizations"]["tight_dpb"] != {
                    "enabled": policy == "tight", "before_capacity": 16, "after_capacity": capacity}
                or manifest["merge_zero_guards"] is not True
                or case["writer"]["prediction_slice_rows"] != 0
                or case["writer"]["merge_zero_guards"] is not True
                or manifest["output_frames"] != expected_frames or case["output_frames"] != expected_frames
                or case["costs"] != manifest["costs"]
                or manifest["costs"]["total_pictures"] != inputs+outputs
                or manifest["costs"]["computation_pictures"] != outputs
                or manifest["costs"]["input_pictures"] != inputs
                or manifest["costs"]["prediction_slices"] != outputs):
            raise AssertionError("Picture, capacity or slice contract differs")
        if (case["frame_bytes"] != shape["frame_bytes"]
                or (case["coded_width"], case["coded_height"]) !=
                (shape["coded_width"], shape["coded_height"])):
            raise AssertionError("Coded shape differs")
        tail = tuple(common.checked(root, record) for record in case["prediction_units"])
        if len(tail) != outputs or sum(map(len, tail)) != case["costs"]["prediction_bytes"]:
            raise AssertionError("Prediction cost differs")
        for unit in tail:
            nals = unit.split(b"\0\0\0\1")[1:]
            if len(nals) != 1 or not nals[0] or nals[0][0] & 31 != 1:
                raise AssertionError("Prediction must contain one non-IDR VCL slice")
        jobs = []
        if len(case["jobs"]) != 4:
            raise AssertionError("Four fixtures required")
        for index, record in enumerate(case["jobs"]):
            if record["fixture"] != bundle["fixtures"][index]:
                raise AssertionError("Fixture order differs")
            compact = {name: common.checked(root, record["inputs"][name]) for name in case["input_names"]}
            prefix = tuple(common.checked(root, item) for item in record["units"])
            if len(prefix) != inputs:
                raise AssertionError("Input picture count differs")
            units = prefix+tail
            if common.checked(root, record["stream"]) != b"".join(units):
                raise AssertionError("Stream differs from framed program")
            raw = common.checked(root, record["expected"])
            if len(raw) != shape["frame_bytes"]*(inputs+outputs):
                raise AssertionError("Reference frame count differs")
            frames = tuple(raw[i:i+shape["frame_bytes"]] for i in range(0, len(raw), shape["frame_bytes"]))
            common.validate_guards(frames, shape)
            expected = {name: common.checked(root, record["outputs"][name]) for name in expected_frames}
            for i, name in enumerate(case["input_names"]):
                if common.extract(frames[i], shape) != compact[name]:
                    raise AssertionError("Input frame differs")
                if index >= 2 and compact[name] != bytes([255 if index == 3 else 0])*case["size"]**2:
                    raise AssertionError("Extreme fixture differs")
            for name, frame in expected_frames.items():
                if common.extract(frames[frame], shape) != expected[name]:
                    raise AssertionError("Output frame differs")
            jobs.append(dict(compact=compact, prefix=prefix, units=units, frames=frames, outputs=expected))
        workloads[key] = dict(case=case, shape=shape, tail=tail, jobs=jobs,
                              selection=tuple(expected_frames.values()), fence=common.checked(root, case["fence"]))
        groups.setdefault("/".join(key.split("/")[:3]), {})[policy] = key
    for variants in groups.values():
        a, b = (workloads[variants[p]] for p in ("dpb16", "tight"))
        if (a["case"]["variants"] != b["case"]["variants"]
                or a["case"]["output_frames"] != b["case"]["output_frames"]):
            raise AssertionError("Policy changed logical program")
        for x, y in zip(a["jobs"], b["jobs"]):
            if any(x[k] != y[k] for k in ("compact", "frames", "outputs")):
                raise AssertionError("DPB policy changed complete I420")
    return data, bundle, workloads, groups


class Corpus:
    def __init__(self, root, *, cpu_helper, pcm_helper, nvdec_helper, build_info):
        self.root = Path(root).resolve()
        data, self.bundle, self.workloads, self.groups = load(self.root)
        self.helpers = {k: Path(p).resolve() for k, p in
                        (("cpu", cpu_helper), ("pcm", pcm_helper), ("nvdec", nvdec_helper))}
        self.build_path = Path(build_info).resolve()
        build_bytes = self.build_path.read_bytes()
        build = json.loads(build_bytes)
        for key in ("cpu", "pcm"):
            if build["helper_sha256"][key] != common.digest(self.helpers[key].read_bytes()):
                raise AssertionError("Native helper differs")
        flags = build["flags"]["cpu"]
        if not all(flag in flags for flag in ("-O3", "-DNDEBUG", "-funroll-loops")) or not any(
                flag in flags for flag in ("-march=native", "-mcpu=native")):
            raise AssertionError("Tuned CPU build required")
        for name, checksum in build["source_sha256"].items():
            if self.bundle["source_sha256"][name] != checksum:
                raise AssertionError("Build source differs")
        self.identity = dict(source_commit=self.bundle["source_commit"],
                             source_sha256=self.bundle["source_sha256"],
                             bundle_sha256=common.digest(data), native_build=build,
                             build_info_sha256=common.digest(build_bytes),
                             helper_sha256={k: common.digest(p.read_bytes()) for k,p in self.helpers.items()})
        Writer = common.writer_class(REPOSITORY/"decoder_core/bitstream.py")
        try:
            for w in self.workloads.values():
                case = w["case"]
                params = dict(case["writer"])
                params["reference_slots"] = {int(k):v for k,v in params["reference_slots"].items()}
                w["encoder"] = NativePCMWriter(Writer(**params), helper=self.helpers["pcm"])
                w["cpus"] = [native.CPU(self.helpers["cpu"], case["size"], v["cpu"]) for v in case["variants"]]
                for index in range(4):
                    self.cpu(w, index)
                    self.units(w, index, "native-host")
        except BaseException:
            self.close()
            raise

    def cpu(self, w, index):
        job = w["jobs"][index]
        for cpu, variant in zip(w["cpus"], w["case"]["variants"]):
            if cpu.run(*(job["compact"][name] for name in w["case"]["input_names"])) != job["outputs"][variant["name"]]:
                raise AssertionError("Native CPU differs")

    def units(self, w, index, mode):
        job = w["jobs"][index]
        if mode != "native-host":
            return job["units"]
        prefix = tuple(w["encoder"].input_luma(job["compact"][name], i)
                       for i, name in enumerate(w["case"]["input_names"]))
        if prefix != job["prefix"]:
            raise AssertionError("Native input serialization differs")
        return prefix+w["tail"]

    def session(self, w):
        return NVDECSession(w["case"]["coded_width"], w["case"]["coded_height"],
                            helper=self.helpers["nvdec"], timeout=60)

    def submit(self, active, w, index, mode="native-host", full=False):
        units = self.units(w, index, mode)
        if mode == "prepared-complete":
            if active.complete(units, fence=w["fence"]) != ():
                raise AssertionError("Unexpected completion-only result")
        else:
            selected = tuple(range(len(units))) if full else w["selection"]
            result = active.submit(units, selected, fence=w["fence"])
            if result != tuple(w["jobs"][index]["frames"][i] for i in selected):
                raise AssertionError("NVDEC differs from complete I420 reference")

    def verify_stable(self):
        if common.digest((self.root/"bundle.json").read_bytes()) != self.identity["bundle_sha256"]:
            raise AssertionError("Bundle changed")
        if common.digest(self.build_path.read_bytes()) != self.identity["build_info_sha256"]:
            raise AssertionError("Build changed")
        for name, checksum in self.bundle["source_sha256"].items():
            if common.digest((REPOSITORY/name).read_bytes()) != checksum:
                raise AssertionError("Source changed: "+name)
        for key, path in self.helpers.items():
            if common.digest(path.read_bytes()) != self.identity["helper_sha256"][key]:
                raise AssertionError("Helper changed")

    def close(self):
        for w in self.workloads.values():
            for cpu in w.get("cpus", []):
                cpu.close()
