"""Render paired saved decoded outputs at 1:1 scale, without hardware claims."""

import argparse
import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_interface_capture_runtime", HERE/"runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def capture(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    data, bundle, workloads = runtime.resident.load(root)
    runtime.audit_pairs(root, bundle, workloads)
    if (not bundle.get("decoders") or bundle["validation"]["local_decoder_runs"] !=
            len(bundle["cases"])*4*len(bundle["decoders"])):
        raise AssertionError("Missing complete passing local decoder evidence")
    output.mkdir(parents=True, exist_ok=True)
    report = {"format": "codec-compute-interface-capture-1", "source_commit": bundle["source_commit"],
              "bundle_sha256": runtime.common.digest(data),
              "script_sha256": runtime.common.digest(Path(__file__).read_bytes()),
              "source": "Fixture 0 saved decoded I420; exact output crops and complete paired I420 audited against frozen bundle",
              "hardware_scope": "No fresh decode or NVDEC claim; separate benchmark report supplies hardware evidence",
              "display": {"scale": 1, "contrast_stretch": False, "interpolation": "none", "zero_luma": "black",
                          "guards_chroma": "Audited in complete I420; only active luma displayed"},
              "panels": [], "sheets": []}
    title = ImageFont.truetype("DejaVuSans.ttf", 22)
    label = ImageFont.truetype("DejaVuSans.ttf", 14)
    for size in bundle["sizes"]:
        width = max(size, 300)+24
        sheet = Image.new("RGB", (2*width+24, 2*(size+80)+105), "#101826")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), "Saved decoder references: "+str(size)+" x "+str(size), font=title, fill="white")
        draw.text((20, 48), "Fixture 0 | native 1:1 pixels | zero remains black", font=label, fill="#c5cede")
        for row, name in enumerate(("motion-interpolation", "resident-chain-"+str(max(bundle["depths"])))):
            a = workloads[str(size)+"/"+name+"/rows-1"]["jobs"][0]
            b = workloads[str(size)+"/"+name+"/full"]["jobs"][0]
            if a["frames"] != b["frames"] or a["output"] != b["output"]:
                raise AssertionError("Displayed paired decoder references differ")
            for column, variant in enumerate(("rows-1", "full")):
                key = str(size)+"/"+name+"/"+variant
                w = workloads[key]
                frame_index = w["case"]["output_frames"]["output"]
                frame = w["jobs"][0]["frames"][frame_index]
                compact = runtime.common.extract(frame, w["shape"])
                if compact != w["jobs"][0]["output"]:
                    raise AssertionError("Panel differs from saved decoded output")
                panel = Image.frombytes("L", (size, size), compact)
                panel_name = str(size)+"-"+name+"-"+variant+".png"
                panel.save(output/panel_name)
                x, y = 20+column*width, 86+row*(size+80)
                draw.text((x, y), name+" / "+variant, font=label, fill="white")
                draw.text((x, y+22), "Exact complete I420 pair", font=label, fill="#9fc1cc")
                sheet.paste(panel.convert("RGB"), (x, y+45))
                report["panels"].append({"case": key, "fixture": 0, "frame": frame_index,
                    "reference_i420": w["case"]["jobs"][0]["expected"],
                    "frame_sha256": runtime.common.digest(frame), "active_luma_sha256": runtime.common.digest(compact),
                    "path": panel_name, "sha256": runtime.common.digest((output/panel_name).read_bytes())})
        filename = "paired-"+str(size)+".png"
        sheet.save(output/filename)
        report["sheets"].append({"path": filename, "sha256": runtime.common.digest((output/filename).read_bytes())})
    (output/"capture.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    capture(args.bundle, args.output)
