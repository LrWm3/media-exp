"""Application-shaped clients of explicit integer and H.264 sampling IR.

Motion fields are externally supplied compile-time constants. No optical flow,
occlusion solver, asynchronous event sensor, or dynamic field estimator appears
in these kernels.
"""

from dataclasses import dataclass
import math

import codec_compute as media


@dataclass(frozen=True)
class Task:
    name: str
    title: str
    description: str
    input_names: tuple
    kernel: object
    fields: dict

    @property
    def input_ranges(self):
        return {name: (0, 255) for name in self.input_names}


def tasks(size=128):
    if type(size) is not int or size < 32 or size % 32:
        raise ValueError("Task canvas must be a multiple of 32, at least 32")
    columns = size//16
    atlas = tuple(((2*bx+by) % columns*16-bx*16, (2*by+bx) % columns*16-by*16)
                  if bx < columns//2 and by < columns//2 else (size-1, size-1)
                  for by in range(columns) for bx in range(columns))
    rolling = tuple(((by-columns//2)*2, 0) for by in range(columns) for bx in range(columns))
    first = tuple(((-6, -2, 2, 6)[bx % 4], (-2, 2)[by % 2])
                  for by in range(columns) for bx in range(columns))
    second = tuple(((-3, 5)[by % 2] + 2*(bx % 2), (-1, 3)[bx % 2])
                   for by in range(columns) for bx in range(columns))
    def interpolate(a, b):
        return media.average(media.gather_h264(a, first), media.gather_h264(b, second))
    def residual(a, b):
        return media.absdiff(a, media.gather_h264(b, second))
    def events(a, b):
        prior = media.gather_h264(b, second)
        return {"positive": media.threshold(media.sub_sat(a, prior), 20),
                "negative": media.threshold(media.sub_sat(prior, a), 20)}
    def stereo(a, b):
        return {"cost_mhalf": media.absdiff(a, media.sample_h264(b, qx=-2)),
                "cost_phalf": media.absdiff(a, media.sample_h264(b, qx=2)),
                "cost_pthreehalf": media.absdiff(a, media.sample_h264(b, qx=6))}
    return (
        Task("patch-atlas", "Patch atlas", "Route selected 16x16 tiles into the top-left half-size ROI; zero elsewhere.",
             ("image",), lambda image: media.gather(image, atlas), {"integer": atlas}),
        Task("motion-interpolation", "Motion interpolation", "Rounded average after two independent fractional block fields.",
             ("a", "b"), interpolate, {"a_quarter": first, "b_quarter": second}),
        Task("rolling-shutter", "Bandwise correction", "One integer horizontal correction per 16-row band, supplied externally.",
             ("image",), lambda image: media.gather(image, rolling), {"integer": rolling}),
        Task("motion-residual", "Compensated residual", "Absolute byte difference after a supplied fractional reference warp.",
             ("a", "b"), residual, {"b_quarter": second}),
        Task("bipolar-events", "Dense polarity maps", "Positive and negative brightness changes >=20 after reference warping.",
             ("a", "b"), events, {"b_quarter": second}),
        Task("stereo-costs", "Stereo candidate costs", "Three absolute residual maps at reference offsets -0.5, +0.5 and +1.5 pixels.",
             ("a", "b"), stereo, {"uniform_quarter": ((-2, 0), (2, 0), (6, 0))}),
        Task("control", "Passthrough control", "Expose the uploaded image directly, with no compute picture.",
             ("image",), lambda image: image, {}),
        Task("average-control", "Unwarped average control", "Ordinary rounded average of the same two uploaded inputs.",
             ("a", "b"), lambda a, b: media.average(a, b), {}),
    )


def fixture(task, index=0, size=128):
    """Distinct fixed scenes/noise; never derive a motion field from these pixels."""
    state = (0x51A7C0DE ^ (index*0x9E3779B9)) & 0xffffffff
    images = {}
    for number, name in enumerate(task.input_names):
        board = []
        for y in range(size):
            row = []
            for x in range(size):
                state = (1664525*state+1013904223) & 0xffffffff
                if index % 2:
                    value = state >> 24
                else:
                    value = 24 + (x*2+y+index*11) % 88
                    if size//4+number*3 <= x < 3*size//5+number*3 and size//4 <= y < 3*size//4:
                        value = 208
                    if (x//8+y//8) % 7 == 0:
                        value = min(255, value+32)
                row.append(value)
            board.append(row)
        images[name] = board
    if task.name == "patch-atlas" and not index % 2:
        images["image"] = [[((x//16)*29+(y//16)*47+index*13+(x % 16)*3+(y % 16)) % 256
                            for x in range(size)] for y in range(size)]
    if task.name == "rolling-shutter" and not index % 2:
        ideal = images["image"]
        images["image"] = [[ideal[y][x-task.fields["integer"][(y//16)*(size//16)+x//16][0]]
                             if 0 <= x-task.fields["integer"][(y//16)*(size//16)+x//16][0] < size else 0
                             for x in range(size)] for y in range(size)]
    return images


def warp(board, field, *, quarter=False):
    """Independent gather loop; fractional samples use the legacy normative helper.

    The semantic IR uses codec_compute.sampling instead. This helper is the
    separately tested H.264 oracle from the original compiler.
    """
    size, columns = len(board), len(board[0])//16
    flat = bytes(v for row in board for v in row)
    if quarter:
        from codec_compute.targets.h264 import _legacy
        interpolate = _legacy().verify.interpolate
    result = []
    for y in range(size):
        row = []
        for x in range(len(board[0])):
            dx, dy = field[(y//16)*columns+x//16]
            if quarter:
                value = interpolate(flat, len(board[0]), size, 4*x+dx, 4*y+dy)
            else:
                value = board[y+dy][x+dx] if 0 <= x+dx < len(board[0]) and 0 <= y+dy < size else 0
            row.append(value)
        result.append(row)
    return result


def reference(task, images):
    def combine(a, b, function):
        return [[function(x, y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
    if task.name == "control":
        return {"output": images["image"]}
    if task.name == "average-control":
        return {"output": combine(images["a"], images["b"], lambda x, y: (x+y+1)//2)}
    if task.name in ("patch-atlas", "rolling-shutter"):
        return {"output": warp(images["image"], task.fields["integer"])}
    if task.name == "motion-interpolation":
        a = warp(images["a"], task.fields["a_quarter"], quarter=True)
        b = warp(images["b"], task.fields["b_quarter"], quarter=True)
        return {"output": combine(a, b, lambda x, y: (x+y+1)//2)}
    if task.name in ("motion-residual", "bipolar-events"):
        a, b = images["a"], warp(images["b"], task.fields["b_quarter"], quarter=True)
        if task.name == "motion-residual":
            return {"output": combine(a, b, lambda x, y: abs(x-y))}
        return {"positive": combine(a, b, lambda x, y: int(x-y >= 20)),
                "negative": combine(a, b, lambda x, y: int(y-x >= 20))}
    if task.name == "stereo-costs":
        count = (len(images["a"])*len(images["a"][0]))//256
        return {name: combine(images["a"], warp(images["b"], (offset,)*count, quarter=True), lambda x, y: abs(x-y))
                for name, offset in zip(("cost_mhalf", "cost_phalf", "cost_pthreehalf"), task.fields["uniform_quarter"])}
    raise ValueError("Unknown resampling task")


def quality_fixture(size=128):
    """Nonlinear analytic texture at two known translated times and midpoint."""
    def scene(x, y):
        return int(math.floor(128 + 52*math.cos(2*math.pi*x/16) + 35*math.sin(2*math.pi*y/23) + 0.5))
    images = {"a": [[scene(x-2.5, y) for x in range(size)] for y in range(size)],
              "b": [[scene(x+2.5, y) for x in range(size)] for y in range(size)]}
    midpoint = [[scene(x, y) for x in range(size)] for y in range(size)]
    return images, midpoint


def quality_kernel(a, b):
    return {"compensated": media.average(media.sample_h264(a, 10), media.sample_h264(b, -10)),
            "naive": media.average(a, b)}


def quality_errors(outputs, midpoint):
    """Use an interior crop excluding displacement and the six-tap footprint."""
    size, margin = len(midpoint), 8
    if size <= 2*margin:
        raise ValueError("Quality fixture requires a nonempty interior")
    count = (size-2*margin)**2
    metrics = {}
    for name, board in outputs.items():
        errors = [abs(board[y][x]-midpoint[y][x])
                  for y in range(margin, size-margin) for x in range(margin, size-margin)]
        metrics[name] = {"mae": sum(errors)/count, "max_absolute_error": max(errors),
                         "squared_error_sum": sum(error*error for error in errors)}
    return {"crop": {"x": margin, "y": margin, "width": size-2*margin, "height": size-2*margin},
            "pixels": count, "metrics": metrics}
