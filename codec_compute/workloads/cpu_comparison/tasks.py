"""Fixed workload definitions; timing runners never import the compiler."""

import codec_compute as media
from ..resampling.kernels import Task, tasks as resampling_tasks


def chain_kernel(rounds, first, second):
    def kernel(image):
        for _ in range(rounds):
            image = media.average(media.gather_h264(image, first),
                                  media.gather_h264(image, second))
        return image
    return kernel


def tasks(size):
    existing = {task.name: task for task in resampling_tasks(size)}
    result = []
    for name, kind in (("control", 0), ("average-control", 1),
                       ("patch-atlas", 2), ("motion-interpolation", 3)):
        task = existing[name]
        fields = task.fields
        result.append((task, {"kind": kind, "rounds": 1,
                             "field_a": fields.get("integer", fields.get("a_quarter", ())),
                             "field_b": fields.get("b_quarter", ())}))
    count = (size//16)**2
    first, second = ((1, 2),)*count, ((-1, -2),)*count
    for rounds in (16, 64):
        task = Task("resident-chain-"+str(rounds), "Resident warp/blend recurrence",
                    "Repeated rounded fractional resampling; only the final image is returned.",
                    ("image",), chain_kernel(rounds, first, second), {"a_quarter": first, "b_quarter": second})
        result.append((task, {"kind": 4, "rounds": rounds,
                             "field_a": first, "field_b": second}))
    return result
