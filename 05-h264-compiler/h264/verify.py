"""Independent graph semantics and actual decoded-picture extraction."""


def clip(value):
    return min(255, max(0, value))


def interpolate(plane, width, height, x_q, y_q):
    """H.264 8.4.2.2.1 luma interpolation at an absolute quarter-pel position.

    The flat plane contains only the active image. The coded picture surrounds
    it with a zero guard, whose outer edges also extend as zero. Consequently
    every integer sample outside the active image is zero, at any distance.
    Python divmod supplies the required floor for negative motion coordinates.
    """
    x, fx = divmod(x_q, 4)
    y, fy = divmod(y_q, 4)
    taps = (1, -5, 20, 20, -5, 1)

    def pixel(px, py):
        return plane[py*width+px] if 0 <= px < width and 0 <= py < height else 0

    def horizontal_raw(px, py):
        return sum(weight*pixel(px+offset-2, py) for offset, weight in enumerate(taps))

    def half(hx, hy):
        """Sample the 3x3 half-pel grid around (x, y)."""
        px, py = x+hx//2, y+hy//2
        if hx != 1 and hy != 1:
            return pixel(px, py)
        if hy != 1:
            return clip((horizontal_raw(px, py)+16) >> 5)
        if hx != 1:
            return clip((sum(weight*pixel(px, py+offset-2)
                             for offset, weight in enumerate(taps))+16) >> 5)
        # Keep the complete first six-tap sum: clipping or rounding between
        # horizontal and vertical passes changes diagonal half-pel samples.
        total = sum(weight*horizontal_raw(px, py+offset-2)
                    for offset, weight in enumerate(taps))
        return clip((total+512) >> 10)

    if fx % 2 == 0 and fy % 2 == 0:
        return half(fx//2, fy//2)
    if fx % 2 and fy % 2:
        # The four diagonal quarter-pels average perpendicular half-pels,
        # not an integer sample and the diagonal half-pel.
        first, second = half(1, 2 if fy == 3 else 0), half(2 if fx == 3 else 0, 1)
    elif fx % 2:
        first, second = half(fx//2, fy//2), half(fx//2+1, fy//2)
    else:
        first, second = half(fx//2, fy//2), half(fx//2, fy//2+1)
    return (first+second+1) >> 1


def evaluate(graph, outputs, image):
    boards = image if isinstance(image, dict) else None
    first = next(iter(boards.values())) if boards is not None else image
    height, width = len(first), len(first[0])
    planes = {}
    for item in graph:
        if item.op == "input":
            board = boards[item.params[0]] if boards is not None else image
            result = bytearray(value for row in board for value in row)
        elif item.op == "shift":
            dx, dy = item.params
            source = planes[item.args[0]]
            result = bytearray(width*height)
            for y in range(height):
                if 0 <= y+dy < height:
                    for x in range(width):
                        if 0 <= x+dx < width:
                            result[y*width+x] = source[(y+dy)*width+x+dx]
        elif item.op in ("sample", "gather"):
            source = planes[item.args[0]]
            result = bytearray(width*height)
            for y in range(height):
                for x in range(width):
                    qx, qy = (item.params if item.op == "sample"
                              else item.params[0][(y//16)*(width//16)+x//16])
                    result[y*width+x] = interpolate(source, width, height, 4*x+qx, 4*y+qy)
        elif item.op == "weighted":
            weight, bias = item.params
            result = bytearray(weight*value+bias for value in planes[item.args[0]])
        elif item.op == "combine":
            wa, wb, bias = item.params
            result = bytearray(wa*a+wb*b+bias for a, b in zip(planes[item.args[0]], planes[item.args[1]]))
        elif item.op == "saturating_weighted":
            weight, bias = item.params
            result = bytearray(clip(weight*value+bias) for value in planes[item.args[0]])
        elif item.op == "saturating_combine":
            wa, wb, bias = item.params
            result = bytearray(clip(wa*a+wb*b+bias) for a, b in zip(planes[item.args[0]], planes[item.args[1]]))
        elif item.op == "threshold":
            result = bytearray(int(value >= item.params[0]) for value in planes[item.args[0]])
        else:
            raise AssertionError(f"Unknown graph operation {item.op}")
        planes[item] = result
    return {name: [list(planes[graph[index]][y*width:(y+1)*width]) for y in range(height)]
            for name, index in outputs.items()}


def read(raw, program, *, selected_frames=None):
    """Validate packed I420 for all pictures or an explicit physical selection.

    Each supplied picture is stored once, in increasing physical frame order.
    Every named output must be covered; absent intermediates are never fabricated.
    """
    selection = tuple(range(program.pictures)) if selected_frames is None else tuple(selected_frames)
    if not selection:
        raise ValueError("selected_frames must be nonempty")
    if any(type(frame) is not int for frame in selection):
        raise ValueError("selected_frames must contain integer picture indices")
    if any(frame < 0 or frame >= program.pictures for frame in selection):
        raise ValueError("selected_frames picture index out of range")
    if any(left >= right for left, right in zip(selection, selection[1:])):
        raise ValueError("selected_frames must be strictly increasing without duplicates")
    positions = {frame: index for index, frame in enumerate(selection)}
    missing = [name for name, frame in program.output_frames.items() if frame not in positions]
    if missing:
        raise ValueError("selected_frames must cover every named output; missing: " + ", ".join(missing))
    width, height = program.writer.width, program.writer.height
    luma_size = width*height
    frame_size = luma_size*3//2
    if len(raw) != len(selection)*frame_size:
        raise AssertionError("Decoded picture count does not match manifest")
    view = memoryview(raw)
    neutral, edge, side = bytes([128])*(luma_size//2), bytes(width*16), bytes(16)
    for index, frame in enumerate(selection):
        start = index*frame_size
        if view[start+luma_size:start+frame_size] != neutral:
            raise AssertionError(f"Chroma changed at frame {frame}")
        if view[start:start+len(edge)] != edge or view[start+luma_size-len(edge):start+luma_size] != edge:
            raise AssertionError(f"Guard changed at frame {frame}")
        for y in range(16, height-16):
            if view[start+y*width:start+y*width+16] != side or view[start+(y+1)*width-16:start+(y+1)*width] != side:
                raise AssertionError(f"Guard changed at frame {frame}")
    outputs = {}
    for name, frame in program.output_frames.items():
        offset = positions[frame]*frame_size
        plane = [list(raw[offset+(y+16)*width+16:offset+(y+16)*width+16+program.width]) for y in range(program.height)]
        node = program.graph[program.graph_outputs[name]]
        if any(not node.lo <= value <= node.hi for row in plane for value in row):
            raise AssertionError(f"Decoded output {name} violates its proven range")
        outputs[name] = plane
    return outputs
