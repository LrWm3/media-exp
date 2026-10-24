"""Independent graph semantics and actual decoded-picture extraction."""


def evaluate(graph, outputs, image):
    height, width = len(image), len(image[0])
    planes = {}
    for item in graph:
        if item.op == "input":
            result = bytearray(value for row in image for value in row)
        elif item.op == "shift":
            dx, dy = item.params
            source = planes[item.args[0]]
            result = bytearray(width*height)
            for y in range(height):
                if 0 <= y+dy < height:
                    for x in range(width):
                        if 0 <= x+dx < width:
                            result[y*width+x] = source[(y+dy)*width+x+dx]
        elif item.op == "weighted":
            weight, bias = item.params
            result = bytearray(weight*value+bias for value in planes[item.args[0]])
        elif item.op == "combine":
            wa, wb, bias = item.params
            result = bytearray(wa*a+wb*b+bias for a, b in zip(planes[item.args[0]], planes[item.args[1]]))
        elif item.op == "threshold":
            result = bytearray(int(value >= item.params[0]) for value in planes[item.args[0]])
        else:
            raise AssertionError(f"Unknown graph operation {item.op}")
        planes[item] = result
    return {name: [list(planes[graph[index]][y*width:(y+1)*width]) for y in range(height)]
            for name, index in outputs.items()}


def read(raw, program):
    width, height = program.writer.width, program.writer.height
    luma_size = width*height
    frame_size = luma_size*3//2
    if len(raw) != program.pictures*frame_size:
        raise AssertionError("Decoded picture count does not match manifest")
    view = memoryview(raw)
    neutral, edge, side = bytes([128])*(luma_size//2), bytes(width*16), bytes(16)
    for frame in range(program.pictures):
        start = frame*frame_size
        if view[start+luma_size:start+frame_size] != neutral:
            raise AssertionError(f"Chroma changed at frame {frame}")
        if view[start:start+len(edge)] != edge or view[start+luma_size-len(edge):start+luma_size] != edge:
            raise AssertionError(f"Guard changed at frame {frame}")
        for y in range(16, height-16):
            if view[start+y*width:start+y*width+16] != side or view[start+(y+1)*width-16:start+(y+1)*width] != side:
                raise AssertionError(f"Guard changed at frame {frame}")
    outputs = {}
    for name, frame in program.output_frames.items():
        offset = frame*frame_size
        plane = [list(raw[offset+(y+16)*width+16:offset+(y+16)*width+16+program.width]) for y in range(program.height)]
        node = program.graph[program.graph_outputs[name]]
        if any(not node.lo <= value <= node.hi for row in plane for value in row):
            raise AssertionError(f"Decoded output {name} violates its proven range")
        outputs[name] = plane
    return outputs
