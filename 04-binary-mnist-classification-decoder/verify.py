"""Read classification bits and hidden activations directly from decoded pictures."""


def extract(raw, images, compiled):
    program = compiled.program
    width, height = compiled.writer.width, compiled.writer.height
    luma_size = width * height
    frame_size = luma_size * 3 // 2
    if len(raw) != program.pictures * frame_size:
        raise AssertionError("Decoded picture count differs from compiled program")
    view = memoryview(raw)
    neutral = bytes([128]) * (luma_size // 2)
    zero_edge = bytes(width * 16)
    for frame in range(program.pictures):
        start = frame * frame_size
        if view[start+luma_size:start+frame_size] != neutral:
            raise AssertionError(f"Chroma changed in picture {frame}")
        if view[start:start+width*16] != zero_edge or view[start+luma_size-width*16:start+luma_size] != zero_edge:
            raise AssertionError(f"Guard changed in picture {frame}")
        for y in range(16, height-16):
            if view[start+y*width:start+y*width+16] != bytes(16) or view[start+(y+1)*width-16:start+(y+1)*width] != bytes(16):
                raise AssertionError(f"Guard changed in picture {frame}")

    def pixel(frame, x, y):
        return raw[frame*frame_size + (y+16)*width + x+16]

    board = compiled.board(images)
    for y, row in enumerate(board):
        if any(pixel(0, x, y) != value*255 for x, value in enumerate(row)):
            raise AssertionError("Decoded seed differs from input")
    records = []
    for index in range(len(images)):
        x, y = compiled.anchor(index)
        pooled = [[pixel(program.pooled, x+dx*2, y+dy*2) for dx in range(14)] for dy in range(14)]
        hidden = [pixel(frame, x, y) for frame in program.hidden]
        output = pixel(program.output, x, y)
        if output not in (0, 255) or any(v not in (0, 1) for v in hidden+[v for row in pooled for v in row]):
            raise AssertionError("Nonbinary decoded activation")
        records.append({"bit": output//255, "hidden": hidden, "pooled": pooled})
    return records


def verify(records, images, model):
    if len(records) != len(images):
        raise AssertionError("Inference batch length differs from input batch")
    for index, (record, image) in enumerate(zip(records, images)):
        bit, hidden, pooled = model.predict(image)
        if record != {"bit": bit, "hidden": hidden, "pooled": pooled}:
            raise AssertionError(f"Decoded inference differs from integer model for digit {index}")
