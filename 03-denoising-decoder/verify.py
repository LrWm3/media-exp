"""Extract decoder inference and compare every feature bit with integer inference."""


def extract(raw, images, program):
    height, width = len(images[0]), len(images[0][0])
    coded_w, coded_h = width + 32, height + 32
    luma_size = coded_w * coded_h
    frame_size = luma_size * 3 // 2
    if len(raw) != len(images) * program.pictures * frame_size:
        raise AssertionError("Decoded picture count differs from compiled program")
    neutral = bytes([128]) * (luma_size // 2)
    for frame in range(len(raw) // frame_size):
        picture = raw[frame * frame_size:(frame + 1) * frame_size]
        if picture[luma_size:] != neutral:
            raise AssertionError(f"Chroma changed in picture {frame}")
        if any(picture[:16 * coded_w]) or any(picture[luma_size - 16 * coded_w:luma_size]):
            raise AssertionError(f"Guard changed in picture {frame}")
        for y in range(16, coded_h - 16):
            if any(picture[y * coded_w:y * coded_w + 16]) or any(picture[(y + 1) * coded_w - 16:(y + 1) * coded_w]):
                raise AssertionError(f"Guard changed in picture {frame}")

    def plane(sample, frame):
        offset = (sample * program.pictures + frame) * frame_size
        return [list(raw[offset + (y + 16) * coded_w + 16:offset + (y + 16) * coded_w + 16 + width])
                for y in range(height)]

    outputs, features = [], []
    for sample, image in enumerate(images):
        if plane(sample, 0) != [[value * 255 for value in row] for row in image]:
            raise AssertionError("Decoded seed differs from noisy input")
        feature_maps = [plane(sample, frame) for frame in program.features]
        if any(value not in (0, 1) for feature in feature_maps for row in feature for value in row):
            raise AssertionError("Nonbinary feature activation")
        output = plane(sample, program.output)
        if any(value not in (0, 255) for row in output for value in row):
            raise AssertionError("Nonbinary denoised output")
        outputs.append([[value // 255 for value in row] for row in output])
        features.append(feature_maps)
    return outputs, features


def verify(outputs, features, images, model):
    if len(outputs) != len(images) or len(features) != len(images):
        raise AssertionError("Inference batch length differs from input batch")
    for sample, (actual, actual_features, image) in enumerate(zip(outputs, features, images)):
        expected, expected_features = model.predict(image)
        if actual_features != expected_features:
            raise AssertionError(f"Decoded feature maps differ from model for image {sample}")
        if actual != expected:
            raise AssertionError(f"Decoded output differs from model for image {sample}")
