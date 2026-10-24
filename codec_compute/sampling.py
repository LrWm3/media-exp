"""Codec-defined sampling mathematics, independent of target lowering.

H.264 luma interpolation follows ITU-T H.264 (08/2024), 8.4.2.2.1.
https://www.itu.int/rec/T-REC-H.264-202408-S/en
The semantic image is extended with zero before any filtering; a backend
must implement this boundary rule even if its coded picture extends edges.
"""


_TAPS = (1, -5, 20, 20, -5, 1)


def interpolate_h264(plane, width, height, x_q, y_q):
    """Return one u8 luma sample at absolute quarter-pixel coordinates.

    ``plane`` contains the active image in row-major order. Negative positions
    use floor division. Half-pixel filtering keeps signed precision until the
    specified final rounding and clipping, including both diagonal passes.
    """
    x, fx = divmod(x_q, 4)
    y, fy = divmod(y_q, 4)

    def clip(value):
        return min(255, max(0, value))

    def pixel(px, py):
        return plane[py*width+px] if 0 <= px < width and 0 <= py < height else 0

    def horizontal_raw(px, py):
        return sum(weight*pixel(px+offset-2, py) for offset, weight in enumerate(_TAPS))

    def half(hx, hy):
        px, py = x+hx//2, y+hy//2
        if hx != 1 and hy != 1:
            return pixel(px, py)
        if hy != 1:
            return clip((horizontal_raw(px, py)+16) >> 5)
        if hx != 1:
            total = sum(weight*pixel(px, py+offset-2) for offset, weight in enumerate(_TAPS))
            return clip((total+16) >> 5)
        # No intermediate rounding or clipping between the six-tap passes.
        total = sum(weight*horizontal_raw(px, py+offset-2) for offset, weight in enumerate(_TAPS))
        return clip((total+512) >> 10)

    if fx % 2 == 0 and fy % 2 == 0:
        return half(fx//2, fy//2)
    if fx % 2 and fy % 2:
        # Diagonal quarter positions average perpendicular half samples.
        first, second = half(1, 2 if fy == 3 else 0), half(2 if fx == 3 else 0, 1)
    elif fx % 2:
        first, second = half(fx//2, fy//2), half(fx//2+1, fy//2)
    else:
        first, second = half(fx//2, fy//2), half(fx//2, fy//2+1)
    return (first+second+1) >> 1
