"""Portable ctypes adapter for the independently implemented CPU kernels."""

import ctypes as C
from pathlib import Path


class CPU:
    def __init__(self, helper, size, specification):
        if type(size) is not int or not 16 <= size <= 4096 or size % 16:
            raise ValueError("CPU size must be a multiple of sixteen in 16..4096")
        kind, rounds = specification["kind"], specification["rounds"]
        if type(kind) is not int or kind not in range(5) or type(rounds) is not int or not 1 <= rounds <= 4096:
            raise ValueError("Invalid CPU kind or round count")
        for name, needed in (("field_a", kind in (2, 3, 4)), ("field_b", kind in (3, 4))):
            field = specification[name]
            if len(field) != ((size//16)**2 if needed else 0):
                raise ValueError("CPU field must contain exactly one pair per block when required")
            if any(len(pair) != 2 or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in pair)
                   for pair in field):
                raise ValueError("CPU fields require int32 displacement pairs")
        self.library = C.CDLL(str(Path(helper).resolve()))
        self.library.cpu_create.argtypes = [C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int32), C.POINTER(C.c_int32)]
        self.library.cpu_create.restype = C.c_void_p
        self.library.cpu_execute.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        self.library.cpu_execute.restype = C.c_int
        self.library.cpu_destroy.argtypes = [C.c_void_p]
        self.library.cpu_destroy.restype = None
        self.library.cpu_error.restype = C.c_char_p
        fields = [(C.c_int32*(2*len(specification[name])))(*(v for pair in specification[name] for v in pair))
                  if specification[name] else None for name in ("field_a", "field_b")]
        self.context = self.library.cpu_create(size, specification["kind"], specification["rounds"], *fields)
        if not self.context:
            raise RuntimeError(self.library.cpu_error().decode())
        self.size = size
        self.output = C.create_string_buffer(size*size)

    def run(self, a, b=None):
        if (self.context is None or type(a) is not bytes or len(a) != self.size**2
                or (b is not None and (type(b) is not bytes or len(b) != self.size**2))):
            raise ValueError("CPU context or compact image shape is invalid")
        if self.library.cpu_execute(self.context, a, b, self.output):
            raise RuntimeError(self.library.cpu_error().decode())
        return self.output.raw

    def close(self):
        if self.context:
            self.library.cpu_destroy(self.context)
            self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
