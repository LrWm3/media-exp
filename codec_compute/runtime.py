"""Resident execution adapters; no graph evaluation or instruction synthesis."""


class H264Session:
    def __init__(self, program, *, helper=None, timeout=30):
        from decoder_core.session import NVDECSession
        self.program = program
        self.native = program.native
        self.selection = tuple(sorted(set(self.native.output_frames.values())))
        # Freeze all instruction bytes before opening the runtime. Requests
        # serialize input pixels only; the same tail/fence serve every job.
        self.native.prediction_units
        self.fence = self.native.fence_unit
        self.decoder = NVDECSession(width=self.native.writer.width,
                                    height=self.native.writer.height,
                                    helper=helper, timeout=timeout)

    @property
    def stats(self):
        return dict(self.decoder.stats)

    def run(self, inputs):
        self.program.graph.validate(inputs)
        frames = self.decoder.submit(self.native.access_units(inputs),
                                     selected_frames=self.selection, fence=self.fence)
        width, height = self.native.writer.width, self.native.writer.height
        if len(frames) != len(self.selection) or any(len(frame) != width*height*3//2 for frame in frames):
            self.decoder.close()
            raise RuntimeError("Resident decoder returned an invalid picture count or size")
        from .targets.h264 import _legacy
        try:
            result = _legacy().backend.read(b"".join(frames), self.native, selected_frames=self.selection)
            return self.program._validate_outputs(result)
        except BaseException:
            self.decoder.close()
            raise

    def close(self):
        self.decoder.close()

    def __enter__(self):
        self.decoder.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.decoder.__exit__(exc_type, exc_value, traceback)
