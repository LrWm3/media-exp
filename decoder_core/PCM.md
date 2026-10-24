# Optional native I_PCM input serializer

`NativePCMWriter` accelerates input serialization while preserving the existing
Writer's exact access-unit bytes and resource checks. It accepts compact u8
buffers, builds reusable header templates from the supplied writer, and delegates
macroblock filling and emulation-prevention escaping to an explicitly loaded C++
library. The ordinary Python writer remains the default and independent oracle.

```sh
bash decoder_core/build_pcm.sh /path/to/libpcm_native.so
```

```python
from decoder_core.bitstream import Writer
from decoder_core.pcm import NativePCMWriter

writer = Writer(256, 256, level_idc=41, slice_mode="row")
encoder = NativePCMWriter(writer, helper="/path/to/libpcm_native.so")
first_picture = encoder.input_luma(compact_u8_bytes, frame_index=0)
```

Build and load are explicit; inference never invokes a compiler. Return values
own their byte storage. The adapter caches frame-header templates and scratch
space, serializes calls with a lock, rejects changed writer configuration, and
checks the final escaped bytes against the writer's existing level policy.
Dimension/length/capacity checks also exist at the native boundary. It accepts
contiguous one-dimensional unsigned-byte buffers; partial read-only views can
require a compact copy. It does not encode arbitrary video formats or bypass
the current I_PCM size restrictions.

The [CPU comparison](../codec_compute/workloads/cpu_comparison/README.md) uses
this adapter for the host-input NVDEC path, measuring serialization separately
from warm decoding. It does not silently switch `Program.session.run` to a new
input representation. Source/compiler/binary hashes are preserved with that
experiment.

Thirteen tests compare exact bytes across levels 3.1 and 4.1, multiple byte
patterns, frame indices, reference plans and slice modes, check rejection and
configuration changes, exercise concurrent callers, and decode complete I420
through FFmpeg, OpenH264 and VA-API. The 82-test decoder-core suite passes after the level 5.1 extension. The
512×512 comparison separately checks native serialization against the Python
writer at level 5.1 for every exported input. Native
address/undefined-behavior sanitizers also pass rectangular geometry and capacity
checks. Run `python3 -m unittest decoder_core.test_pcm -v` to reproduce unit and
available-backend checks.
