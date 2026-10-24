"""Bounded continuous NVDEC access units; appending never means completion.

Only the first picture contains SPS/PPS/IDR. Subsequent appends contain non-IDR
pictures and keep one decoder pipeline and its prediction references alive.
``collect`` waits for output samples, may return partial progress at its bounded
wait deadline, and discards unselected output pixels. Only ``close`` sends EOS.

Private NVST wire v1 uses unsigned 64-bit network-order integers. Hello has
magic, version, width, height, I420 bytes, byte limit, picture limit, wait limit.
Requests have opcode (1 append, 2 collect, 3 close), increasing id, AU count or
cumulative target (zero for close), selected count, wait milliseconds, 0, 0, 0.
Append payload has (AU byte count, AU bytes) pairs; collection/close payload has
absolute selected indices. Replies have magic, id, submitted, decoded, copied,
pending, outstanding input bytes, returned count; each returned picture has its
absolute index followed by packed I420 bytes. Decode counts mean samples pulled
from appsink, not merely input buffers accepted. An omitted picture is discarded
and cannot be selected later. Native decoded reference surfaces remain owned by
the decoder; this interface does not expose or import them.
"""

import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time

from .decode import _nvdec_helper
from .session import NVDECSession, _unit

_MAGIC = 0x4E565354
_LIMIT = 128 * 1024 * 1024
_MAX_PICTURES = 32767
_MAX_WAIT_MS = 60000
_HEADER = struct.Struct('!8Q')
_U64 = struct.Struct('!Q')


class NVDECStream:
    """One fixed-shape stream, bounded to 32767 pictures and 128 MiB pending input.

    Calls must be sequential. Transport, decode and diagnostic failures poison
    the stream; a successful partial ``collect`` does not. Input validation
    failures before transmission also leave the stream usable. Always close to
    drain pending output and validate delayed diagnostics.
    """

    # Reuse the resident worker's exact transport and diagnostic handling while
    # keeping its job/fence protocol and public API separate.
    _drain = NVDECSession._drain
    _consume_line = NVDECSession._consume_line
    _remaining = NVDECSession._remaining
    _read = NVDECSession._read
    _send = NVDECSession._send
    _validate_logs = NVDECSession._validate_logs
    _abort = NVDECSession._abort
    _raise_worker_error = NVDECSession._raise_worker_error
    __enter__ = NVDECSession.__enter__
    __exit__ = NVDECSession.__exit__

    def __init__(self, width, height, helper=None, timeout=180):
        if any(type(n) is not int or n <= 0 or n % 2 for n in (width, height)):
            raise ValueError('Coded dimensions must be positive even integers')
        if width * height * 3 // 2 > _LIMIT:
            raise ValueError('Coded frame exceeds stream byte limit')
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Timeout must be finite and positive')
        self.width, self.height, self.timeout = width, height, float(timeout)
        self.frame_bytes = width * height * 3 // 2
        self._closed = self._poisoned = False
        self._lock = threading.Lock()
        self._diagnostic_lock = threading.Lock()
        self._partial_lines = [b'', b'']
        self._diagnostic_error = None
        self._diagnostic_counts = {'decoded': None, 'copied': None}
        self._stats = dict(requests=0, submitted_frames=0, decoded_frames=0,
                           copied_frames=0, pending_frames=0, input_bytes=0,
                           outstanding_input_bytes=0, pipeline_starts=0)
        self._unit_sizes = []
        executable = Path(helper).resolve() if helper is not None else _nvdec_helper()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError('Build decoder_core/build_nvdec.sh or supply an executable NVDEC helper')
        self._socket, child = socket.socketpair()
        self._process = None
        self._failure_exit_status = None
        self._threads = []
        try:
            self._process = subprocess.Popen(
                [str(executable), '--stream', str(child.fileno()), '--width', str(width), '--height', str(height)],
                pass_fds=(child.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            child.close()
            for index, pipe in enumerate((self._process.stdout, self._process.stderr)):
                thread = threading.Thread(target=self._drain, args=(index, pipe), daemon=True)
                thread.start(); self._threads.append(thread)
            self._deadline = time.monotonic() + self.timeout
            hello = _HEADER.unpack(self._read(_HEADER.size))
            if hello != (_MAGIC, 1, width, height, self.frame_bytes, _LIMIT, _MAX_PICTURES, _MAX_WAIT_MS):
                raise RuntimeError('Invalid NVDEC stream handshake')
            self.protocol_version = 1
            self._stats['pipeline_starts'] = 1
        except BaseException as error:
            child.close()
            self._abort()
            self._raise_worker_error(error)
            raise

    @property
    def stats(self):
        return dict(self._stats)

    def _acquire(self):
        if self._closed or self._poisoned:
            raise RuntimeError('NVDEC stream is closed or poisoned')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('NVDEC stream only accepts one request at a time')

    def _selection(self, selected_frames, target):
        selected = tuple(selected_frames)
        start = self._stats['decoded_frames']
        if (any(type(index) is not int or not start <= index < target for index in selected)
                or any(a >= b for a, b in zip(selected, selected[1:]))):
            raise ValueError('Selection must be sorted unique undecoded absolute picture indices below target')
        if len(selected) * self.frame_bytes > _LIMIT:
            raise ValueError('Stream readback exceeds 128 MiB limit')
        return selected

    def append(self, access_units):
        """Queue complete AUs and return counters; this does not await completion."""
        self._acquire()
        try:
            units = tuple(access_units)
            previous = self._stats['submitted_frames']
            if not units or previous + len(units) > _MAX_PICTURES:
                raise ValueError('Stream requires 1..32767 cumulative complete pictures')
            if any(type(unit) is not bytes for unit in units):
                raise ValueError('Access units must be immutable bytes')
            total = sum(map(len, units))
            if total + self._stats['outstanding_input_bytes'] > _LIMIT:
                raise ValueError('Stream outstanding input exceeds 128 MiB limit')
            for i, unit in enumerate(units):
                initial = previous == 0 and i == 0
                _unit(unit, idr=initial, idr_id=0 if initial else None)
            self._exchange(1, len(units), (), 0, units)
            return self.stats
        finally:
            self._lock.release()

    def collect(self, target=None, selected_frames=(), wait_ms=1000):
        """Return ``(absolute_index, I420_bytes)`` pairs observed within a budget.

        Target is the cumulative output count to await (default all submitted).
        An empty selection waits without copying. The wait budget is 0..60000
        milliseconds, excluding transport allowance. Partial progress returns
        normally: consult ``stats['decoded_frames']`` and repeat only still
        undecoded selections. Already collected unselected pictures are gone.
        """
        self._acquire()
        try:
            if target is None:
                target = self._stats['submitted_frames']
            if (type(target) is not int or not self._stats['decoded_frames'] <= target
                    <= self._stats['submitted_frames']):
                raise ValueError('Collection target must be between decoded and submitted counts')
            if type(wait_ms) is not int or not 0 <= wait_ms <= _MAX_WAIT_MS:
                raise ValueError('Collection wait must be 0..60000 integer milliseconds')
            selected = self._selection(selected_frames, target)
            return self._exchange(2, target, selected, wait_ms)
        finally:
            self._lock.release()

    def _exchange(self, opcode, target, selected, wait_ms, units=()):
        """Only entered after validation. Any failure here poisons the stream."""
        try:
            self._deadline = time.monotonic() + self.timeout + wait_ms / 1000
            request = self._stats['requests'] + 1
            self._send(_HEADER.pack(opcode, request, target, len(selected), wait_ms, 0, 0, 0))
            for index in selected:
                self._send(_U64.pack(index))
            for unit in units:
                self._send(_U64.pack(len(unit))); self._send(unit)
            magic, identity, submitted, decoded, copied, pending, outstanding, returned = _HEADER.unpack(
                self._read(_HEADER.size))
            sizes = self._unit_sizes + [len(unit) for unit in units]
            prior_decoded = self._stats['decoded_frames']
            expected_submitted = self._stats['submitted_frames'] + len(units)
            end = expected_submitted if opcode == 3 else target
            expected_returned = tuple(index for index in selected if index < decoded)
            if (magic != _MAGIC or identity != request or submitted != expected_submitted
                    or not prior_decoded <= decoded <= submitted
                    or (opcode == 1 and decoded != prior_decoded)
                    or (opcode == 2 and decoded > end)
                    or (opcode == 3 and decoded != submitted)
                    or pending != submitted - decoded or outstanding != sum(sizes[decoded:])
                    or returned != len(expected_returned)
                    or copied != self._stats['copied_frames'] + returned):
                raise RuntimeError('Invalid NVDEC stream reply identity or picture/byte counts')
            frames = []
            for index in expected_returned:
                if _U64.unpack(self._read(_U64.size))[0] != index:
                    raise RuntimeError('Invalid NVDEC stream returned picture index')
                frames.append((index, self._read(self.frame_bytes)))
            self._unit_sizes = sizes
            self._stats.update(requests=request, submitted_frames=submitted, decoded_frames=decoded,
                               copied_frames=copied, pending_frames=pending, outstanding_input_bytes=outstanding,
                               input_bytes=self._stats['input_bytes'] + sum(map(len, units)))
            if opcode != 3:
                self._validate_logs(final=False)
            return tuple(frames)
        except BaseException as error:
            self._abort()
            self._raise_worker_error(error)
            raise

    def close(self, selected_frames=()):
        """Send sole EOS, drain all pending pictures, return selected pending I420."""
        if self._closed:
            return ()
        self._acquire()
        try:
            selected = self._selection(selected_frames, self._stats['submitted_frames'])
            result = self._exchange(3, 0, selected, 0)
            try:
                self._socket.close()
                if self._process.wait(timeout=max(0.001, self._deadline - time.monotonic())):
                    raise RuntimeError('NVDEC stream helper failed')
                for thread in self._threads:
                    thread.join(timeout=5)
                    if thread.is_alive():
                        raise RuntimeError('NVDEC stream diagnostics did not finish')
                self._validate_logs(final=True)
                for pipe in (self._process.stdout, self._process.stderr):
                    pipe.close()
                self._closed = True
                return result
            except BaseException as error:
                self._abort()
                self._raise_worker_error(error)
                raise
        finally:
            self._lock.release()
