"""Persistent exact NVDEC jobs over a private inherited Unix socket.

Only compiler-produced, four-byte Annex-B access units are supported. Each job
starts with SPS/PPS/IDR and ends with an explicit independent SPS/PPS/IDR fence.
The fence is a real extra decoded picture; it is never returned as a result.
The pipeline retains its DPB and stays PLAYING until close sends its sole EOS.

Wire v2 adds completion-only requests (zero selected pictures) to v1.
Clients still accept v1 workers for ordinary readback requests. All fields are
unsigned 64-bit network-order integers. Worker hello contains magic,
version, width, height, I420 bytes, maximum bytes. Request contains opcode (1
submit, 2 close), increasing id, real AU count, selected count, fence byte count,
zero; then indices, (AU byte count, AU bytes) pairs, fence bytes. Reply contains
magic, id, aggregate decoded count, aggregate copied count, decoded fence count,
pending fence count; selected packed I420 frames follow a submit reply. Plugin
stdout/stderr are drained independently and strictly validated at close.
"""

import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import threading
import time

from .decode import _NVDEC_FRAME, _NVDEC_STARTUP, _nvdec_helper

_MAGIC = 0x4E564453
_LIMIT = 128 * 1024 * 1024
_HEADER = struct.Struct('!6Q')
_U64 = struct.Struct('!Q')
_START = re.compile(b'\x00\x00\x00\x01')


def _unit(data, *, idr, idr_id=None):
    if type(data) is not bytes or not data.startswith(b'\x00\x00\x00\x01'):
        raise ValueError('Access units must be immutable four-byte Annex-B bytes')
    nals = _START.split(data)[1:]
    kinds, first_slices = [], 0
    for nal in nals:
        if len(nal) < 2 or nal[0] & 128:
            raise ValueError('Truncated or invalid NAL unit')
        kind = nal[0] & 31
        kinds.append(kind)
        if kind in (1, 5):
            first_slices += bool(nal[1] & 128)  # first_mb_in_slice == 0
        if kind == 5 and idr_id is not None:
            rbsp = nal[1:].replace(b'\0\0\3', b'\0\0')
            position = 0

            def bit():
                nonlocal position
                if position >= len(rbsp) * 8:
                    raise ValueError('Truncated IDR slice header')
                value = (rbsp[position // 8] >> (7 - position % 8)) & 1
                position += 1
                return value

            def ue():
                zeros = 0
                while not bit():
                    zeros += 1
                    if zeros > 31:
                        raise ValueError('Invalid IDR slice header')
                value = 1
                for _ in range(zeros):
                    value = value * 2 + bit()
                return value - 1

            ue(); ue(); ue()  # first_mb_in_slice, slice_type, pic_parameter_set_id
            for _ in range(16):  # Compiler SPS fixes log2_max_frame_num to 16.
                bit()
            if ue() != idr_id:
                raise ValueError('Compiler input IDR id must be 0 and fence IDR id must be 1')
    wanted = 5 if idr else 1
    if (first_slices != 1 or wanted not in kinds or
            any(k not in ((7, 8, 5) if idr else (1,)) for k in kinds) or
            (idr and (kinds[:2] != [7, 8] or kinds.count(7) != 1 or kinds.count(8) != 1))):
        raise ValueError('Access unit must contain exactly one picture with the required IDR boundary')


class NVDECSession:
    """One fixed-shape resident worker; any transport/decode error poisons it.

    ``width`` and ``height`` are coded dimensions, including compiler guards.
    Calls are sequential and must finish before another thread may use the
    session. Always close (prefer a context manager): delayed diagnostics and
    the final fence are checked at that point. Constructor handshake does not
    decode a warmup picture. Each job's IDR resets reference semantics; image
    surfaces remain owned by the decoder.
    """

    def __init__(self, width, height, helper=None, timeout=180):
        if any(type(n) is not int or n <= 0 or n % 2 for n in (width, height)):
            raise ValueError('Coded dimensions must be positive even integers')
        if width * height * 3 // 2 > _LIMIT:
            raise ValueError('Coded frame exceeds session byte limit')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Timeout must be finite and positive')
        self.width, self.height, self.timeout = width, height, float(timeout)
        self.frame_bytes = width * height * 3 // 2
        self._closed = False
        self._poisoned = False
        self._lock = threading.Lock()
        self._diagnostic_lock = threading.Lock()
        self._partial_lines = [b'', b'']
        self._diagnostic_error = None
        self._diagnostic_counts = {'decoded': None, 'copied': None}
        self._stats = dict(requests=0, submitted_frames=0, decoded_frames=0,
                           copied_frames=0, fence_pictures=0, fence_bytes=0,
                           decoded_fences=0, pending_fences=0, input_bytes=0,
                           pipeline_starts=0)
        executable = Path(helper).resolve() if helper is not None else _nvdec_helper()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError('Build decoder_core/build_nvdec.sh or supply an executable NVDEC helper')
        self._socket, child = socket.socketpair()
        self._process = None
        self._failure_exit_status = None
        self._threads = []
        try:
            self._process = subprocess.Popen(
                [str(executable), '--serve', str(child.fileno()), '--width', str(width), '--height', str(height)],
                pass_fds=(child.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            child.close()
            for index, pipe in enumerate((self._process.stdout, self._process.stderr)):
                thread = threading.Thread(target=self._drain, args=(index, pipe), daemon=True)
                thread.start(); self._threads.append(thread)
            self._deadline = time.monotonic() + self.timeout
            hello = _HEADER.unpack(self._read(_HEADER.size))
            if (hello[0] != _MAGIC or hello[1] not in (1, 2)
                    or hello[2:] != (width, height, self.frame_bytes, _LIMIT)):
                raise RuntimeError('Invalid NVDEC session handshake')
            self.protocol_version = hello[1]
            self._stats['pipeline_starts'] = 1
        except BaseException as error:
            child.close()
            self._abort()
            self._raise_worker_error(error)
            raise

    def _drain(self, index, pipe):
        try:
            while True:
                chunk = os.read(pipe.fileno(), 65536)
                with self._diagnostic_lock:
                    if self._diagnostic_error is not None:
                        # Still drain the pipe so a failing worker cannot deadlock.
                        if not chunk:
                            return
                        continue
                    lines = (self._partial_lines[index] + chunk).split(b'\n')
                    self._partial_lines[index] = lines.pop() if chunk else b''
                    for line in lines:
                        self._consume_line(line)
                    if not chunk:
                        return
                    if len(self._partial_lines[index]) > 4096:
                        self._diagnostic_error = 'NVDEC diagnostic line exceeds 4096 bytes'
                        self._partial_lines[index] = b''
        except OSError as error:
            with self._diagnostic_lock:
                self._diagnostic_error = 'Cannot read NVDEC diagnostics: ' + str(error)

    def _consume_line(self, raw):
        """Called under the diagnostic lock; consume each complete line once."""
        if self._diagnostic_error is not None:
            return
        if len(raw) > 4096:
            self._diagnostic_error = 'NVDEC diagnostic line exceeds 4096 bytes'
            return
        line = raw.decode(errors='replace').strip()
        if not line or _NVDEC_STARTUP.fullmatch(line):
            return
        for name, marker in (('decoded', 'NVDEC_DECODE_FRAMES'), ('copied', 'NVDEC_COPIED_FRAMES')):
            match = re.fullmatch(marker + r'=(\d+)', line)
            if match:
                if self._diagnostic_counts[name] is not None:
                    self._diagnostic_error = 'Duplicate NVDEC aggregate diagnostic count'
                else:
                    self._diagnostic_counts[name] = int(match.group(1))
                return
        frame = _NVDEC_FRAME.fullmatch(line)
        if frame and all(int(v, 16 if v.lower().startswith('0x') else 10) == 0
                         for v in (frame.group('error'), frame.group('concealed'))):
            return
        self._diagnostic_error = 'Unexpected NVDEC session diagnostic: ' + line[:1024]

    def _remaining(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('NVDEC session request timed out')
        self._socket.settimeout(remaining)

    def _read(self, size):
        result = bytearray()
        while len(result) < size:
            self._remaining()
            chunk = self._socket.recv(min(size - len(result), 1024 * 1024))
            if not chunk:
                raise RuntimeError('NVDEC session closed before its response completed')
            result.extend(chunk)
        return bytes(result)

    def _send(self, data):
        self._remaining()
        self._socket.sendall(data)

    @property
    def stats(self):
        return dict(self._stats)

    def submit(self, access_units, selected_frames=None, *, fence):
        """Decode a request and return its selected full I420 pictures."""
        return self._submit(access_units, selected_frames, fence=fence, completion_only=False)

    def complete(self, access_units, *, fence):
        """Decode every real picture without copying or returning image pixels.

        This diagnostic operation returns an empty tuple after all real pictures
        have reached the output queue. It still uploads/decodes the input and a
        counted fence; final close drains any pending fence. It does not expose
        or retain a surface for a GPU consumer, and cannot verify output pixels.
        Use full readback outside timing to validate the same inputs/program.
        Requires a protocol-v2 helper; ordinary submit remains v1 compatible.
        """
        if self._closed or self._poisoned:
            raise RuntimeError('NVDEC session is closed or poisoned')
        if self.protocol_version < 2:
            raise RuntimeError('Completion-only requests require a protocol-v2 NVDEC helper')
        return self._submit(access_units, (), fence=fence, completion_only=True)

    def _submit(self, access_units, selected_frames, *, fence, completion_only):
        if self._closed or self._poisoned:
            raise RuntimeError('NVDEC session is closed or poisoned')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('NVDEC session only accepts one request at a time')
        try:
            units = tuple(access_units)
            if not units or len(units) >= 32768:
                raise ValueError('Submit requires 1..32767 complete real pictures')
            selected = tuple(range(len(units))) if selected_frames is None else tuple(selected_frames)
            if ((not selected and not completion_only) or any(type(i) is not int or i < 0 or i >= len(units) for i in selected)
                    or any(a >= b for a, b in zip(selected, selected[1:]))):
                raise ValueError('Picture selection must be nonempty, sorted, unique and in range')
            if type(fence) is not bytes or any(type(unit) is not bytes for unit in units):
                raise ValueError('Access units and fence must be immutable bytes')
            total_bytes = sum(map(len, units)) + len(fence)
            if total_bytes > _LIMIT or len(selected) * self.frame_bytes > _LIMIT:
                raise ValueError('Session request exceeds 128 MiB input/readback limit')
            for i, unit in enumerate(units):
                _unit(unit, idr=i == 0, idr_id=0 if i == 0 else None)
            _unit(fence, idr=True, idr_id=1)
            self._deadline = time.monotonic() + self.timeout
            request = self._stats['requests'] + 1
            self._send(_HEADER.pack(1, request, len(units), len(selected), len(fence), 0))
            for index in selected:
                self._send(_U64.pack(index))
            for unit in units:
                self._send(_U64.pack(len(unit))); self._send(unit)
            self._send(fence)
            magic, identity, decoded, copied, fences, pending = _HEADER.unpack(self._read(_HEADER.size))
            submitted = self._stats['submitted_frames'] + len(units) + 1
            expected_copied = self._stats['copied_frames'] + len(selected)
            if (magic != _MAGIC or identity != request or pending not in (0, 1)
                    or decoded + pending != submitted or copied != expected_copied
                    or fences + pending != request):
                raise RuntimeError('Invalid NVDEC session reply identity or picture counts')
            result = tuple(self._read(self.frame_bytes) for _ in selected)
            self._stats.update(requests=request, submitted_frames=submitted,
                               decoded_frames=decoded, copied_frames=copied,
                               fence_pictures=request, decoded_fences=fences,
                               pending_fences=pending,
                               fence_bytes=self._stats['fence_bytes'] + len(fence),
                               input_bytes=self._stats['input_bytes'] + total_bytes)
            self._validate_logs(final=False)
            return result
        except BaseException as error:
            self._abort()
            self._raise_worker_error(error)
            raise
        finally:
            self._lock.release()

    def _validate_logs(self, *, final):
        with self._diagnostic_lock:
            if self._diagnostic_error is not None:
                raise RuntimeError(self._diagnostic_error)
            counts = self._diagnostic_counts
            if final and counts != {'decoded': self._stats['decoded_frames'], 'copied': self._stats['copied_frames']}:
                raise RuntimeError('NVDEC session aggregate diagnostic counts do not match replies')
            if not final and any(value is not None for value in counts.values()):
                raise RuntimeError('NVDEC helper finished before session close')

    def _abort(self):
        self._poisoned = True
        self._closed = True
        self._socket.close()
        if self._process is not None:
            try:
                # EOF may precede waitpid readiness by a few scheduling ticks.
                # Let an already-exiting worker preserve its real status/logs.
                self._failure_exit_status = self._process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process.wait(timeout=5)
        for thread in self._threads:
            thread.join(timeout=5)
        if self._process is not None:
            for pipe in (self._process.stdout, self._process.stderr):
                pipe.close()

    def _raise_worker_error(self, original):
        """Preserve the actual worker failure after pipes finish draining."""
        if not isinstance(original, (RuntimeError, OSError)):
            return
        with self._diagnostic_lock:
            diagnostic = self._diagnostic_error
        details = []
        if diagnostic is not None and diagnostic not in str(original):
            details.append(diagnostic)
        if self._failure_exit_status is not None:
            details.append('worker exit status ' + str(self._failure_exit_status))
        if details:
            raise RuntimeError(str(original) + '; ' + '; '.join(details)) from original

    def close(self):
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close while NVDEC request is in flight')
        try:
            self._deadline = time.monotonic() + self.timeout
            request = self._stats['requests'] + 1
            self._send(_HEADER.pack(2, request, 0, 0, 0, 0))
            response = _HEADER.unpack(self._read(_HEADER.size))
            if response != (_MAGIC, request, self._stats['submitted_frames'],
                            self._stats['copied_frames'], self._stats['fence_pictures'], 0):
                raise RuntimeError('NVDEC session did not drain exactly its final fence')
            self._stats.update(decoded_frames=response[2], decoded_fences=response[4], pending_fences=0)
            self._socket.close()
            if self._process.wait(timeout=max(0.001, self._deadline - time.monotonic())):
                raise RuntimeError('NVDEC session helper failed')
            for thread in self._threads:
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError('NVDEC session diagnostics did not finish')
            self._validate_logs(final=True)
            for pipe in (self._process.stdout, self._process.stderr):
                pipe.close()
            self._closed = True
        except BaseException as error:
            self._abort()
            self._raise_worker_error(error)
            raise
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self._abort()
        return False
