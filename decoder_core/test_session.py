"""Resident protocol controls without requiring NVDEC hardware."""

from pathlib import Path
import sys
import tempfile
import unittest

from decoder_core.session import NVDECSession, _unit
from decoder_core.bitstream import Bits, nal

def idr(identity):
    bits = Bits()
    bits.ue(0); bits.ue(2); bits.ue(0); bits.u(0, 16); bits.ue(identity)
    return b'\0\0\0\1\x67\x80\0\0\0\1\x68\x80' + nal(5, bits)


_IDR = idr(0)
_FENCE = idr(1)
_P = b'\0\0\0\1\x61\x80'
_FAKE = r'''
import os,socket,struct,sys,time
s=socket.socket(fileno=int(sys.argv[2]));h=struct.Struct('!6Q');u=struct.Struct('!Q');magic=0x4e564453
w=int(sys.argv[4]);height=int(sys.argv[6]);size=w*height*3//2
mode=__MODE__

def read(n):
    out=b''
    while len(out)<n:
        p=s.recv(n-len(out))
        if not p:sys.exit(0)
        out+=p
    return out

def send(data):
    for i in range(0,len(data),7):s.sendall(data[i:i+7])

send(h.pack(magic if mode!='hello' else 1,1,w,height,size,128*1024*1024))
if mode=='partial':os.write(1,b'Opening in ')
submitted=copied=requests=0
while True:
    op,ident,n,k,fence,res=h.unpack(read(h.size))
    if op==2:
        if mode=='workerclose':
            os.write(2,b'NVDEC_DECODE_ERROR: test final fence drain failure\n')
            sys.exit(23)
        send(h.pack(magic,ident,submitted,copied,requests,0))
        if mode=='diagnostic':print('ErrorType= 1 Decoded MBs= 1 Concealed MBs= 0 FrameDecodeTime 0',file=sys.stderr)
        if mode=='partial':os.write(1,b'BLOCKING MODE \n')
        if mode=='late':os.write(1,b'ErrorType= 0 Decoded MBs= 1 Concealed MBs= 1 FrameDecodeTime 0')
        if mode=='duplicate':print('NVDEC_DECODE_FRAMES='+str(submitted),file=sys.stderr)
        print('NVDEC_DECODE_FRAMES='+str(submitted+(mode=='finalcounts')),file=sys.stderr)
        print('NVDEC_COPIED_FRAMES='+str(copied),file=sys.stderr)
        sys.exit(0)
    indices=[u.unpack(read(8))[0] for _ in range(k)]
    for _ in range(n):read(u.unpack(read(8))[0])
    read(fence)
    requests+=1;submitted+=n+1;copied+=k
    if mode=='partial':os.write(2,b'ErrorType= 0 Decoded MBs= 1 Concealed MBs= 0 FrameDecodeTime 0\n')
    if mode=='timeout':time.sleep(2)
    if mode=='truncated':sys.exit(0)
    send(h.pack(magic,ident+(mode=='identity'),submitted-1+(mode=='counts'),copied,requests-1,1))
    for index in indices:send(bytes([(requests+index)%256])*size)
'''


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def helper(self, mode='normal'):
        path = Path(self.directory.name) / ('helper-' + mode)
        path.write_text('#!' + sys.executable + '\n' + _FAKE.replace('__MODE__', repr(mode)))
        path.chmod(0o700)
        return path

    def test__given_two_jobs__when_selecting_outputs__then_boundaries_counts_and_final_fence_are_exact(self):
        with NVDECSession(16, 16, helper=self.helper()) as session:
            self.assertEqual(session.submit((_IDR, _P, _P), (0, 2), fence=_FENCE),
                             (bytes([1])*384, bytes([3])*384))
            self.assertEqual(session.submit((_IDR,), fence=_FENCE), (bytes([2])*384,))
            self.assertEqual(session.stats['pending_fences'], 1)
            self.assertEqual(session.stats['decoded_frames'], 5)
        self.assertEqual(session.stats, dict(requests=2, submitted_frames=6, decoded_frames=6,
                         copied_frames=3, fence_pictures=2, fence_bytes=2*len(_FENCE),
                         decoded_fences=2, pending_fences=0,
                         input_bytes=2*len(_IDR)+2*len(_FENCE)+2*len(_P), pipeline_starts=1))
        session.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            session.submit((_IDR,), fence=_FENCE)

    def test__given_no_jobs__when_closing__then_zero_counts_are_checked(self):
        with NVDECSession(16, 16, helper=self.helper()) as session:
            pass
        self.assertEqual(session.stats['decoded_frames'], 0)

    def test__given_bad_reply_or_transport__when_submitting__then_session_is_poisoned(self):
        for mode in ('identity', 'counts', 'truncated', 'timeout'):
            with self.subTest(mode=mode):
                session = NVDECSession(16, 16, helper=self.helper(mode), timeout=0.5)
                with self.assertRaises((RuntimeError, TimeoutError)):
                    session.submit((_IDR, _P), fence=_FENCE)
                with self.assertRaisesRegex(RuntimeError, 'poisoned'):
                    session.submit((_IDR,), fence=_FENCE)
                session.close()

    def test__given_concealment_or_bad_handshake__when_validating__then_failure_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, 'handshake'):
            NVDECSession(16, 16, helper=self.helper('hello'))
        with self.assertRaisesRegex(RuntimeError, 'diagnostic'):
            with NVDECSession(16, 16, helper=self.helper('diagnostic')) as session:
                session.submit((_IDR,), fence=_FENCE)

    def test__given_malformed_units_or_selection__when_submitting__then_no_request_is_accepted(self):
        cases = (((_IDR,), (1,), _IDR), ((_IDR, _P), (1, 0), _IDR),
                 ((_IDR,), (True,), _IDR), ((_P,), None, _IDR),
                 ((_IDR,), None, _P), ((_IDR,), None, _IDR), ((_IDR, _IDR), None, _IDR),
                 ((_IDR + _P,), None, _IDR))
        for units, selected, fence in cases:
            with self.subTest(units=units, selected=selected):
                session = NVDECSession(16, 16, helper=self.helper())
                with self.assertRaises(ValueError):
                    session.submit(units, selected, fence=fence)
                self.assertEqual(session.stats['requests'], 0)
                session.close()
        for unit in (b'', b'\0\0\1\x65\x80', _IDR[:16], _IDR + _IDR, _IDR + b'\0\0\0\1',
                     _IDR[6:12] + _IDR[:6] + _IDR[12:]):
            with self.assertRaises(ValueError):
                _unit(unit, idr=True)

    def test__given_invalid_dimensions_or_timeout__when_opening__then_rejected_before_process(self):
        for width, height in ((True, 16), (0, 16), (17, 16), (16, -2)):
            with self.assertRaises(ValueError):
                NVDECSession(width, height)
        for timeout in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                NVDECSession(16, 16, timeout=timeout)

    def test__given_partial_lines_and_reused_session__when_consuming_diagnostics__then_history_stays_bounded(self):
        with NVDECSession(16, 16, helper=self.helper('partial')) as session:
            for _ in range(100):
                session.submit((_IDR,), fence=_FENCE)
            self.assertEqual(session.stats['requests'], 100)
            self.assertLessEqual(sum(map(len, session._partial_lines)), 32)
        self.assertEqual(session._partial_lines, [b'', b''])
        self.assertEqual(session._diagnostic_counts, {'decoded': 200, 'copied': 100})

    def test__given_late_unterminated_error_or_bad_final_counts__when_closing__then_failure_is_retained(self):
        for mode in ('late', 'duplicate', 'finalcounts'):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'diagnostic'):
                with NVDECSession(16, 16, helper=self.helper(mode)) as session:
                    session.submit((_IDR,), fence=_FENCE)

    def test__given_worker_failure_before_close_reply__when_cleanup_drains_stderr__then_native_error_survives(self):
        with self.assertRaisesRegex(RuntimeError, 'test final fence drain failure'):
            with NVDECSession(16, 16, helper=self.helper('workerclose')) as session:
                session.submit((_IDR,), fence=_FENCE)
        self.assertEqual(session._process.returncode, 23)
        with self.assertRaisesRegex(RuntimeError, 'closed or poisoned'):
            session.submit((_IDR,), fence=_FENCE)


if __name__ == '__main__':
    unittest.main()
