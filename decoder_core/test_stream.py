"""Continuous-stream framing, bounded progress and late diagnostics without GPU."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from decoder_core.stream import NVDECStream

_IDR = b'\0\0\0\1g\x80\0\0\0\1h\x80\0\0\0\1e\xb8\0\x06'
_P = b'\0\0\0\1\x61\x80'

_FAKE = r'''
import os,socket,struct,sys,time
s=socket.socket(fileno=int(sys.argv[2]));h=struct.Struct('!8Q');u=struct.Struct('!Q');magic=0x4e565354
assert sys.argv[1]=='--stream'
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
    for i in range(0,len(data),13):s.sendall(data[i:i+13])
send(h.pack(magic if mode!='hello' else 1,2 if mode=='version' else 1,w,height,size,128*1024*1024,32767,60000))
sizes=[];decoded=copied=requests=0
while True:
    op,ident,target,k,wait,r1,r2,r3=h.unpack(read(h.size))
    assert ident==requests+1 and not (r1 or r2 or r3)
    selected=[u.unpack(read(8))[0] for _ in range(k)]
    returned=[]
    if op==1:
        assert not k and not wait
        for _ in range(target):
            n=u.unpack(read(8))[0];read(n);sizes.append(n)
    else:
        if op==3 and mode=='workerclose':
            os.write(2,b'NVDEC_DECODE_ERROR: test continuous drain failure\n');sys.exit(23)
        end=len(sizes) if op==3 else target
        if op==2:
            if mode=='partial':end=min(end,decoded+1)
            if mode=='lookahead':end=min(end,max(0,len(sizes)-1))
            if mode=='stalled':end=decoded
            if mode=='transport-timeout':time.sleep(2)
            if mode=='truncated':sys.exit(0)
        returned=[i for i in selected if i<end]
        decoded=end;copied+=len(returned)
    requests+=1
    corrupt=op==2
    send(h.pack(magic,ident+int(corrupt and mode=='identity'),
                len(sizes)+int(corrupt and mode=='submitted'),
                decoded+int(corrupt and mode=='decoded'),
                copied+int(corrupt and mode=='copied'),
                len(sizes)-decoded+int(corrupt and mode=='pending'),
                sum(sizes[decoded:])+int(corrupt and mode=='outstanding'),
                len(returned)+int(corrupt and mode=='returned')))
    for i in returned:
        send(u.pack(i+int(corrupt and mode=='index')))
        send(bytes([i%256])*size)
    if op==3:
        if mode=='late':os.write(2,b'ErrorType= 0 Decoded MBs= 1 Concealed MBs= 1 FrameDecodeTime 0')
        if mode=='duplicate':print('NVDEC_DECODE_FRAMES='+str(decoded),file=sys.stderr)
        print('NVDEC_DECODE_FRAMES='+str(decoded+int(mode=='finalcounts')),file=sys.stderr)
        print('NVDEC_COPIED_FRAMES='+str(copied),file=sys.stderr)
        sys.exit(0)
'''


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def helper(self, mode='normal'):
        path = Path(self.directory.name) / ('stream-' + mode)
        path.write_text('#!' + sys.executable + '\n' + _FAKE.replace('__MODE__', repr(mode)))
        path.chmod(0o700)
        return path

    def test__given_partial_collection__when_appending_predictions__then_completion_continues_without_reset(self):
        with NVDECStream(16, 16, helper=self.helper('lookahead')) as stream:
            first = stream.append((_IDR, _P))
            self.assertEqual(first['decoded_frames'], 0)
            self.assertEqual(first['pending_frames'], 2)
            self.assertEqual(stream.collect(2, (0, 1), wait_ms=1), ((0, bytes(384)),))
            self.assertEqual(stream.stats['decoded_frames'], 1)
            stream.append((_P,))
            self.assertEqual(stream.collect(3, (1, 2), wait_ms=1), ((1, bytes([1])*384),))
            self.assertEqual(stream.close((2,)), ((2, bytes([2])*384),))
        self.assertEqual(stream.stats, dict(requests=5, submitted_frames=3, decoded_frames=3,
                         copied_frames=3, pending_frames=0, input_bytes=len(_IDR)+2*len(_P),
                         outstanding_input_bytes=0, pipeline_starts=1))
        self.assertEqual(stream.close(), ())

    def test__given_partial_selected_outputs__when_repeating_remaining_indices__then_each_pixel_returns_once(self):
        with NVDECStream(16, 16, helper=self.helper('partial')) as stream:
            stream.append((_IDR, _P, _P))
            self.assertEqual(stream.collect(3, (0, 2)), ((0, bytes(384)),))
            self.assertEqual(stream.collect(3, (2,)), ())
            self.assertEqual(stream.collect(3, (2,)), ((2, bytes([2])*384),))
            self.assertEqual(stream.stats['copied_frames'], 2)
            self.assertEqual(stream.stats['pending_frames'], 0)

    def test__given_empty_selection__when_collecting__then_samples_complete_without_copies(self):
        with NVDECStream(16, 16, helper=self.helper()) as stream:
            stream.append((_IDR, _P, _P))
            self.assertEqual(stream.collect(2), ())
            self.assertEqual(stream.stats['decoded_frames'], 2)
            self.assertEqual(stream.stats['copied_frames'], 0)
            with self.assertRaisesRegex(ValueError, 'undecoded'):
                stream.collect(3, (0,))
            self.assertEqual(stream.close((2,)), ((2, bytes([2])*384),))

    def test__given_bounded_wait_expiry__when_collecting_again__then_stream_remains_usable(self):
        with NVDECStream(16, 16, helper=self.helper('stalled')) as stream:
            stream.append((_IDR,))
            for budget in (0, 1, 10):
                self.assertEqual(stream.collect(1, (0,), wait_ms=budget), ())
                self.assertEqual(stream.stats['decoded_frames'], 0)
            stream.append((_P,))
            self.assertEqual(stream.close((0, 1)), ((0, bytes(384)), (1, bytes([1])*384)))

    def test__given_invalid_au_boundaries__when_appending__then_no_bytes_submit_and_stream_can_continue(self):
        with NVDECStream(16, 16, helper=self.helper()) as stream:
            for units in ((), (_P,), (_IDR, _IDR), (_IDR+_P,), (bytearray(_IDR),)):
                with self.subTest(units=units), self.assertRaises(ValueError):
                    stream.append(units)
            self.assertEqual(stream.stats['requests'], 0)
            stream.append((_IDR,))
            with self.assertRaises(ValueError):
                stream.append((_IDR,))
            self.assertEqual(stream.stats['submitted_frames'], 1)
            stream.append((_P,))
            stream.collect()

    def test__given_invalid_target_selection_or_wait__when_collecting__then_validation_preserves_state(self):
        with NVDECStream(16, 16, helper=self.helper()) as stream:
            stream.append((_IDR, _P))
            for target in (True, -1, 3):
                with self.assertRaises(ValueError):
                    stream.collect(target)
            for indices in ((True,), (-1,), (2,), (1, 0), (0, 0)):
                with self.assertRaises(ValueError):
                    stream.collect(2, indices)
            for wait in (True, -1, 60001, 0.1):
                with self.assertRaises(ValueError):
                    stream.collect(wait_ms=wait)
            self.assertEqual(stream.stats['requests'], 1)
            stream.collect(1)
            with self.assertRaises(ValueError):
                stream.collect(0)
            with self.assertRaises(ValueError):
                stream.close((0,))
            self.assertEqual(stream.close((1,)), ((1, bytes([1])*384),))

    def test__given_outstanding_bytes__when_append_exceeds_aggregate_limit__then_collection_releases_budget(self):
        with NVDECStream(16, 16, helper=self.helper()) as stream:
            stream.append((_IDR,))
            with patch('decoder_core.stream._LIMIT', len(_IDR)+len(_P)-1):
                with self.assertRaisesRegex(ValueError, 'outstanding'):
                    stream.append((_P,))
                self.assertEqual(stream.stats['submitted_frames'], 1)
                stream.collect()
                stream.append((_P,))
            self.assertEqual(stream.stats['outstanding_input_bytes'], len(_P))

    def test__given_picture_budget__when_cumulative_append_overflows__then_rejected_before_transmission(self):
        with NVDECStream(16, 16, helper=self.helper()) as stream:
            with patch('decoder_core.stream._MAX_PICTURES', 3):
                stream.append((_IDR, _P))
                stream.collect()
                stream.append((_P,))
                with self.assertRaisesRegex(ValueError, 'cumulative'):
                    stream.append((_P,))
            self.assertEqual(stream.stats['submitted_frames'], 3)

    def test__given_readback_budget__when_selecting_excess_pixels__then_request_is_rejected(self):
        with NVDECStream(16, 16, helper=self.helper()) as stream:
            stream.append((_IDR, _P))
            with patch('decoder_core.stream._LIMIT', 384):
                with self.assertRaisesRegex(ValueError, 'readback'):
                    stream.collect(2, (0, 1))
            self.assertEqual(stream.stats['requests'], 1)

    def test__given_wrong_counts_or_index__when_collecting__then_transport_is_poisoned(self):
        for mode in ('identity', 'submitted', 'decoded', 'copied', 'pending', 'outstanding', 'returned', 'index', 'truncated'):
            with self.subTest(mode=mode):
                stream = NVDECStream(16, 16, helper=self.helper(mode))
                stream.append((_IDR, _P))
                with self.assertRaisesRegex(RuntimeError, 'stream|response'):
                    stream.collect(2, (1,))
                with self.assertRaisesRegex(RuntimeError, 'poisoned'):
                    stream.append((_P,))
                stream.close()

    def test__given_transport_timeout__when_worker_never_replies__then_stream_is_poisoned(self):
        stream = NVDECStream(16, 16, helper=self.helper('transport-timeout'), timeout=0.3)
        stream.append((_IDR,))
        with self.assertRaises(TimeoutError):
            stream.collect(wait_ms=0)
        with self.assertRaisesRegex(RuntimeError, 'poisoned'):
            stream.collect()
        stream.close()

    def test__given_delayed_worker_errors__when_closing__then_errors_survive_cleanup(self):
        for mode in ('late', 'duplicate', 'finalcounts', 'workerclose'):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'diagnostic|continuous drain failure'):
                with NVDECStream(16, 16, helper=self.helper(mode)) as stream:
                    stream.append((_IDR, _P))
                    stream.collect()

    def test__given_bad_handshake_or_arguments__when_opening__then_rejected(self):
        for mode in ('hello', 'version'):
            with self.assertRaisesRegex(RuntimeError, 'handshake'):
                NVDECStream(16, 16, helper=self.helper(mode))
        for width, height in ((True, 16), (0, 16), (17, 16), (16, -2)):
            with self.assertRaises(ValueError):
                NVDECStream(width, height)
        for timeout in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                NVDECStream(16, 16, timeout=timeout)

    def test__given_no_appends__when_closing__then_zero_counts_match(self):
        with NVDECStream(16, 16, helper=self.helper()) as stream:
            self.assertEqual(stream.collect(wait_ms=0), ())
        self.assertEqual(stream.stats['decoded_frames'], 0)
        self.assertEqual(stream.stats['pending_frames'], 0)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            stream.append((_IDR,))


if __name__ == '__main__':
    unittest.main()
