"""Retained-input runtime contracts with real native CPU and fake NVDEC workers."""

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ..interface_limits import build
from . import benchmark, prepare, probe, runtime


class FakeStream:
    """Hold one output until another AU or EOS, preserving all earlier inputs."""

    def __init__(self, workloads, width, height, mode='normal', **kwargs):
        self.candidates = [job for w in workloads.values() if
                           (w['case']['coded_width'], w['case']['coded_height']) == (width, height)
                           for job in w['jobs']]
        self.units = []
        self.closed = False
        self.mode = mode
        self.events = []
        self._stats = dict(requests=0, submitted_frames=0, decoded_frames=0,
                           copied_frames=0, pending_frames=0, input_bytes=0,
                           outstanding_input_bytes=0, pipeline_starts=1)

    @property
    def stats(self):
        return dict(self._stats)

    def append(self, units):
        if self.closed:
            raise AssertionError('Append after close')
        self.units.extend(units)
        self.candidates = [job for job in self.candidates if job['units'][:len(self.units)] == tuple(self.units)]
        if not self.candidates:
            raise AssertionError('Stream reloaded inputs, reset or changed instruction order')
        self.events.append(('append', len(units)))
        self._stats['requests'] += 1
        self._stats['submitted_frames'] += len(units)
        self._stats['pending_frames'] += len(units)
        self._stats['input_bytes'] += sum(map(len, units))
        self._stats['outstanding_input_bytes'] += sum(map(len, units))
        if self.mode == 'false-append':
            self._stats['decoded_frames'] += 1
        return self.stats

    def _collect(self, end, selected):
        start = self._stats['decoded_frames']
        chosen = tuple(index for index in selected if start <= index < end)
        frames = tuple((i, self.candidates[0]['frames'][i]) for i in chosen)
        self._stats['requests'] += 1
        self._stats['decoded_frames'] = end
        self._stats['copied_frames'] += len(frames)
        self._stats['pending_frames'] = len(self.units)-end
        self._stats['outstanding_input_bytes'] = sum(map(len, self.units[end:]))
        if frames and self.mode == 'bad-pixels':
            index, pixels = frames[0]
            frames = ((index, bytes([pixels[0] ^ 1])+pixels[1:]),)+frames[1:]
        return frames

    def collect(self, target, selected_frames, wait_ms):
        self.events.append(('collect', target, tuple(selected_frames), wait_ms))
        return self._collect(min(target, max(0, len(self.units)-1)), selected_frames)

    def close(self, selected_frames=()):
        if self.closed:
            return ()
        self.events.append(('eos', tuple(selected_frames)))
        frames = self._collect(len(self.units), selected_frames)
        self.closed = True
        if self.mode == 'bad-counts':
            self._stats['copied_frames'] += 1
        return frames


class FakeSession:
    """Existing fenced protocol with exact fixture pixels and one delayed fence."""

    protocol_version = 2

    def __init__(self, w, mode='normal'):
        self.w, self.mode = w, mode
        self.closed = False
        self.events = []
        self._process = mock.Mock(pid=12345)
        self._stats = dict(requests=0, submitted_frames=0, decoded_frames=0, copied_frames=0,
                           fence_pictures=0, decoded_fences=0, pending_fences=0,
                           pipeline_starts=1, fence_bytes=0, input_bytes=0)

    @property
    def stats(self):
        return dict(self._stats)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _request(self, units, selected, fence, operation):
        if self.closed or fence != self.w['fence']:
            raise AssertionError('Wrong fence or closed worker')
        index = next(i for i, job in enumerate(self.w['jobs']) if tuple(units) == job['units'])
        job = self.w['jobs'][index]
        selected = tuple(selected)
        self.events.append((operation, index, selected))
        prior = self._stats['pending_fences']
        self._stats['requests'] += 1
        self._stats['submitted_frames'] += len(units)+1
        self._stats['decoded_frames'] += len(units)+prior
        self._stats['decoded_fences'] += prior
        self._stats['fence_pictures'] += 1
        self._stats['fence_bytes'] += len(fence)
        self._stats['input_bytes'] += sum(map(len, units))+len(fence)
        self._stats['pending_fences'] = 1
        self._stats['copied_frames'] += len(selected)
        if self.mode == 'copied-completion' and operation == 'complete':
            self._stats['copied_frames'] += 1
        frames = tuple(job['frames'][i] for i in selected)
        if self.mode == 'bad-pixels' and frames:
            frames = (bytes([frames[0][0] ^ 1])+frames[0][1:],)+frames[1:]
        return frames

    def submit(self, units, selected_frames=None, *, fence):
        selected = range(len(units)) if selected_frames is None else selected_frames
        return self._request(units, selected, fence, 'submit')

    def complete(self, units, *, fence):
        return self._request(units, (), fence, 'complete')

    def close(self):
        if not self.closed:
            self._stats['decoded_frames'] += self._stats['pending_fences']
            self._stats['decoded_fences'] += self._stats['pending_fences']
            self._stats['pending_fences'] = 0
            if self.mode == 'bad-close-counts':
                self._stats['decoded_frames'] += 1
            self.closed = True


class FakeAccounting:
    def __init__(self, pid=None):
        self.tick = 0
        self.worker = pid is not None

    def snapshot(self):
        self.tick += 0.001
        result = dict(parent_cpu_s=self.tick, parent_user_s=self.tick, parent_system_s=0)
        if self.worker:
            result.update(worker_cpu_s=self.tick, worker_user_s=self.tick, worker_system_s=0)
        return result


class RetainedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='retained-runtime-tests-')
        cls.root = Path(cls.directory.name)
        cls.helpers, cls.bundle = cls.root/'helpers', cls.root/'bundle'
        build.build(cls.helpers)
        cls.export = prepare.prepare(cls.bundle, cpu_helper=cls.helpers/'libcpu.so',
                                     sizes=(32,), output_counts=(1, 8), decoders=('ffmpeg',))
        cls.arguments = dict(cpu_helper=cls.helpers/'libcpu.so', pcm_helper=cls.helpers/'libpcm.so',
                             nvdec_helper=Path(__file__), build_info=cls.helpers/'build.json')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def corpus(self):
        corpus = runtime.Corpus(self.bundle, **self.arguments)
        self.addCleanup(corpus.close)
        return corpus

    def test__given_verified_smoke_bundle__when_loading_native_runtime__then_every_output_and_prefix_match(self):
        corpus = self.corpus()
        self.assertEqual(len(corpus.workloads), 8)
        self.assertEqual(len(corpus.groups), 4)
        for w in corpus.workloads.values():
            for index in range(4):
                corpus.cpu(w, index)
                self.assertEqual(corpus.units(w, index, 'native-host'), w['jobs'][index]['units'])
            self.assertEqual(len(w['selection']), w['case']['output_count'])
        corpus.verify_stable()

    def test__given_altered_cpu_field__when_binding_to_semantic_ir__then_audit_rejects_it(self):
        case = copy.deepcopy(json.loads((self.bundle/'bundle.json').read_text())['cases']['32/motion-interpolation/8/tight'])
        manifest = json.loads(runtime.common.checked(self.bundle, case['manifest']))
        case['variants'][3]['cpu']['field_a'][0][0] += 1
        with self.assertRaisesRegex(AssertionError, 'semantic IR'):
            runtime.semantic_audit(case, manifest)

    def test__given_altered_capacity_or_output_mapping__when_loading__then_contract_is_rejected(self):
        original = (self.bundle/'bundle.json').read_bytes()
        for mutation in ('capacity', 'output', 'guards'):
            data = json.loads(original)
            case = data['cases']['32/motion-interpolation/8/tight']
            if mutation == 'capacity':
                case['reference_capacity'] = 16
            elif mutation == 'output':
                case['output_frames']['output_000'] += 1
            else:
                case['writer']['merge_zero_guards'] = False
            try:
                (self.bundle/'bundle.json').write_text(json.dumps(data))
                with self.subTest(mutation=mutation), self.assertRaisesRegex(AssertionError, 'contract'):
                    runtime.load(self.bundle)
            finally:
                (self.bundle/'bundle.json').write_bytes(original)

    def test__given_corrupted_reference_guard_with_updated_hash__when_loading__then_pixel_contract_rejects_it(self):
        original_bundle = (self.bundle/'bundle.json').read_bytes()
        data = json.loads(original_bundle)
        record = data['cases']['32/patch-atlas/8/tight']['jobs'][0]['expected']
        path = self.bundle/record['path']
        original = path.read_bytes()
        corrupted = bytes([original[0] ^ 1])+original[1:]
        try:
            path.write_bytes(corrupted)
            record['sha256'] = runtime.common.digest(corrupted)
            (self.bundle/'bundle.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError, 'guard|Guard'):
                runtime.load(self.bundle)
        finally:
            path.write_bytes(original)
            (self.bundle/'bundle.json').write_bytes(original_bundle)

    def test__given_prepared_and_native_requests__when_running__then_only_native_reencodes_inputs(self):
        corpus = self.corpus()
        w = corpus.workloads['32/motion-interpolation/8/tight']
        active = FakeSession(w)
        with mock.patch.object(w['encoder'], 'input_luma', wraps=w['encoder'].input_luma) as encoder:
            corpus.submit(active, w, 0, 'native-host')
            self.assertEqual(encoder.call_count, 2)
            corpus.submit(active, w, 1, 'prepared-host')
            corpus.submit(active, w, 2, 'prepared-complete')
            self.assertEqual(encoder.call_count, 2)
        self.assertEqual(active.stats['copied_frames'], 16)
        active.close()

    def test__given_partial_stream_outputs__when_probing__then_eos_checks_remaining_pixels_and_exact_counts(self):
        _, _, workloads, _ = runtime.load(self.bundle)
        streams = []
        def stream(*args, **kwargs):
            active = FakeStream(workloads, *args, **kwargs)
            streams.append(active)
            return active
        output = self.root/'fake-probe.json'
        with mock.patch.object(probe.runtime, 'NVDECStream', side_effect=stream):
            report = probe.run(self.bundle, output, nvdec_helper=Path(__file__), wait_ms=1)
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(len(streams), 32)
        self.assertTrue(all(active.closed for active in streams))
        self.assertEqual(json.loads(output.read_text())['status'], 'pass')
        for key, records in report['cases'].items():
            total = workloads[key]['case']['costs']['total_pictures']
            for record in records:
                self.assertEqual(record['before_eos']['pending_frames'], 1)
                self.assertEqual(record['after_close']['copied_frames'], total)
                self.assertEqual(record['after_close']['requests'], 2*total+2)
                self.assertEqual(record['after_close']['outstanding_input_bytes'], 0)
                self.assertEqual(record['output_boundaries_complete'], 0)
                self.assertEqual(sum(len(step['returned_indices']) for step in record['steps']), total-1)
        self.assertTrue(all(sum(event[0] == 'eos' for event in active.events) == 1 for active in streams))
        self.assertTrue(all(active.events[-2][0] == 'collect' and active.events[-2][-1] == 500
                            for active in streams))

    def test__given_false_completion_bad_pixels_or_counts__when_probing__then_failed_report_is_saved(self):
        _, _, workloads, _ = runtime.load(self.bundle)
        for mode in ('false-append', 'bad-pixels', 'bad-counts'):
            output = self.root/('failed-probe-'+mode+'.json')
            def stream(*args, **kwargs):
                return FakeStream(workloads, *args, mode=mode, **kwargs)
            with self.subTest(mode=mode), mock.patch.object(probe.runtime, 'NVDECStream', side_effect=stream):
                with self.assertRaises(AssertionError):
                    probe.run(self.bundle, output, nvdec_helper=Path(__file__), wait_ms=1)
            self.assertEqual(json.loads(output.read_text())['status'], 'failed')

    def test__given_completion_only_trial__when_timed__then_copies_postflight_and_fence_drain_are_separate(self):
        corpus = self.corpus()
        key = '32/motion-interpolation/8/tight'
        w = corpus.workloads[key]
        active = FakeSession(w)
        with mock.patch.object(corpus, 'session', return_value=active), \
                mock.patch.object(benchmark.runtime, 'Accounting', FakeAccounting):
            result = benchmark.trial(corpus, key, 'prepared-complete', 0)
        self.assertEqual([event[:2] for event in active.events],
                         [('complete', 3)]+[('complete', i) for i in range(4)]+[('submit', i) for i in range(4)])
        self.assertEqual(result['timed_copied_frames'], 0)
        self.assertEqual(result['postflight_full_i420_jobs'], 4)
        self.assertEqual(result['stats_after_timing']['copied_frames'], 0)
        self.assertEqual(result['stats_after_close']['copied_frames'], 40)
        self.assertEqual(result['stats_after_close']['requests'], 9)
        self.assertEqual(result['stats_after_close']['pending_fences'], 0)
        self.assertEqual(result['stats_after_close']['decoded_frames'], 99)
        self.assertEqual(result['requests'], 4)
        self.assertEqual(result['outputs_per_request'], 8)
        self.assertIn('worker_cpu_s', result)
        self.assertTrue(active.closed)

    def test__given_hidden_copies_or_bad_final_counts__when_timing__then_claim_is_rejected(self):
        corpus = self.corpus()
        key = '32/motion-interpolation/8/tight'
        for mode in ('copied-completion', 'bad-close-counts'):
            active = FakeSession(corpus.workloads[key], mode)
            with self.subTest(mode=mode), mock.patch.object(corpus, 'session', return_value=active), \
                    mock.patch.object(benchmark.runtime, 'Accounting', FakeAccounting):
                with self.assertRaisesRegex(AssertionError, 'counters'):
                    benchmark.trial(corpus, key, 'prepared-complete', 0)
            self.assertTrue(active.closed)

    def test__given_returned_guard_corruption__when_submitting_host_outputs__then_complete_i420_check_rejects_it(self):
        corpus = self.corpus()
        w = corpus.workloads['32/patch-atlas/8/tight']
        active = FakeSession(w, 'bad-pixels')
        with self.assertRaisesRegex(AssertionError, 'complete I420'):
            corpus.submit(active, w, 0, 'prepared-host')
        active.close()

    def test__given_fake_workers_and_native_cpu__when_full_benchmark_runs__then_report_normalization_and_order_are_exact(self):
        corpus = self.corpus()
        sessions = []
        def session(w):
            active = FakeSession(w)
            sessions.append(active)
            return active
        output = self.root/'fake-benchmark.json'
        with mock.patch.object(benchmark.runtime, 'Corpus', return_value=corpus), \
                mock.patch.object(corpus, 'session', side_effect=session), \
                mock.patch.object(benchmark.runtime, 'Accounting', FakeAccounting):
            report = benchmark.run(self.bundle, output, **self.arguments, repeats=2)
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(len(report['preflight']), 8)
        self.assertEqual(len(report['groups']), 4)
        self.assertNotEqual(report['order'][0], report['order'][1])
        self.assertEqual(report['bundle_sha256'], runtime.common.digest((self.bundle/'bundle.json').read_bytes()))
        self.assertIn('Unequal-result', report['contract']['prepared-complete'])
        self.assertIn('not interactive', report['contract']['scope'])
        self.assertTrue(all(active.closed for active in sessions))
        for group, record in report['groups'].items():
            self.assertEqual(len(record['modes']), 7)
            count = record['outputs_per_request']
            for label, mode in record['modes'].items():
                self.assertEqual(len(mode['trials']), 2)
                self.assertEqual(mode['wall_ms_per_output'], mode['wall_ms_per_job']/count)
                self.assertEqual(mode['cpu_ms_per_output'], mode['cpu_ms_per_job']/count)
                if label.endswith('prepared-complete'):
                    self.assertTrue(all(trial['timed_copied_frames'] == 0 for trial in mode['trials']))
                    self.assertTrue(all(trial['postflight_full_i420_jobs'] == 4 for trial in mode['trials']))
            for order in report['order']:
                self.assertEqual(sum(g == group and mode == 'cpu' for g, _, mode in order), 1)

    def test__given_runtime_entrypoints__when_parsed_for_python38__then_syntax_remains_portable(self):
        for name in ('runtime.py', 'probe.py', 'benchmark.py'):
            ast.parse(Path(__file__).with_name(name).read_text(), filename=name, feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
