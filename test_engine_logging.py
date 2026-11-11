"""Deterministic integration tests. Fake modules prevent any YOLO/video execution.

All files, including the engine's redirected outputs directory, live in an OS
temporary directory. No project outputs or existing evaluation files are touched.
"""
import csv
import hashlib
import json
import sys
import tempfile
import threading
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import engine
from counter import LineCounter
from run_logging import LoggingError, RunLogger


class FakeFrame:
    def __init__(self, width=100, height=100):
        self.shape = (height, width, 3)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def int(self):
        return FakeTensor([int(value) for value in self.values])

    def tolist(self):
        return list(self.values)


class FakeBoxes:
    def __init__(self, detections):
        self.xyxy = FakeTensor([row[:4] for row in detections])
        self.id = None if detections and detections[0][4] is None else FakeTensor([row[4] for row in detections])
        self.cls = FakeTensor([row[5] for row in detections])
        self.conf = FakeTensor([row[6] for row in detections])


def detection(y, identity=1, x=50, cls=0):
    return (x-5, y-10, x+5, y, identity, cls, 0.875)


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 2, 3, 4, 5, 123456, tzinfo=tz)


class EngineLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='doan1-engine-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.messages = []
        self.timeline = []
        self.run_folder = None
        self.case = 0

    def execute(self, script, *, camera=False, fps=25.0, width=100, height=100,
                axis='horizontal', position=0.5, stop_after=None, prestop=False,
                opened=True, release_error=None, plot_error_on=None, publish_error=None,
                model_error=None, collision=False):
        self.case += 1
        root = self.root / f'case{self.case}'
        root.mkdir()
        source = root / 'người.mp4'
        source.write_bytes(b'synthetic video data, not a real video')
        weights = root / 'dummy.pt'
        weights.write_bytes(b'synthetic weights, never loaded')
        tracker = root / 'fake_ultra/cfg/trackers/bytetrack.yaml'
        tracker.parent.mkdir(parents=True)
        tracker.write_text('tracker_type: bytetrack\n', encoding='utf-8')
        self.run_folder = root / 'outputs/20260102_030405_123456'
        if collision:
            self.run_folder.mkdir(parents=True)
            (self.run_folder / 'events.csv').write_text('existing data', encoding='utf-8')
        stop = threading.Event()
        if prestop:
            stop.set()
        read_index = 0
        self.capture = Mock()
        self.capture.isOpened.return_value = opened
        self.capture.get.side_effect = lambda prop: {
            3: width, 4: height, 5: fps, 7: len(script),
        }[prop]

        def read():
            nonlocal read_index
            if read_index == len(script):
                return False, None
            read_index += 1
            return True, FakeFrame(width, height)

        self.capture.read.side_effect = read
        self.capture.release.side_effect = release_error
        cv2 = types.ModuleType('cv2')
        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT = 3, 4
        cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT = 5, 7
        cv2.COLOR_BGR2RGB, cv2.FONT_HERSHEY_SIMPLEX = 1, 0
        cv2.VideoCapture = Mock(return_value=self.capture)
        cv2.resize = Mock(side_effect=lambda f, _, fx, fy: FakeFrame(round(f.shape[1]*fx), round(f.shape[0]*fy)))
        cv2.line, cv2.putText = Mock(), Mock()
        cv2.cvtColor = Mock(side_effect=lambda f, _: f)
        self.cv2 = cv2
        self.model = Mock()
        self.model.ckpt_path = str(weights)
        self.model.names = {0: 'person', 2: 'car'}
        sequence = iter(script)
        tracked = 0

        def track(frame, **kwargs):
            nonlocal tracked
            tracked += 1
            item = next(sequence)
            if isinstance(item, BaseException):
                raise item
            result = Mock()
            result.boxes = None if item is None else FakeBoxes(item)
            result.plot.side_effect = RuntimeError('plot failed') if tracked == plot_error_on else None
            result.plot.return_value = frame
            return [result]

        self.model.track.side_effect = track
        ultra = types.ModuleType('ultralytics')
        ultra.__file__ = str(root / 'fake_ultra/__init__.py')
        ultra.YOLO = Mock(return_value=self.model, side_effect=model_error)
        self.ultra = ultra
        timeline = self.timeline

        class ObservedCounter(LineCounter):
            def update(counter, identity, label, point, frame, w, h):
                timeline.append(('update', frame, identity))
                return super().update(identity, label, point, frame, w, h)

            def prune(counter, frame):
                timeline.append(('prune', frame))
                return super().prune(frame)

        def publish(data):
            self.messages.append(data)
            if 'frame' in data:
                if stop_after and len([m for m in self.messages if 'frame' in m]) == stop_after:
                    stop.set()
                if publish_error is not None:
                    raise publish_error

        config = dict(source='0' if camera else str(source), camera=camera,
                      model=str(weights), category=next(k for k,v in engine.CLASSES.items() if v == [0]),
                      axis=axis, position=position, confidence=0.25, imgsz=640, device='cpu')
        self.config = config
        with patch.dict(sys.modules, {'cv2': cv2, 'ultralytics': ultra}), \
                patch.object(engine, '__file__', str(root / 'engine.py')), \
                patch.object(engine, 'datetime', FixedDatetime), \
                patch.object(engine, '_runtime_metadata', return_value={'python_version': 'synthetic'}), \
                patch.object(engine, 'LineCounter', ObservedCounter):
            return engine.run(config, stop, publish)

    def rows(self, name):
        with (self.run_folder / name).open(encoding='utf-8-sig', newline='') as file:
            return list(csv.DictReader(file))

    def manifest(self):
        return json.loads((self.run_folder / 'run.json').read_text(encoding='utf-8'))

    def test_normal_run_preserves_events_schemas_update_count_and_order(self):
        result = self.execute([
            [detection(20, 7), detection(80, 9)],
            [detection(80, 7), detection(20, 9)], None,
        ])
        self.assertEqual(result['counts'], {('person','down'):1, ('person','up'):1})
        self.assertEqual(self.timeline, [
            ('update',1,7),('update',1,9),('prune',1),
            ('update',2,7),('update',2,9),('prune',2),('prune',3)])
        self.assertEqual(self.manifest()['status'], 'completed')
        self.assertEqual(self.manifest()['frames_written'], 3)
        self.capture.release.assert_called_once()
        events = self.rows('events.csv')
        self.assertEqual(list(events[0]), ['wall_time','source_seconds','frame','track_id','class','direction'])
        self.assertEqual([(r['frame'],r['track_id'],r['direction'],r['source_seconds']) for r in events],
                         [('2','7','down','0.04'),('2','9','up','0.04')])
        self.assertEqual(list(self.rows('summary.csv')[0]), ['class','direction','count'])
        self.assertEqual(list(self.rows('performance.csv')[0]),
                         ['model','imgsz','device','frames','elapsed_seconds','pipeline_fps','status'])
        self.assertEqual([(r['direction'],r['count']) for r in self.rows('summary.csv')], [('down','1'),('up','1')])
        for call in self.model.track.call_args_list:
            self.assertEqual(call.kwargs, dict(persist=True, tracker='bytetrack.yaml', classes=[0],
                                              conf=0.25, imgsz=640, device='cpu', verbose=False))
        self.ultra.YOLO.assert_called_once_with(self.config['model'])

    def test_snapshots_do_not_alias_mutated_counter_state(self):
        self.execute([[detection(y)] for y in [20,50,80,20,80]])
        rows = self.rows('tracks.csv')
        self.assertEqual([r['observed_side'] for r in rows], ['-1','0','1','-1','1'])
        self.assertEqual([r['state_side_after'] for r in rows], ['-1','-1','1','-1','1'])
        self.assertEqual([r['counted_before'] for r in rows], ['', '', '', 'down', 'down|up'])
        self.assertEqual([r['event_direction'] for r in rows], ['', '', 'down', 'up', ''])
        self.assertEqual(rows[2]['state_side_before'], '-1')
        self.assertEqual(rows[2]['state_seen_before'], '2')

    def test_empty_results_and_empty_box_arrays_are_logged(self):
        self.execute([None, [], None])
        self.assertEqual([r['detection_count'] for r in self.rows('frames.csv')], ['0']*3)
        self.assertEqual(self.rows('tracks.csv'), [])
        self.assertEqual(self.rows('events.csv'), [])
        self.assertEqual(self.timeline, [('prune',1),('prune',2),('prune',3)])

    def test_missing_ids_log_detections_without_counter_updates(self):
        self.execute([[detection(20,None), detection(80,None)]])
        rows = self.rows('tracks.csv')
        self.assertEqual(len(rows), 2)
        for row in rows:
            for key in ['track_id','state_side_before','state_seen_before','counted_before','state_side_after','event_direction']:
                self.assertEqual(row[key], '')
        self.assertEqual(self.rows('frames.csv')[0]['tracked_count'], '0')
        self.assertEqual(self.rows('frames.csv')[0]['detection_count'], '2')
        self.assertEqual(self.timeline, [('prune',1)])

    def test_resize_geometry_confidence_and_metadata_hashes(self):
        self.execute([[detection(340,7,x=630)]], width=1920, height=1080, position=.501)
        row = self.rows('frames.csv')[0]
        self.assertEqual((row['source_width'],row['source_height']), ('1920','1080'))
        self.assertEqual((row['processed_width'],row['processed_height']), ('1280','720'))
        self.assertAlmostEqual(float(row['line_coordinate']), 360.72)
        self.assertEqual(row['drawn_line_coordinate'], '360')
        track = self.rows('tracks.csv')[0]
        self.assertEqual((track['anchor_x'],track['anchor_y'],track['confidence']), ('630.0','340','0.875'))
        self.assertAlmostEqual(float(track['signed_distance']), -20.72)
        metadata = self.manifest()['metadata']
        self.assertEqual(metadata['source_name'], 'người.mp4')
        self.assertEqual(metadata['model_name'], 'dummy.pt')
        self.assertEqual(metadata['model_sha256'], hashlib.sha256(b'synthetic weights, never loaded').hexdigest())
        self.assertEqual(metadata['source_sha256'], hashlib.sha256(b'synthetic video data, not a real video').hexdigest())
        self.assertIsNotNone(metadata['tracker_sha256'])
        self.assertEqual(metadata['margin'], 8)
        self.assertEqual(metadata['max_age'], 90)
        self.assertNotIn(str(self.root), (self.run_folder/'run.json').read_text(encoding='utf-8'))

    def test_vertical_line_keeps_right_direction(self):
        result = self.execute([[detection(20,x=20)], [detection(20,x=80)]], axis='vertical')
        self.assertEqual(result['counts'], {('person','right'):1})
        self.assertEqual(self.rows('events.csv')[0]['direction'], 'right')
        self.assertEqual(self.cv2.line.call_args.args[1:3], ((50,0),(50,99)))

    def test_prune_still_expires_id_across_empty_frames(self):
        self.execute([[detection(20)]] + [None]*91 + [[detection(80)], [detection(20)]])
        rows = self.rows('tracks.csv')
        self.assertEqual(rows[1]['state_side_before'], '')
        self.assertEqual(rows[1]['event_direction'], '')
        self.assertEqual(rows[2]['event_direction'], 'up')
        self.assertEqual(len([item for item in self.timeline if item[0]=='prune']), 94)

    def test_manual_stop_logs_only_completed_frames(self):
        self.execute([None,None,None], stop_after=1)
        self.assertEqual(self.manifest()['status'], 'stopped')
        self.assertEqual(len(self.rows('frames.csv')), 1)
        self.assertEqual(self.rows('performance.csv')[0]['status'], 'stopped')
        self.assertEqual(self.model.track.call_count, 1)

    def test_preexisting_stop_has_no_fabricated_frame(self):
        self.execute([None], prestop=True)
        self.assertEqual(self.rows('frames.csv'), [])
        self.assertEqual(self.manifest()['status'], 'stopped')
        self.model.track.assert_not_called()

    def test_inference_failure_does_not_log_failed_frame_as_empty(self):
        failure = RuntimeError('fake inference failed')
        with self.assertRaises(RuntimeError) as caught:
            self.execute([None,failure])
        self.assertIs(caught.exception, failure)
        self.assertEqual([r['frame'] for r in self.rows('frames.csv')], ['1'])
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.rows('performance.csv')[0]['status'], 'error')
        self.capture.release.assert_called_once()

    def test_plot_failure_can_leave_partial_tracks_but_not_successful_frame(self):
        with self.assertRaisesRegex(RuntimeError, 'plot failed'):
            self.execute([[detection(20)], [detection(80)]], plot_error_on=2)
        self.assertEqual([r['frame'] for r in self.rows('frames.csv')], ['1'])
        self.assertEqual([r['frame'] for r in self.rows('tracks.csv')], ['1','2'])
        self.assertEqual(len(self.rows('events.csv')), 1)
        self.assertEqual(self.manifest()['status'], 'error')

    def test_frame_logging_failure_stops_engine(self):
        original = RunLogger.write_frame
        calls = 0

        def fail_second(logger, **fields):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise LoggingError('simulated frame log failure')
            return original(logger, **fields)

        with patch.object(RunLogger, 'write_frame', fail_second):
            with self.assertRaises(LoggingError):
                self.execute([None,None,None])
        self.assertEqual(len(self.rows('frames.csv')), 1)
        self.assertEqual(self.model.track.call_count, 2)
        self.assertEqual(self.manifest()['status'], 'error')

    def test_track_logging_failure_does_not_repeat_counter_update(self):
        original = RunLogger.write_track

        def fail_crossing(logger, **fields):
            if fields['event_direction']:
                raise LoggingError('simulated track log failure')
            return original(logger, **fields)

        with patch.object(RunLogger, 'write_track', fail_crossing):
            with self.assertRaises(LoggingError):
                self.execute([[detection(20)], [detection(80)]])
        self.assertEqual(self.timeline, [('update',1,1),('prune',1),('update',2,1)])
        self.assertEqual(len(self.rows('events.csv')), 1)
        self.assertEqual(len(self.rows('frames.csv')), 1)
        self.assertEqual(self.manifest()['status'], 'error')

    def test_model_startup_failure_is_recorded(self):
        with self.assertRaisesRegex(RuntimeError, 'model startup'):
            self.execute([None], model_error=RuntimeError('model startup'))
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.rows('frames.csv'), [])
        self.cv2.VideoCapture.assert_not_called()

    def test_capture_open_failure_is_recorded_and_released(self):
        with self.assertRaises(RuntimeError):
            self.execute([None], opened=False)
        self.assertEqual(self.manifest()['status'], 'error')
        self.capture.release.assert_called_once()
        self.model.track.assert_not_called()

    def test_empty_video_is_error_not_completed(self):
        with self.assertRaises(RuntimeError):
            self.execute([])
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.rows('frames.csv'), [])

    def test_webcam_read_failure_after_frame_remains_error(self):
        with self.assertRaises(RuntimeError):
            self.execute([None], camera=True)
        self.cv2.VideoCapture.assert_called_once_with(0)
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(len(self.rows('frames.csv')), 1)
        self.assertIsNone(self.manifest()['metadata']['source_name'])
        self.assertIsNone(self.manifest()['metadata']['source_sha256'])

    def test_invalid_fps_uses_existing_fallback(self):
        self.execute([[detection(20)], [detection(80)]], fps=float('nan'))
        metadata = self.manifest()['metadata']
        self.assertIsNone(metadata['source_fps'])
        self.assertEqual(metadata['effective_fps'], 30.0)
        self.assertTrue(metadata['fps_fallback_used'])
        self.assertEqual(self.rows('events.csv')[0]['source_seconds'], '0.033')

    def test_capture_cleanup_failure_prevents_completed(self):
        failure = OSError('release failed')
        with self.assertRaises(OSError) as caught:
            self.execute([None], release_error=failure)
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.rows('performance.csv')[0]['status'], 'error')

    def test_summary_failure_prevents_completed_but_performance_is_attempted(self):
        with patch.object(engine, '_write_summary', side_effect=OSError('summary failed')):
            with self.assertRaises(OSError):
                self.execute([None])
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.rows('performance.csv')[0]['status'], 'error')

    def test_performance_failure_prevents_manifest_completed(self):
        with patch.object(engine, '_write_performance', side_effect=OSError('performance failed')):
            with self.assertRaises(OSError):
                self.execute([None])
        self.assertEqual(self.manifest()['status'], 'error')

    def test_logger_finalization_failure_corrects_legacy_status(self):
        original = RunLogger._save_manifest

        def fail_completed(logger):
            if logger.manifest['status'] == 'completed':
                raise OSError('finalization failed')
            return original(logger)

        with patch.object(RunLogger, '_save_manifest', fail_completed):
            with self.assertRaises(LoggingError):
                self.execute([None])
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.rows('performance.csv')[0]['status'], 'error')

    def test_primary_error_survives_all_cleanup_failures(self):
        failure = RuntimeError('primary inference error')
        original_close = RunLogger.close

        def fail_close(logger, status='completed'):
            original_close(logger, status)
            raise OSError('secondary close error')

        with patch.object(engine, '_write_summary', side_effect=OSError('summary')), \
                patch.object(engine, '_write_performance', side_effect=OSError('performance')), \
                patch.object(RunLogger, 'close', fail_close):
            with self.assertRaises(RuntimeError) as caught:
                self.execute([failure], release_error=OSError('release'))
        self.assertIs(caught.exception, failure)
        self.assertGreaterEqual(len(caught.exception.__notes__), 4)
        self.assertEqual(self.manifest()['status'], 'error')
        self.capture.release.assert_called_once()

    def test_existing_run_directory_is_never_touched(self):
        with self.assertRaises(FileExistsError):
            self.execute([None], collision=True)
        self.assertEqual((self.run_folder/'events.csv').read_text(), 'existing data')
        self.assertEqual([p.name for p in self.run_folder.iterdir()], ['events.csv'])
        self.ultra.YOLO.assert_not_called()

    def test_preview_callback_failure_keeps_processed_frame_but_fails_run(self):
        failure = RuntimeError('preview delivery failed')
        with self.assertRaises(RuntimeError) as caught:
            self.execute([None], publish_error=failure)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(self.rows('frames.csv')), 1)
        self.assertEqual(self.manifest()['status'], 'error')

    def test_events_csv_close_failure_preserves_primary_or_becomes_error(self):
        actual_open = Path.open
        opened = []

        class FailingClose:
            def __init__(self, file):
                self.file = file

            def __getattr__(self, name):
                return getattr(self.file, name)

            def close(self):
                self.file.close()
                raise OSError('event close failed')

        def open_with_failure(path, *args, **kwargs):
            file = actual_open(path, *args, **kwargs)
            if path.name == 'events.csv' and args and args[0] == 'w':
                opened.append(file)
                return FailingClose(file)
            return file

        for failure in [None, RuntimeError('primary inference failure')]:
            with self.subTest(inference_error=failure is not None):
                with patch.object(Path, 'open', open_with_failure):
                    with self.assertRaises(RuntimeError if failure else OSError) as caught:
                        self.execute([failure])
                if failure is not None:
                    self.assertIs(caught.exception, failure)
                    self.assertIn('Additional event CSV close failure: OSError', caught.exception.__notes__)
                self.assertTrue(opened[-1].closed)
                self.assertEqual(self.manifest()['status'], 'error')

    def test_metadata_logging_failure_before_model_load_is_reported(self):
        original = RunLogger._save_manifest

        def fail_update(logger):
            if logger.manifest['status'] == 'running' and logger.manifest['metadata']['source_sha256']:
                raise OSError('update failed')
            return original(logger)

        with patch.object(RunLogger, '_save_manifest', fail_update):
            with self.assertRaises(LoggingError):
                self.execute([None])
        self.ultra.YOLO.assert_not_called()
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.rows('frames.csv'), [])


if __name__ == '__main__':
    unittest.main()
