"""Synthetic tests only; artifacts live in OS temporary directories, not outputs/."""
import codecs
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_logging import (
    FRAME_FIELDS, TRACK_FIELDS, LoggingError, RunLogger,
)


def metadata(**overrides):
    return dict(run_id='synthetic-001', source_type='video', model_name='dummy.pt',
                confidence=0.25, imgsz=640, device='cpu', class_ids=[0],
                axis='horizontal', position=0.5, margin=8, max_age=90,
                **overrides)


def frame(**overrides):
    row = dict(frame=1, source_seconds=0.0, processed_width=1280,
               processed_height=720, detection_count=0, tracked_count=0,
               status='ok')
    row.update(overrides)
    return row


def track(**overrides):
    row = dict(frame=1, source_seconds=0.0, detection_index=0, track_id=7,
               class_id=0, class_name='person', confidence=0.9,
               x1=10.5, y1=200.0, x2=50.5, y2=340.0,
               anchor_x=30.5, anchor_y=340.0)
    row.update(overrides)
    return row


class RunLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='doan1-logger-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / 'new-run'

    def manifest(self):
        return json.loads((self.directory / 'run.json').read_text(encoding='utf-8'))

    def rows(self, name):
        with (self.directory / name).open(encoding='utf-8-sig', newline='') as file:
            return list(csv.DictReader(file))

    def assert_closed(self, logger):
        self.assertTrue(logger.closed)
        self.assertTrue(all(file.closed for file in logger._files.values()))

    def test_fixed_csv_headers_and_bom(self):
        # Literal expectations catch schema changes, including column order.
        expected_frames = (
            'frame,source_seconds,wall_time,source_width,source_height,'
            'processed_width,processed_height,line_coordinate,'
            'drawn_line_coordinate,detection_count,tracked_count,processing_ms,status'
        )
        expected_tracks = (
            'frame,source_seconds,detection_index,track_id,class_id,class_name,'
            'confidence,x1,y1,x2,y2,anchor_x,anchor_y,signed_distance,observed_side,'
            'state_side_before,state_seen_before,counted_before,state_side_after,event_direction'
        )
        with RunLogger(self.directory, metadata()):
            pass
        for name, expected in [('frames.csv', expected_frames), ('tracks.csv', expected_tracks)]:
            data = (self.directory / name).read_bytes()
            self.assertTrue(data.startswith(codecs.BOM_UTF8))
            self.assertEqual(data.decode('utf-8-sig').strip(), expected)
            self.assertEqual(data.count(codecs.BOM_UTF8), 1)
        self.assertEqual(','.join(FRAME_FIELDS), expected_frames)
        self.assertEqual(','.join(TRACK_FIELDS), expected_tracks)

    def test_empty_frame_has_row_and_no_tracks(self):
        with RunLogger(self.directory, metadata()) as logger:
            logger.write_frame(**frame())
        rows = self.rows('frames.csv')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['detection_count'], '0')
        self.assertEqual(rows[0]['tracked_count'], '0')
        self.assertEqual(self.rows('tracks.csv'), [])
        self.assertEqual(self.manifest()['frames_written'], 1)
        self.assertEqual(self.manifest()['tracks_written'], 0)

    def test_multiple_tracks_preserve_ids_geometry_and_direction(self):
        with RunLogger(self.directory, metadata()) as logger:
            logger.write_track(**track(event_direction='down', state_side_before=-1,
                                       state_side_after=1, counted_before=''))
            logger.write_track(**track(detection_index=1, track_id=9, anchor_x=80.25))
            logger.write_frame(**frame(detection_count=2, tracked_count=2))
        rows = self.rows('tracks.csv')
        self.assertEqual([row['track_id'] for row in rows], ['7', '9'])
        self.assertEqual([row['detection_index'] for row in rows], ['0', '1'])
        self.assertEqual(rows[0]['x1'], '10.5')
        self.assertEqual(rows[0]['event_direction'], 'down')
        self.assertEqual(rows[1]['anchor_x'], '80.25')
        self.assertEqual(self.manifest()['tracks_written'], 2)

    def test_missing_track_id_and_optional_fields_are_blank(self):
        for omission in ('omitted', 'none'):
            with self.subTest(omission=omission):
                directory = self.directory.parent / omission
                row = track(track_id=None)
                if omission == 'omitted':
                    row.pop('track_id')
                with RunLogger(directory, metadata()) as logger:
                    logger.write_track(**row)
                    logger.write_frame(**frame(detection_count=1, tracked_count=0))
                with (directory / 'tracks.csv').open(encoding='utf-8-sig', newline='') as file:
                    saved = next(csv.DictReader(file))
                for key in ['track_id', 'event_direction', 'state_side_before', 'counted_before']:
                    self.assertEqual(saved[key], '')

    def test_unicode_round_trips_without_ascii_escaping(self):
        name = 'người_đi_bộ.mp4'
        with RunLogger(self.directory, metadata(source_name=name)) as logger:
            logger.write_track(**track(class_name='người, "đi bộ"'))
        self.assertEqual(self.manifest()['metadata']['source_name'], name)
        self.assertIn(name, (self.directory / 'run.json').read_text(encoding='utf-8'))
        self.assertEqual(self.rows('tracks.csv')[0]['class_name'], 'người, "đi bộ"')

    def test_manifest_lifecycle_schema_nulls_and_normal_close(self):
        with RunLogger(self.directory, metadata()) as logger:
            active = self.manifest()
            self.assertEqual(active['schema_version'], 1)
            self.assertEqual(active['status'], 'running')
            self.assertIsNone(active['finished_at'])
            self.assertIsNone(active['metadata']['source_sha256'])
            self.assertTrue(active['started_at'].endswith('+00:00'))
            logger.write_frame(**frame())
        saved = self.manifest()
        self.assertEqual(saved['status'], 'completed')
        self.assertIsNone(saved['error_code'])
        self.assertIsNotNone(saved['finished_at'])
        self.assert_closed(logger)
        self.assertFalse((self.directory / 'run.json.tmp').exists())

    def test_explicit_stop_is_not_completed(self):
        with RunLogger(self.directory, metadata()) as logger:
            logger.close('stopped')
        self.assertEqual(self.manifest()['status'], 'stopped')
        self.assert_closed(logger)

    def test_caller_exception_is_preserved_and_details_not_logged(self):
        failure = RuntimeError('private caller details must not be logged')
        with self.assertRaises(RuntimeError) as caught:
            with RunLogger(self.directory, metadata()) as logger:
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.manifest()['error_code'], 'run_failed')
        self.assertNotIn(str(failure), (self.directory / 'run.json').read_text())
        self.assert_closed(logger)

    def test_csv_write_failure_closes_files_and_poison_is_sticky(self):
        for name in ['frames', 'tracks']:
            with self.subTest(csv=name):
                self.directory = Path(self.temporary.name) / name
                logger = RunLogger(self.directory, metadata())
                with patch.object(logger._writers[name], 'writerow', side_effect=OSError('simulated')):
                    with self.assertRaises(LoggingError):
                        if name == 'frames':
                            logger.write_frame(**frame())
                        else:
                            logger.write_track(**track())
                logger.close('completed')
                self.assertEqual(self.manifest()['status'], 'error')
                self.assertEqual(self.manifest()['error_code'], 'csv_write_failed')
                self.assertEqual(self.manifest()[f'{name}_written'], 0)
                self.assert_closed(logger)

    def test_csv_flush_failure_is_not_success(self):
        logger = RunLogger(self.directory, metadata())
        with patch.object(logger._files['frames'], 'flush', side_effect=OSError('simulated')):
            with self.assertRaises(LoggingError):
                logger.write_frame(**frame())
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.manifest()['frames_written'], 0)
        self.assert_closed(logger)

    def test_close_failure_still_attempts_other_file_and_reports_error(self):
        logger = RunLogger(self.directory, metadata())
        file = logger._files['frames']
        original_close = file.close

        def fail_close():
            original_close()
            raise OSError('simulated close failure')

        with patch.object(file, 'close', side_effect=fail_close):
            with self.assertRaises(LoggingError):
                logger.close()
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.manifest()['error_code'], 'csv_close_failed')
        self.assert_closed(logger)

    def test_initialization_failure_closes_first_csv(self):
        actual_open = Path.open
        opened = []

        def fail_second(path, *args, **kwargs):
            if path.name == 'tracks.csv':
                raise OSError('simulated open failure')
            file = actual_open(path, *args, **kwargs)
            if path.name == 'frames.csv':
                opened.append(file)
            return file

        with patch.object(Path, 'open', fail_second):
            with self.assertRaises(LoggingError):
                RunLogger(self.directory, metadata())
        self.assertEqual(self.manifest()['status'], 'error')
        self.assertEqual(self.manifest()['error_code'], 'initialization_failed')
        self.assertTrue(opened[0].closed)

    def test_manifest_replace_failure_never_publishes_completed(self):
        for persistent in [False, True]:
            with self.subTest(persistent=persistent):
                self.directory = Path(self.temporary.name) / str(persistent)
                logger = RunLogger(self.directory, metadata())
                actual_replace = Path.replace
                calls = 0

                def fail_replace(path, target):
                    nonlocal calls
                    calls += 1
                    if persistent or calls == 1:
                        raise OSError('simulated replace failure')
                    return actual_replace(path, target)

                with patch.object(Path, 'replace', fail_replace):
                    with self.assertRaises(LoggingError):
                        logger.close()
                self.assertEqual(logger.manifest['status'], 'error')
                self.assertEqual(self.manifest()['status'], 'running' if persistent else 'error')
                self.assert_closed(logger)

    def test_original_exception_survives_manifest_failure(self):
        logger = RunLogger(self.directory, metadata())
        failure = RuntimeError('caller failed')
        with patch.object(Path, 'replace', side_effect=OSError('storage failed')):
            with self.assertRaises(RuntimeError) as caught:
                with logger:
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertNotEqual(self.manifest()['status'], 'completed')
        self.assert_closed(logger)

    def test_unknown_metadata_and_paths_are_rejected_before_creation(self):
        for extra in [dict(credentials='not-a-real-credential'), dict(environment={}),
                      dict(source_name=r'C:\Users\private\video.mp4')]:
            with self.subTest(field=next(iter(extra))):
                with self.assertRaises(ValueError):
                    RunLogger(self.directory, metadata(**extra))
                self.assertFalse(self.directory.exists())

    def test_required_metadata_and_nested_payloads_rejected(self):
        for remove in ['run_id', 'model_name', 'axis']:
            values = metadata()
            values.pop(remove)
            with self.subTest(missing=remove), self.assertRaises(ValueError):
                RunLogger(self.directory, values)
        values = metadata()
        values['device'] = {'credentials': 'not-a-real-credential'}
        with self.assertRaises(ValueError):
            RunLogger(self.directory, values)
        self.assertFalse(self.directory.exists())

    def test_invalid_record_cannot_be_completed(self):
        for invalid in [dict(unexpected='value'), dict(processing_ms=float('nan'))]:
            self.directory = Path(self.temporary.name) / str(len(list(Path(self.temporary.name).iterdir())))
            logger = RunLogger(self.directory, metadata())
            with self.assertRaises(ValueError):
                logger.write_frame(**frame(**invalid))
            logger.close()
            self.assertEqual(self.manifest()['status'], 'error')
            self.assertEqual(self.manifest()['error_code'], 'invalid_record')
            self.assert_closed(logger)

    def test_existing_directory_is_not_overwritten(self):
        self.directory.mkdir()
        sentinel = self.directory / 'run.json'
        sentinel.write_text('existing evaluation', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            RunLogger(self.directory, metadata())
        self.assertEqual(sentinel.read_text(), 'existing evaluation')
        self.assertEqual(list(self.directory.iterdir()), [sentinel])

    def test_directory_creation_failure_is_explicit(self):
        with patch.object(Path, 'mkdir', side_effect=PermissionError('simulated')):
            with self.assertRaises(LoggingError):
                RunLogger(self.directory, metadata())
        self.assertFalse(self.directory.exists())

    def test_initial_manifest_failure_never_completes(self):
        with patch.object(Path, 'replace', side_effect=OSError('simulated')):
            with self.assertRaises(LoggingError):
                RunLogger(self.directory, metadata())
        self.assertFalse((self.directory / 'run.json').exists())
        self.assertFalse((self.directory / 'frames.csv').exists())
        self.assertFalse((self.directory / 'tracks.csv').exists())

    def test_closed_logger_rejects_more_rows(self):
        logger = RunLogger(self.directory, metadata())
        logger.close()
        before = (self.directory / 'frames.csv').read_bytes()
        with self.assertRaises(LoggingError):
            logger.write_frame(**frame())
        self.assertEqual((self.directory / 'frames.csv').read_bytes(), before)
        self.assertEqual(self.manifest()['status'], 'completed')

    def test_optional_metadata_updates_are_persisted(self):
        with RunLogger(self.directory, metadata()) as logger:
            logger.update_metadata(source_name='người.mp4', effective_fps=25.0,
                                   fps_fallback_used=False)
            saved = self.manifest()
            self.assertEqual(saved['status'], 'running')
            self.assertEqual(saved['metadata']['source_name'], 'người.mp4')
            self.assertEqual(saved['metadata']['effective_fps'], 25.0)
            self.assertFalse(saved['metadata']['fps_fallback_used'])
            self.assertEqual(saved['metadata']['model_name'], 'dummy.pt')
            logger.update_metadata(source_fps=None)
        self.assertIsNone(self.manifest()['metadata']['source_fps'])

    def test_invalid_metadata_updates_leave_snapshot_unchanged(self):
        with RunLogger(self.directory, metadata()) as logger:
            before = (self.directory / 'run.json').read_bytes()
            for values in [dict(position=0.7), dict(credentials='not-a-secret'),
                           dict(source_name=r'C:\private\file.mp4'),
                           dict(effective_fps=float('nan')), dict(git_commit={})]:
                with self.subTest(fields=list(values)), self.assertRaises(ValueError):
                    logger.update_metadata(**values)
                self.assertEqual((self.directory / 'run.json').read_bytes(), before)
                self.assertEqual(logger.manifest['metadata'], self.manifest()['metadata'])

    def test_metadata_update_failure_closes_and_poisons_run(self):
        for persistent in [False, True]:
            with self.subTest(persistent=persistent):
                self.directory = Path(self.temporary.name) / str(persistent)
                logger = RunLogger(self.directory, metadata())
                actual_replace = Path.replace
                calls = 0

                def fail_replace(path, target):
                    nonlocal calls
                    calls += 1
                    if persistent or calls == 1:
                        raise OSError('simulated update failure')
                    return actual_replace(path, target)

                with patch.object(Path, 'replace', fail_replace):
                    with self.assertRaises(LoggingError):
                        logger.update_metadata(effective_fps=25.0)
                self.assertEqual(logger.manifest['status'], 'error')
                self.assertEqual(logger.manifest['error_code'], 'metadata_write_failed')
                self.assertEqual(self.manifest()['status'], 'running' if persistent else 'error')
                self.assert_closed(logger)

    def test_metadata_update_after_close_is_rejected(self):
        with RunLogger(self.directory, metadata()) as logger:
            pass
        before = (self.directory / 'run.json').read_bytes()
        with self.assertRaises(LoggingError):
            logger.update_metadata(source_name='different.mp4')
        self.assertEqual((self.directory / 'run.json').read_bytes(), before)

    def test_filename_with_at_sign_is_not_mistaken_for_credentials(self):
        with RunLogger(self.directory, metadata(source_name='people@crossing.mp4')) as logger:
            logger.update_metadata(source_name='người@đường.mp4')
        self.assertEqual(self.manifest()['metadata']['source_name'], 'người@đường.mp4')


if __name__ == '__main__':
    unittest.main()
