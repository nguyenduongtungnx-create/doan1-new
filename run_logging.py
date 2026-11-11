"""Standalone run logger, schema v1. Uses only the Python standard library.

Use ``with RunLogger(new_directory, metadata) as log: ...``. The directory must
not exist. This module never collects metadata, detects objects, or counts them.
The caller supplies observations and is responsible for their accuracy/linkage.

run.json: schema_version, metadata, status, started_at, finished_at, error_code,
frames_written, tracks_written. Times are timezone-aware UTC. Required metadata
is listed below; absent optional metadata is JSON null. Unknown keys are rejected
to avoid accidentally serializing credentials, environment dictionaries or paths.
model_name/source_name are filenames only. No raw exception messages are saved.

CSVs have fixed headers and UTF-8 BOM. Required columns are listed below; omitted
or None optional values become blank cells. Frame numbers are one-based, detection
indices zero-based, bbox/anchor coordinates refer to the processed image. Sides
are -1/0/1; counted_before is a canonical string such as 'down|up'. source_seconds
and all geometry/timing are supplied by the caller, not calculated here.

Each row is flushed immediately (not fsync). Counts denote flushed rows, not
necessarily successfully processed frames. A write/flush/close failure poisons
the run and closes both CSVs. run.json is replaced atomically after closing CSVs;
if storage remains unavailable it may retain 'running', never a false 'completed'.
An incomplete .tmp file is not authoritative. No crash/power-loss durability is
promised. This logger is single-threaded. Metadata updates accept optional fields
only; the run's required configuration is immutable through the public API.
"""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED_METADATA = (
    'run_id', 'source_type', 'model_name', 'confidence', 'imgsz', 'device',
    'class_ids', 'axis', 'position', 'margin', 'max_age',
)
OPTIONAL_METADATA = (
    'source_name', 'source_sha256', 'model_sha256', 'tracker_name',
    'tracker_sha256', 'source_width', 'source_height', 'source_fps',
    'effective_fps', 'fps_fallback_used', 'reported_frames', 'python_version',
    'ultralytics_version', 'opencv_version', 'torch_version', 'git_commit',
    'git_dirty',
)
FRAME_FIELDS = (
    'frame', 'source_seconds', 'wall_time', 'source_width', 'source_height',
    'processed_width', 'processed_height', 'line_coordinate',
    'drawn_line_coordinate', 'detection_count', 'tracked_count',
    'processing_ms', 'status',
)
REQUIRED_FRAME_FIELDS = (
    'frame', 'source_seconds', 'processed_width', 'processed_height',
    'detection_count', 'tracked_count', 'status',
)
TRACK_FIELDS = (
    'frame', 'source_seconds', 'detection_index', 'track_id', 'class_id',
    'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'anchor_x', 'anchor_y',
    'signed_distance', 'observed_side', 'state_side_before', 'state_seen_before',
    'counted_before', 'state_side_after', 'event_direction',
)
REQUIRED_TRACK_FIELDS = (
    'frame', 'source_seconds', 'detection_index', 'class_id', 'class_name',
    'confidence', 'x1', 'y1', 'x2', 'y2', 'anchor_x', 'anchor_y',
)


class LoggingError(RuntimeError):
    """A logging I/O failure; error messages deliberately omit supplied data."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _validate_fields(data, required, allowed):
    if set(data) - set(allowed):
        raise ValueError('Unknown logging fields')
    if any(key not in data or data[key] is None or data[key] == '' for key in required):
        raise ValueError('Missing required logging fields')
    for value in data.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite logging value')


def _metadata_snapshot(metadata):
    _validate_fields(metadata, REQUIRED_METADATA, REQUIRED_METADATA + OPTIONAL_METADATA)
    if not isinstance(metadata['class_ids'], list) or any(
            type(value) is not int for value in metadata['class_ids']):
        raise ValueError('class_ids must be a list of integers')
    if any(value is not None and not isinstance(value, (str, int, float, bool))
           for key, value in metadata.items() if key != 'class_ids'):
        raise ValueError('Metadata fields must be primitive values')
    if metadata['source_type'] not in ('video', 'webcam'):
        raise ValueError('Invalid source type')
    if metadata['axis'] not in ('horizontal', 'vertical'):
        raise ValueError('Invalid line axis')
    for key in ('model_name', 'source_name', 'tracker_name'):
        value = metadata.get(key)
        if value is not None and (not isinstance(value, str) or any(c in value for c in '/\\:')):
            raise ValueError('Logging names must be filenames without paths')
    return json.loads(json.dumps(metadata, allow_nan=False))


class RunLogger:
    """Write one new run. close('stopped') supports an explicit user stop.

    Context exit completes normally or records 'run_failed' on a caller exception,
    preserving that exception. Invalid records also poison the run. Required-field
    checks and primitive CSV types are enforced; full semantic validation of
    caller-provided geometry, IDs and cross-file consistency is out of scope.
    """

    def __init__(self, directory, metadata):
        # Validate and snapshot JSON without retaining a mutable caller dictionary.
        snapshot = _metadata_snapshot(metadata)
        self.directory = Path(directory)
        try:
            self.directory.mkdir(exist_ok=False)
        except FileExistsError:
            raise
        except OSError as exc:
            raise LoggingError('Could not create run log directory') from exc
        self.closed = False
        self._files = {}
        self._writers = {}
        self.manifest = dict(
            schema_version=SCHEMA_VERSION,
            metadata={key: snapshot.get(key) for key in REQUIRED_METADATA + OPTIONAL_METADATA},
            status='running', started_at=_now(), finished_at=None, error_code=None,
            frames_written=0, tracks_written=0,
        )
        try:
            self._save_manifest()
            for name, fields in (('frames', FRAME_FIELDS), ('tracks', TRACK_FIELDS)):
                file = (self.directory / f'{name}.csv').open('x', encoding='utf-8-sig', newline='')
                self._files[name] = file
                self._writers[name] = csv.DictWriter(file, fieldnames=fields)
                self._writers[name].writeheader()
                file.flush()
        except (OSError, csv.Error, UnicodeError) as exc:
            self._mark_error('initialization_failed')
            try:
                self.close('error')
            except LoggingError:
                pass
            raise LoggingError('Run logger initialization failed') from exc

    def _save_manifest(self):
        temporary = self.directory / 'run.json.tmp'
        with temporary.open('w', encoding='utf-8', newline='\n') as file:
            json.dump(self.manifest, file, ensure_ascii=False, allow_nan=False, indent=2)
            file.write('\n')
        temporary.replace(self.directory / 'run.json')

    def _mark_error(self, code):
        self.manifest['status'] = 'error'
        if self.manifest['error_code'] is None:
            self.manifest['error_code'] = code

    def write_frame(self, **fields):
        self._write('frames', fields, REQUIRED_FRAME_FIELDS, FRAME_FIELDS)

    def write_track(self, **fields):
        self._write('tracks', fields, REQUIRED_TRACK_FIELDS, TRACK_FIELDS)

    def update_metadata(self, **fields):
        """Atomically persist validated optional metadata; never change configuration.

        Invalid input raises ValueError without changing metadata. I/O failure
        closes/poisons the run and raises LoggingError, like CSV writes.
        """
        if self.closed:
            raise LoggingError('Run logger is closed')
        _validate_fields(fields, (), OPTIONAL_METADATA)
        updated = _metadata_snapshot({**self.manifest['metadata'], **fields})
        self.manifest['metadata'] = updated
        try:
            self._save_manifest()
        except (OSError, UnicodeError) as exc:
            self._mark_error('metadata_write_failed')
            try:
                self.close('error')
            except LoggingError:
                pass
            raise LoggingError('Run metadata update failed') from exc

    def _write(self, name, data, required, allowed):
        if self.closed:
            raise LoggingError('Run logger is closed')
        try:
            _validate_fields(data, required, allowed)
            if any(value is not None and not isinstance(value, (str, int, float, bool))
                   for value in data.values()):
                raise ValueError('CSV logging fields must be primitive values')
        except ValueError:
            self._mark_error('invalid_record')
            self.close('error')
            raise
        try:
            self._writers[name].writerow(data)
            self._files[name].flush()
            self.manifest[f'{name}_written'] += 1
        except (OSError, csv.Error, UnicodeError) as exc:
            self._mark_error('csv_write_failed')
            try:
                self.close('error')
            except LoggingError:
                pass
            raise LoggingError('CSV logging write failed') from exc

    def close(self, status='completed'):
        """Close all streams before publishing the terminal manifest; idempotent."""
        if status not in ('completed', 'stopped', 'error'):
            raise ValueError('Invalid terminal run status')
        if self.closed:
            return
        if status == 'error':
            self._mark_error('run_failed')
        close_error = None
        for file in self._files.values():
            try:
                file.close()
            except (OSError, UnicodeError) as exc:
                close_error = exc
                self._mark_error('csv_close_failed')
        self.closed = True
        if self.manifest['status'] != 'error':
            self.manifest['status'] = status
        self.manifest['finished_at'] = _now()
        try:
            self._save_manifest()
        except (OSError, UnicodeError) as exc:
            self._mark_error('metadata_write_failed')
            try:
                self._save_manifest()
            except (OSError, UnicodeError):
                pass
            raise LoggingError('Run manifest finalization failed') from exc
        if close_error is not None:
            raise LoggingError('CSV logging close failed') from close_error

    def __enter__(self):
        if self.closed:
            raise LoggingError('Run logger is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close('error' if exc_type is not None else 'completed')
        except LoggingError:
            if exc_type is None:
                raise
        return False
