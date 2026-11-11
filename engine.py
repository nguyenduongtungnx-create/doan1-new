import csv
import hashlib
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from counter import LineCounter
from run_logging import RunLogger

CLASSES = {'Người': [0], 'Phương tiện': [1, 2, 3, 5, 7], 'Tất cả': [0, 1, 2, 3, 5, 7]}


def _file_hash(path):
    """Optional provenance; unavailable files are null, never guessed."""
    if path is None:
        return None
    try:
        digest = hashlib.sha256()
        with Path(path).open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _runtime_metadata(root):
    """Collect versions/code identity only, not environment or account settings."""
    values = {'python_version': sys.version.split()[0]}
    for package, field in [('ultralytics', 'ultralytics_version'),
                           ('opencv-python', 'opencv_version'), ('torch', 'torch_version')]:
        try:
            values[field] = version(package)
        except PackageNotFoundError:
            values[field] = None
    try:
        values['git_commit'] = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
            text=True, timeout=2).strip()
        values['git_dirty'] = bool(subprocess.check_output(
            ['git', '-C', str(root), 'status', '--porcelain'], stderr=subprocess.DEVNULL,
            text=True, timeout=2).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return values


def _write_summary(folder, counts):
    with (folder / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['class', 'direction', 'count'])
        for (label, direction), count in sorted(counts.items()):
            writer.writerow([label, direction, count])


@contextmanager
def _events_file(path):
    """A secondary close failure must not replace the original processing error."""
    file = path.open('w', newline='', encoding='utf-8-sig')
    try:
        yield file
    except BaseException as exc:
        try:
            file.close()
        except BaseException as secondary:
            exc.add_note(f'Additional event CSV close failure: {type(secondary).__name__}')
        raise
    else:
        file.close()


def _write_performance(folder, config, processed, started, status):
    with (folder / 'performance.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'imgsz', 'device', 'frames', 'elapsed_seconds', 'pipeline_fps', 'status'])
        elapsed = time.perf_counter()-started
        writer.writerow([config['model'], config['imgsz'], config['device'], processed, round(elapsed,3), round(processed/max(elapsed,.001),2), status])


def run(config, stop, publish):
    """Log successful frames before preview delivery; partial failed frames have no ok row.

    Track/event rows can exist for a partial frame; only a completed run promises
    cross-file consistency. Webcam frame time is sampled after read, while event
    time retains its original sampling at crossing. FPS includes logging overhead;
    processing_ms stops before the frame CSV write and preview callback.
    """
    import cv2
    import ultralytics
    from ultralytics import YOLO
    root = Path(__file__).parent
    parent = root / 'outputs'
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    counter = LineCounter(config['axis'], config['position'])
    cap, processed, status = None, 0, 'completed'
    logger, failure = None, None
    started = time.perf_counter()

    def remember_failure(exc):
        nonlocal failure, status
        status = 'error'
        if failure is None:
            failure = (exc, exc.__traceback__)
        else:
            failure[0].add_note(f'Additional cleanup failure: {type(exc).__name__}')

    try:
        logger = RunLogger(folder, dict(
            run_id=folder.name, source_type='webcam' if config['camera'] else 'video',
            model_name=Path(config['model']).name, confidence=config['confidence'],
            imgsz=config['imgsz'], device=config['device'], class_ids=CLASSES[config['category']],
            axis=counter.axis, position=counter.position, margin=counter.margin,
            max_age=counter.max_age, tracker_name='bytetrack.yaml',
            source_name=None if config['camera'] else Path(config['source']).name,
        ))
        publish({'status': 'Đang tải mô hình (lần đầu cần Internet)…', 'folder': str(folder)})
        tracker_root = getattr(ultralytics, '__file__', None)
        tracker_path = Path(tracker_root).parent / 'cfg/trackers/bytetrack.yaml' if tracker_root else None
        logger.update_metadata(
            **_runtime_metadata(root), tracker_sha256=_file_hash(tracker_path),
            source_sha256=None if config['camera'] else _file_hash(config['source']))
        model = YOLO(config['model'])
        logger.update_metadata(model_sha256=_file_hash(getattr(model, 'ckpt_path', None) or config['model']))
        cap = cv2.VideoCapture(int(config['source']) if config['camera'] else config['source'])
        if not cap.isOpened():
            raise RuntimeError('Không mở được video/webcam. Kiểm tra nguồn và quyền camera.')
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps
        if not 0 < fps < 1000:
            fps = 30.0
        source_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        source_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        reported_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        logger.update_metadata(
            source_fps=raw_fps if math.isfinite(raw_fps) else None,
            effective_fps=fps, fps_fallback_used=not 0 < raw_fps < 1000,
            source_width=int(source_width) if math.isfinite(source_width) and source_width > 0 else None,
            source_height=int(source_height) if math.isfinite(source_height) and source_height > 0 else None,
            reported_frames=int(reported_frames) if math.isfinite(reported_frames) and reported_frames >= 0 else None,
        )
        with _events_file(folder / 'events.csv') as file:
            writer = csv.writer(file)
            writer.writerow(['wall_time', 'source_seconds', 'frame', 'track_id', 'class', 'direction'])
            file.flush()
            while not stop.is_set():
                tick = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if config['camera'] or processed == 0:
                        raise RuntimeError('Không đọc được khung hình từ nguồn.')
                    break
                processed += 1
                source_h, source_w = frame.shape[:2]
                frame_seconds = time.perf_counter()-started if config['camera'] else (processed-1)/fps
                scale = min(1.0, 1280 / frame.shape[1])
                if scale < 1:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                h, w = frame.shape[:2]
                result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=CLASSES[config['category']], conf=config['confidence'], imgsz=config['imgsz'], device=config['device'], verbose=False)[0]
                boxes = result.boxes
                detection_count = tracked_count = 0
                line_coordinate = counter.position * (h if counter.axis == 'horizontal' else w)
                if boxes is not None:
                    coordinates = boxes.xyxy.cpu().tolist()
                    identities = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
                    classes = boxes.cls.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    if not len(coordinates) == len(identities) == len(classes) == len(confidences):
                        raise ValueError('Inconsistent detection arrays')
                    detection_count = len(coordinates)
                    for detection_index, (box, identity, cls, confidence) in enumerate(zip(coordinates, identities, classes, confidences)):
                        x1, y1, x2, y2 = box
                        label = model.names[cls]
                        point = ((x1+x2)/2, y2)
                        before_side = before_seen = counted_before = after_side = direction = None
                        if identity is not None:
                            state = counter.tracks.get(identity)
                            if state is not None:
                                before_side, before_seen = state['side'], state['seen']
                                counted_before = '|'.join(sorted(state['counted']))
                            direction = counter.update(identity, label, point, processed, w, h)
                            after_side = counter.tracks[identity]['side']
                            tracked_count += 1
                        if direction:
                            seconds = time.perf_counter()-started if config['camera'] else (processed-1)/fps
                            writer.writerow([datetime.now().isoformat(timespec='seconds'), round(seconds, 3), processed, identity, label, direction])
                            file.flush()
                        distance = (point[1] if counter.axis == 'horizontal' else point[0]) - line_coordinate
                        logger.write_track(
                            frame=processed, source_seconds=frame_seconds, detection_index=detection_index,
                            track_id=identity, class_id=cls, class_name=label, confidence=confidence,
                            x1=x1, y1=y1, x2=x2, y2=y2, anchor_x=point[0], anchor_y=point[1],
                            signed_distance=distance,
                            observed_side=1 if distance > counter.margin else -1 if distance < -counter.margin else 0,
                            state_side_before=before_side, state_seen_before=before_seen,
                            counted_before=counted_before, state_side_after=after_side, event_direction=direction,
                        )
                counter.prune(processed)
                annotated = result.plot()
                if config['axis'] == 'horizontal':
                    pos = int(h * config['position'])
                    a, b = (0, pos), (w-1, pos)
                else:
                    pos = int(w * config['position'])
                    a, b = (pos, 0), (pos, h-1)
                cv2.line(annotated, a, b, (0, 240, 255), 3)
                rate = 1 / max(time.perf_counter()-tick, 1e-6)
                cv2.putText(annotated, f'Count: {sum(counter.counts.values())} | FPS: {rate:.1f}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                logger.write_frame(
                    frame=processed, source_seconds=frame_seconds, wall_time=datetime.now().astimezone().isoformat(),
                    source_width=source_w, source_height=source_h, processed_width=w, processed_height=h,
                    line_coordinate=line_coordinate, drawn_line_coordinate=pos,
                    detection_count=detection_count, tracked_count=tracked_count,
                    processing_ms=(time.perf_counter()-tick)*1000, status='ok',
                )
                publish({'frame': preview, 'counts': dict(counter.counts), 'status': f'Frame {processed} • {rate:.1f} FPS', 'folder': str(folder)})
            if stop.is_set():
                status = 'stopped'
    except BaseException as exc:
        remember_failure(exc)
    finally:
        if cap is not None:
            try:
                cap.release()
            except BaseException as exc:
                remember_failure(exc)
        # Only touch a directory successfully created/owned by this logger.
        if logger is not None:
            try:
                _write_summary(folder, counter.counts)
            except BaseException as exc:
                remember_failure(exc)
            try:
                _write_performance(folder, config, processed, started, status)
            except BaseException as exc:
                remember_failure(exc)
            try:
                logger.close(status)
            except BaseException as exc:
                remember_failure(exc)
                # A close failure happened after the legacy status was written.
                try:
                    _write_performance(folder, config, processed, started, status)
                except BaseException as secondary:
                    remember_failure(secondary)
    if failure is not None:
        raise failure[0].with_traceback(failure[1])
    return {'counts': dict(counter.counts), 'folder': str(folder)}
