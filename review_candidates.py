"""Suggest manual-review intervals from completed schema-v1 video logs only.

No inference, video decoding, annotations, error verdicts or accuracy metrics.
Thresholds are initial heuristics in processed-image pixels: near line <=24 px,
confidence <0.35, anchor displacement >80 px/frame between consecutive frames,
same ID returning after >=1 missing frame, >=2 nonzero observed-side changes
within 25 frames, and near-line tracks with no events anywhere in the run.
Zero side (inside dead band) does not count as a side change. Motion and side
history reset after max_age; an ID is never equated to a physical person.
Missing-ID detections still contribute near-line/low-confidence signals.

Every signal is retained as raw evidence with its original inclusive interval,
reason and track ID (null when unavailable). Review windows use a fixed 25-frame
grid anchored at frame 1; empty windows are omitted. Each evidence item links to
EVERY window intersecting its raw interval. Windows never merge. Playback context
defaults to +/-12 frames, clipped to processed frames, and does not affect links,
window boundaries or ranking. Playback ranges may overlap and exceed 25 frames.
Priority is the maximum reason weight (3: gaps/jumps/repeated sides, 2: low
confidence/no counted event, 1: near line), then number of distinct reasons,
then start frame. This ranking is not a probability or severity of an error.
Timestamps use frames.csv source_seconds (one-based frame indexing), not wall
time. Complete, fixed-FPS video logs are required; partial/webcam logs rejected.

CLI: python review_candidates.py RUN_DIRECTORY [--top-n N] [--output NEW_JSON]
top-N only selects window IDs; the complete evidence and ranked windows remain
in the output. Evidence IDs follow heuristic emission order; window IDs identify
grid positions independently of ranking. Repeated detections do not boost rank.
Output is advisory, not GT or documented review coverage. It may span most/all
of a busy video and cannot find people absent from detections. Always review
the whole source once to search independently for missed events. Without
--output prints JSON; with it creates a new file exclusively, never overwrites.
"""
import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path


NEAR_LINE_PIXELS = 24
LOW_CONFIDENCE = 0.35
JUMP_PIXELS_PER_FRAME = 80
SIDE_WINDOW_FRAMES = 25
REVIEW_WINDOW_FRAMES = 25
WEIGHTS = dict(near_line=1, low_confidence=2, track_without_counted_event=2,
               tracking_id_gap=3, abrupt_position_change=3, repeated_side_changes=3)
REQUIRED_FIELDS = {
    'frames': {'frame', 'source_seconds', 'detection_count', 'tracked_count', 'status'},
    'tracks': {'frame', 'source_seconds', 'detection_index', 'track_id', 'confidence',
               'anchor_x', 'anchor_y', 'signed_distance', 'observed_side', 'class_name', 'event_direction'},
    'events': {'frame', 'source_seconds', 'track_id', 'class', 'direction'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=1):
    require(type(value) is int or isinstance(value, str) and value.isascii() and value.isdigit(),
            f'{name}: expected integer')
    value = int(value)
    require(value >= minimum, f'{name}: out of range')
    return value


def number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: expected finite number') from exc
    require(math.isfinite(result), f'{name}: expected finite number')
    return result


def read_run(directory):
    """Read logs only. Extra schema-v1 CSV columns are retained, never modified."""
    directory = Path(directory)
    with (directory / 'run.json').open(encoding='utf-8-sig') as file:
        manifest = json.load(file)
    tables = []
    for name, required in REQUIRED_FIELDS.items():
        with (directory / f'{name}.csv').open(encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
                    and required <= set(reader.fieldnames), f'{name}: missing/duplicate headers')
            rows = list(reader)
        require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
                f'{name}: malformed CSV row')
        tables.append(rows)
    return manifest, *tables


def generate_candidates(manifest, frames, tracks, events, context_frames=12, top_n=None):
    """Return deterministic suggestions; raise ValueError for inconsistent logs."""
    require(manifest.get('schema_version') == 1 and manifest.get('status') == 'completed'
            and manifest.get('error_code') is None, 'Requires completed schema-v1 run')
    metadata = manifest['metadata']
    require(metadata['source_type'] == 'video', 'Requires video timestamps')
    fps = number(metadata['effective_fps'], 'effective_fps')
    require(fps > 0, 'Invalid FPS')
    max_age = integer(metadata['max_age'], 'max_age', 0)
    margin = number(metadata['margin'], 'margin')
    require(0 <= margin <= NEAR_LINE_PIXELS, 'Unsupported dead-band margin')
    context_frames = integer(context_frames, 'context_frames', 0)
    if top_n is not None:
        top_n = integer(top_n, 'top_n', 0)
    require(frames and len(frames) == manifest['frames_written']
            and len(tracks) == manifest['tracks_written'], 'Manifest row count mismatch')
    for name, rows in zip(REQUIRED_FIELDS, (frames, tracks, events)):
        require(all(REQUIRED_FIELDS[name] <= set(row) and
                    all(isinstance(row[k], str) for k in REQUIRED_FIELDS[name]) for row in rows),
                f'{name}: missing/invalid fields')
    total = len(frames)
    times, by_frame, by_id = {}, defaultdict(list), defaultdict(list)

    def frame_number(row):
        n = integer(row['frame'], 'frame')
        require(n <= total, 'Frame outside processed range')
        return n

    for expected, row in enumerate(frames, 1):
        n = frame_number(row)
        require(n == expected and row['status'] == 'ok', 'Frames must be consecutive and successful')
        seconds = number(row['source_seconds'], 'source_seconds')
        require(abs(seconds - (n - 1) / fps) < 1e-6, 'Inconsistent frame timestamp')
        times[n] = seconds
    signals = []

    def signal(start, end, reason, identity=None):
        signals.append(dict(evidence_id=len(signals)+1, start_frame=start, end_frame=end,
                            start_seconds=times[start], end_seconds=times[end],
                            reason=reason, track_id=identity))

    for row in tracks:
        n = frame_number(row)
        require(abs(number(row['source_seconds'], 'track time') - times[n]) < 1e-6,
                'Track timestamp mismatch')
        confidence = number(row['confidence'], 'confidence')
        require(0 <= confidence <= 1, 'Invalid confidence')
        distance = number(row['signed_distance'], 'signed_distance')
        side = 1 if distance > margin else -1 if distance < -margin else 0
        require(row['observed_side'] == str(side), 'Observed side inconsistent with distance')
        point = (number(row['anchor_x'], 'anchor_x'), number(row['anchor_y'], 'anchor_y'))
        identity = integer(row['track_id'], 'track_id') if row['track_id'] else None
        detection = integer(row['detection_index'], 'detection_index', 0)
        by_frame[n].append((detection, identity))
        if abs(distance) <= NEAR_LINE_PIXELS:
            signal(n, n, 'near_line', identity)
        if confidence < LOW_CONFIDENCE:
            signal(n, n, 'low_confidence', identity)
        if identity is not None:
            by_id[identity].append((n, point, side, abs(distance) <= NEAR_LINE_PIXELS))
        else:
            require(not row['event_direction'], 'Event without track ID')
    for row in frames:
        detections = by_frame[int(row['frame'])]
        require(sorted(d for d, _ in detections) == list(range(integer(row['detection_count'], 'detection_count', 0)))
                and sum(i is not None for _, i in detections) == integer(row['tracked_count'], 'tracked_count', 0),
                'Frame/track counts or detection indices mismatch')
        ids = [i for _, i in detections if i is not None]
        require(len(ids) == len(set(ids)), 'Duplicate ID in one frame')
    directions = ('up', 'down') if metadata['axis'] == 'horizontal' else ('left', 'right')
    require(metadata['axis'] in ('horizontal', 'vertical'), 'Invalid axis')
    predicted = []
    for row in events:
        n = frame_number(row)
        require(abs(number(row['source_seconds'], 'event time') - round(times[n], 3)) < 1e-8,
                'Event timestamp mismatch')
        require(row['direction'] in directions, 'Invalid event direction')
        predicted.append((n, integer(row['track_id'], 'track_id'), row['class'], row['direction']))
    logged = [(int(r['frame']), int(r['track_id']), r['class_name'], r['event_direction'])
              for r in tracks if r['event_direction']]
    require(predicted == logged, 'Events/tracks ordered event sequence mismatch')
    counted_ids = {identity for _, identity, _, _ in predicted}
    for identity, observations in by_id.items():
        observations.sort()
        previous = None
        last_side = 0
        changes = deque()
        for n, point, side, near in observations:
            if previous is not None:
                before, old_point = previous
                gap = n - before
                if gap > 1:
                    signal(before, n, 'tracking_id_gap', identity)
                if gap == 1 and math.dist(point, old_point) > JUMP_PIXELS_PER_FRAME:
                    signal(before, n, 'abrupt_position_change', identity)
                if gap > max_age:
                    last_side = 0
                    changes.clear()
            while changes and n - changes[0] > SIDE_WINDOW_FRAMES:
                changes.popleft()
            if side:
                if last_side and side != last_side:
                    changes.append(n)
                    if len(changes) >= 2:
                        signal(changes[0], n, 'repeated_side_changes', identity)
                last_side = side
            if near and identity not in counted_ids:
                signal(n, n, 'track_without_counted_event', identity)
            previous = (n, point)
    window_evidence = defaultdict(list)
    for evidence in signals:
        first = (evidence['start_frame']-1) // REVIEW_WINDOW_FRAMES
        last = (evidence['end_frame']-1) // REVIEW_WINDOW_FRAMES
        for index in range(first, last+1):
            window_evidence[index].append(evidence)
    windows = []
    for index, evidence in sorted(window_evidence.items()):
        start = index * REVIEW_WINDOW_FRAMES + 1
        end = min(total, start + REVIEW_WINDOW_FRAMES - 1)
        playback_start, playback_end = max(1, start-context_frames), min(total, end+context_frames)
        reasons = sorted({e['reason'] for e in evidence})
        windows.append(dict(window_id=index+1, start_frame=start, end_frame=end,
                            start_seconds=times[start], end_seconds=times[end],
                            playback_start_frame=playback_start, playback_end_frame=playback_end,
                            playback_start_seconds=times[playback_start],
                            playback_end_seconds=times[playback_end], reasons=reasons,
                            track_ids=sorted({e['track_id'] for e in evidence if e['track_id'] is not None}),
                            evidence_ids=[e['evidence_id'] for e in evidence],
                            priority=max(WEIGHTS[r] for r in reasons)))
    windows.sort(key=lambda r: (-r['priority'], -len(r['reasons']), r['start_frame']))
    return dict(status='REVIEW_CANDIDATES_ONLY', run_id=metadata['run_id'],
                warning='Not ground truth or confirmed errors; review the entire source independently.',
                processed_frames=total, thresholds=dict(near_line_pixels=NEAR_LINE_PIXELS,
                    low_confidence=LOW_CONFIDENCE, jump_pixels_per_frame=JUMP_PIXELS_PER_FRAME,
                    side_window_frames=SIDE_WINDOW_FRAMES, context_frames=context_frames),
                window_size_frames=REVIEW_WINDOW_FRAMES, evidence=signals, windows=windows,
                selected_window_ids=[w['window_id'] for w in windows[:top_n]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--output')
    parser.add_argument('--top-n', type=int)
    args = parser.parse_args()
    report = generate_candidates(*read_run(args.run_directory), top_n=args.top_n)
    result = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as file:
            file.write(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
