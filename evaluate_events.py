"""Standalone manual event evaluation; standard library only, no video/inference.

Input schemas (exact CSV headers, UTF-8/BOM accepted):
ground_truth.csv: event_id,person_id,class,direction,crossing_frame,
confirmation_frame,status,uncertainty,reviewer,verified_by,notes
Status: verified/ambiguous/incomplete/rejected/pending. Only verified rows enter
metrics; they require independent reviewer names and both frames. Uncertain rows
require uncertainty text and may omit frames. Rejected rows require notes.
IDs identify physical people independently of tracker IDs; retain repeated
physical crossings as separate events, including repeats in the same direction.

matches.csv: prediction_id,event_id,verdict,reviewer,notes
prediction_id is the one-based events.csv data-row number (never track_id).
matched requires a verified GT of the same class/direction within tolerance;
the reviewer attests visual identity, not merely timing. duplicate is an explicit
FP referencing a GT already matched to another prediction, of same class/direction.
false_positive has no GT reference; wrong direction is recorded this way with a
note and leaves the real event unmatched (FN). pending has no GT reference.
Missing decisions remain pending. Non-pending decisions require reviewer/notes.

evaluation.json: schema_version=1, source_type="video", frame_count (positive),
fps (positive), axis="horizontal"/"vertical", timestamp_semantics=
"confirmation_frame", complete_video (bool), review_coverage (list of objects
with start_frame,end_frame,reviewer,notes). Optional tolerance_frames defaults to
12. Coverage is inclusive, non-overlapping, and documents review of ALL activity,
including intervals without predictions. Fixed-FPS videos only: source_seconds
must equal round((frame-1)/fps,3), as emitted by engine.run for video sources.
Webcam and variable-FPS timing are not supported by this schema.

Only covered events enter metrics. GT coverage requires both annotated frames;
GT or predictions outside coverage remain pending. Explicit FP/duplicate decisions
outside coverage are invalid. Complete-video metrics require full coverage and
no pending predictions or unresolved (ambiguous/incomplete/pending) GT. Partial
metrics are labelled partial, with exclusions enumerated. Zero denominators are
None (JSON null). Count error = decided predictions minus verified physical GT,
per direction. These metrics do not evaluate the once-per-tracker-ID count policy.

API: evaluate(predictions, ground_truth, matches, metadata), using CSV-style dicts.
CLI: python evaluate_events.py EVENTS GT MATCHES EVALUATION [--output NEW_JSON]
Without --output prints JSON; output uses exclusive creation, never overwrites.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

GT_FIELDS = ('event_id', 'person_id', 'class', 'direction', 'crossing_frame',
             'confirmation_frame', 'status', 'uncertainty', 'reviewer', 'verified_by', 'notes')
MATCH_FIELDS = ('prediction_id', 'event_id', 'verdict', 'reviewer', 'notes')
EVENT_FIELDS = ('wall_time', 'source_seconds', 'frame', 'track_id', 'class', 'direction')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=1):
    require(type(value) is int or isinstance(value, str) and value.isascii() and value.isdigit(),
            f'{name}: expected integer')
    number = int(value)
    require(number >= minimum, f'{name}: out of range')
    return number


def fields(row, expected):
    require(set(row) == set(expected) and all(isinstance(v, str) for v in row.values()),
            'Invalid CSV fields or incomplete row')


def read_csv(path, expected):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames == list(expected), f'{path}: invalid header')
        rows = list(reader)
    for row in rows:
        fields(row, expected)
    return rows


def evaluate(predictions, ground_truth, matches, metadata):
    """Validate manual decisions and return metrics; never infer matches or labels."""
    required = {'schema_version', 'source_type', 'frame_count', 'fps', 'axis',
                'timestamp_semantics', 'complete_video', 'review_coverage'}
    require(required <= set(metadata) <= required | {'tolerance_frames'}, 'Invalid evaluation schema')
    require(type(metadata['schema_version']) is int and metadata['schema_version'] == 1,
            'Unsupported schema version')
    require(metadata['source_type'] == 'video' and
            metadata['timestamp_semantics'] == 'confirmation_frame', 'Unsupported timestamp semantics')
    total = integer(metadata['frame_count'], 'frame_count')
    fps = metadata['fps']
    require(type(fps) in (int, float) and math.isfinite(fps) and fps > 0, 'Invalid fps')
    tolerance = integer(metadata.get('tolerance_frames', 12), 'tolerance_frames', 0)
    require(type(metadata['complete_video']) is bool, 'complete_video must be boolean')
    require(metadata['axis'] in ('horizontal', 'vertical'), 'Invalid axis')
    directions = ('up', 'down') if metadata['axis'] == 'horizontal' else ('left', 'right')

    def frame(value):
        number = integer(value, 'frame')
        require(number <= total, 'Frame exceeds frame_count')
        return number

    require(isinstance(metadata['review_coverage'], list), 'Invalid review coverage')
    covered = set()
    for interval in metadata['review_coverage']:
        require(isinstance(interval, dict) and set(interval) ==
                {'start_frame', 'end_frame', 'reviewer', 'notes'}, 'Invalid coverage fields')
        require(all(isinstance(interval[k], str) and interval[k].strip()
                    for k in ('reviewer', 'notes')), 'Coverage needs reviewer and notes')
        start, end = frame(interval['start_frame']), frame(interval['end_frame'])
        require(start <= end, 'Reversed coverage interval')
        span = set(range(start, end + 1))
        require(not covered & span, 'Overlapping coverage intervals')
        covered.update(span)

    gt, eligible, unresolved = {}, {}, []
    for row in ground_truth:
        fields(row, GT_FIELDS)
        identity = row['event_id']
        require(identity.strip() and identity not in gt, 'Empty or duplicate GT event ID')
        require(row['person_id'].strip() and row['class'].strip(), 'Missing person/class')
        require(row['direction'] in directions, 'Invalid GT direction')
        status = row['status']
        require(status in ('verified', 'ambiguous', 'incomplete', 'rejected', 'pending'), 'Invalid GT status')
        crossing = frame(row['crossing_frame']) if row['crossing_frame'] else None
        confirmation = frame(row['confirmation_frame']) if row['confirmation_frame'] else None
        require(crossing is None or confirmation is None or crossing <= confirmation,
                'Confirmation precedes crossing')
        if status == 'verified':
            require(crossing is not None and confirmation is not None and not row['uncertainty'].strip(),
                    'Verified GT needs exact frames and resolved uncertainty')
            require(row['reviewer'].strip() and row['verified_by'].strip() and
                    row['reviewer'].strip() != row['verified_by'].strip(), 'Independent verification required')
            if all(n in covered for n in range(crossing, confirmation + 1)):
                eligible[identity] = row
            else:
                unresolved.append(identity)
        elif status == 'rejected':
            require(row['reviewer'].strip() and row['notes'].strip(), 'Rejection needs explanation')
        else:
            require(row['uncertainty'].strip(), 'Unresolved GT needs uncertainty')
            unresolved.append(identity)
        gt[identity] = row

    predicted = {}
    for index, row in enumerate(predictions, 1):
        fields(row, EVENT_FIELDS)
        n = frame(row['frame'])
        require(row['class'].strip() and row['direction'] in directions, 'Invalid prediction class/direction')
        integer(row['track_id'], 'track_id')
        try:
            seconds = float(row['source_seconds'])
        except ValueError as exc:
            raise ValueError('Invalid source_seconds') from exc
        require(math.isfinite(seconds) and abs(seconds - round((n - 1) / fps, 3)) < 1e-8,
                'Prediction timestamp does not match fixed-FPS confirmation frame')
        predicted[index] = row

    decisions, linked, duplicates = {}, set(), []
    for row in matches:
        fields(row, MATCH_FIELDS)
        pid = integer(row['prediction_id'], 'prediction_id')
        require(pid in predicted and pid not in decisions, 'Unknown or duplicate prediction decision')
        verdict = row['verdict']
        require(verdict in ('matched', 'false_positive', 'duplicate', 'pending'), 'Invalid verdict')
        event_id = row['event_id']
        if verdict == 'pending':
            require(not event_id, 'Pending decision must not link GT')
        else:
            require(row['reviewer'].strip() and row['notes'].strip(), 'Decision needs reviewer and notes')
            require(int(predicted[pid]['frame']) in covered, 'Decision outside reviewed coverage')
            if verdict == 'false_positive':
                require(not event_id, 'False positive must not link GT')
            else:
                require(event_id in eligible, 'Match needs covered, verified GT')
                actual, prediction = eligible[event_id], predicted[pid]
                require(actual['class'] == prediction['class'] and
                        actual['direction'] == prediction['direction'], 'Inconsistent class/direction')
                require(abs(int(prediction['frame']) - int(actual['confirmation_frame'])) <= tolerance,
                        'Match exceeds confirmation-frame tolerance')
                if verdict == 'matched':
                    require(event_id not in linked, 'GT already matched')
                    linked.add(event_id)
                else:
                    duplicates.append(event_id)
        decisions[pid] = row
    require(all(event_id in linked for event_id in duplicates), 'Duplicate needs an existing true match')
    pending = [pid for pid in predicted if pid not in decisions or decisions[pid]['verdict'] == 'pending']
    if metadata['complete_video']:
        require(len(covered) == total, 'Complete video requires documented full coverage')
        require(not pending and not unresolved, 'Complete video has unresolved events')

    tp = len(linked)
    fp = sum(r['verdict'] in ('false_positive', 'duplicate') for r in decisions.values())
    missed = [identity for identity in eligible if identity not in linked]
    fn = len(missed)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    gt_counts = Counter(r['direction'] for r in eligible.values())
    pred_counts = Counter(predicted[pid]['direction'] for pid, r in decisions.items() if r['verdict'] != 'pending')
    return dict(scope='complete_video' if metadata['complete_video'] else 'partial',
                tolerance_frames=tolerance, reviewed_frames=len(covered), frame_count=total,
                TP=tp, FP=fp, FN=fn, precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                F1=ratio(2 * tp, 2 * tp + fp + fn),
                directional_count_error={d: dict(predicted=pred_counts[d], ground_truth=gt_counts[d],
                    signed=pred_counts[d] - gt_counts[d], absolute=abs(pred_counts[d] - gt_counts[d]))
                    for d in directions},
                pending_prediction_ids=pending, unresolved_gt_ids=unresolved, missed_gt_ids=missed,
                gt_status_counts=dict(Counter(r['status'] for r in gt.values())),
                prediction_verdict_counts=dict(Counter(
                    decisions[pid]['verdict'] if pid in decisions else 'pending' for pid in predicted)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('events', 'ground_truth', 'matches', 'evaluation'):
        parser.add_argument(name)
    parser.add_argument('--output')
    args = parser.parse_args()
    with Path(args.evaluation).open(encoding='utf-8-sig') as file:
        metadata = json.load(file)
    report = evaluate(read_csv(args.events, EVENT_FIELDS), read_csv(args.ground_truth, GT_FIELDS),
                      read_csv(args.matches, MATCH_FIELDS), metadata)
    result = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as file:
            file.write(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
