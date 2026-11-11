"""Synthetic logs only; no real video, inference or project output writes."""
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from review_candidates import REQUIRED_FIELDS, generate_candidates, main, read_run


def track(frame, identity=1, distance=10, confidence=.9, x=50, event=''):
    side = 1 if distance > 8 else -1 if distance < -8 else 0
    return dict(frame=str(frame), source_seconds=str((frame-1)/25), detection_index='0',
                track_id='' if identity is None else str(identity), confidence=str(confidence),
                anchor_x=str(x), anchor_y=str(360+distance), signed_distance=str(distance),
                observed_side=str(side), class_name='person', event_direction=event)


def logs(tracks=(), total=100):
    tracks = [dict(r) for r in tracks]
    frames = []
    for n in range(1, total+1):
        detections = [r for r in tracks if int(r['frame']) == n]
        for index, row in enumerate(detections):
            row['detection_index'] = str(index)
        frames.append(dict(frame=str(n), source_seconds=str((n-1)/25),
                           detection_count=str(len(detections)),
                           tracked_count=str(sum(bool(r['track_id']) for r in detections)), status='ok'))
    events = [dict(frame=r['frame'], source_seconds=str(round(float(r['source_seconds']), 3)),
                   track_id=r['track_id'], direction=r['event_direction'], **{'class': r['class_name']})
              for r in tracks if r['event_direction']]
    manifest = dict(schema_version=1, status='completed', error_code=None,
                    frames_written=total, tracks_written=len(tracks),
                    metadata=dict(run_id='synthetic', source_type='video', effective_fps=25,
                                  max_age=90, margin=8, axis='horizontal'))
    return manifest, frames, tracks, events


class ReviewCandidatesTests(unittest.TestCase):
    def generate(self, tracks=(), total=100, context=0):
        return generate_candidates(*logs(tracks, total), context_frames=context)

    def reasons(self, report):
        return {reason for r in report['windows'] for reason in r['reasons']}

    def test_empty_and_far_ordinary_tracks(self):
        for tracks in ([], [track(20, distance=100), track(21, distance=101)]):
            self.assertEqual(self.generate(tracks)['windows'], [])

    def test_near_line_and_no_counted_event(self):
        r = self.generate([track(20)])
        self.assertEqual(self.reasons(r), {'near_line', 'track_without_counted_event'})
        interval = r['windows'][0]
        self.assertEqual((interval['start_frame'], interval['end_frame']), (1, 25))
        self.assertEqual(interval['track_ids'], [1])
        self.assertEqual(interval['start_seconds'], 0)
        self.assertEqual(interval['end_seconds'], 24/25)
        self.assertEqual([(e['start_frame'], e['end_frame']) for e in r['evidence']], [(20, 20)]*2)
        self.assertEqual(r['status'], 'REVIEW_CANDIDATES_ONLY')

    def test_near_line_includes_threshold_both_sides(self):
        for distance in (-24, 0, 24):
            self.assertIn('near_line', self.reasons(self.generate([track(20, distance=distance)])))
        for distance in (-24.01, 24.01):
            self.assertNotIn('near_line', self.reasons(self.generate([track(20, distance=distance)])))

    def test_low_confidence_strict_threshold(self):
        self.assertEqual(self.reasons(self.generate([track(20, distance=100, confidence=.349)])),
                         {'low_confidence'})
        self.assertEqual(self.generate([track(20, distance=100, confidence=.35)])['windows'], [])

    def test_missing_id_still_generates_detection_signals(self):
        r = self.generate([track(20, identity=None, confidence=.2)])
        self.assertEqual(self.reasons(r), {'near_line', 'low_confidence'})
        self.assertEqual(r['windows'][0]['track_ids'], [])

    def test_counted_track_does_not_get_no_event_reason(self):
        r = self.generate([track(20), track(21, distance=-10, event='up')])
        self.assertNotIn('track_without_counted_event', self.reasons(r))
        self.assertIn('near_line', self.reasons(r))

    def test_id_gap_spans_both_endpoints(self):
        r = self.generate([track(20, distance=100), track(23, distance=101)])
        self.assertEqual(self.reasons(r), {'tracking_id_gap'})
        self.assertEqual((r['windows'][0]['start_frame'], r['windows'][0]['end_frame']), (1, 25))
        self.assertEqual((r['evidence'][0]['start_frame'], r['evidence'][0]['end_frame']), (20, 23))
        self.assertEqual(r['windows'][0]['priority'], 3)

    def test_different_ids_do_not_imply_id_gap(self):
        r = self.generate([track(20, 1, distance=100), track(23, 2, distance=101)])
        self.assertEqual(r['windows'], [])

    def test_abrupt_jump_only_consecutive_frames_and_strict_threshold(self):
        r = self.generate([track(20, distance=100), track(21, distance=100, x=131)])
        self.assertEqual(self.reasons(r), {'abrupt_position_change'})
        self.assertNotIn('abrupt_position_change', self.reasons(self.generate(
            [track(20, distance=100), track(21, distance=100, x=130)])))
        self.assertNotIn('abrupt_position_change', self.reasons(self.generate(
            [track(20, distance=100), track(23, distance=100, x=500)])))

    def test_repeated_side_changes_ignore_dead_band(self):
        r = self.generate([track(20, distance=-10), track(21, distance=0),
                           track(22, distance=10), track(23, distance=-10)])
        self.assertIn('repeated_side_changes', self.reasons(r))
        r = self.generate([track(20, distance=-10), track(21, distance=0), track(22, distance=-10)])
        self.assertNotIn('repeated_side_changes', self.reasons(r))

    def test_side_change_window_expires(self):
        r = self.generate([track(1, distance=-10), track(2, distance=10), track(28, distance=-10)])
        self.assertNotIn('repeated_side_changes', self.reasons(r))
        r = self.generate([track(1, distance=-10), track(2, distance=10), track(27, distance=-10)])
        self.assertIn('repeated_side_changes', self.reasons(r))

    def test_side_history_resets_after_expiration(self):
        args = logs([track(1, distance=-10), track(2, distance=10), track(5, distance=-10)])
        args[0]['metadata']['max_age'] = 2
        r = generate_candidates(*args, context_frames=0)
        self.assertNotIn('repeated_side_changes', self.reasons(r))

    def test_context_clipped_to_run_boundaries(self):
        r = self.generate([track(1), track(100, 2)], context=12)
        self.assertEqual([(i['start_frame'], i['end_frame']) for i in r['windows']], [(1, 25), (76, 100)])
        self.assertEqual([(i['playback_start_frame'], i['playback_end_frame'])
                          for i in r['windows']], [(1, 37), (64, 100)])

    def test_overlaps_do_not_merge_windows_union_local_reasons_ids(self):
        r = self.generate([track(20), track(30, 2, distance=100, confidence=.2), track(40, 3)], context=5)
        self.assertEqual(len(r['windows']), 2)
        i = r['windows'][0]
        self.assertEqual((i['start_frame'], i['end_frame']), (26, 50))
        self.assertEqual(i['track_ids'], [2, 3])
        self.assertEqual(i['reasons'], ['low_confidence', 'near_line', 'track_without_counted_event'])

    def test_adjacent_grid_windows_not_merged(self):
        r = self.generate([track(25, 1), track(26, 2)])
        self.assertEqual(len(r['windows']), 2)

    def test_priority_then_reason_count_then_start(self):
        r = self.generate([track(1), track(20, 2, distance=100, confidence=.2),
                           track(40, 3, distance=100), track(42, 3, distance=100), track(70, 4)])
        self.assertEqual([i['start_frame'] for i in r['windows']], [26, 1, 51])

    def test_events_consistency_and_timestamp_validation(self):
        args = logs([track(20, event='down')])
        args[3].clear()
        with self.assertRaisesRegex(ValueError, 'sequence mismatch'):
            generate_candidates(*args)
        args = logs([track(20, event='down')])
        args[3][0]['source_seconds'] = '9'
        with self.assertRaisesRegex(ValueError, 'Event timestamp'):
            generate_candidates(*args)

    def test_partial_webcam_manifest_mismatch_rejected(self):
        for change in ('partial', 'webcam', 'count'):
            args = logs()
            if change == 'partial':
                args[0]['status'] = 'running'
            elif change == 'webcam':
                args[0]['metadata']['source_type'] = 'webcam'
            else:
                args[0]['tracks_written'] = 10
            with self.subTest(change=change), self.assertRaises(ValueError):
                generate_candidates(*args)

    def test_bad_frames_counts_ids_geometry_rejected(self):
        for field, value in [('frame', '0'), ('frame', '101'), ('track_id', '-1'),
                             ('anchor_x', 'nan'), ('confidence', '1.1'), ('observed_side', '-1'),
                             ('source_seconds', '99'), ('detection_index', '9')]:
            args = logs([track(20)])
            args[2][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                generate_candidates(*args)
        with self.assertRaisesRegex(ValueError, 'Duplicate ID'):
            self.generate([track(20), track(20)])

    def test_frames_must_be_consecutive_and_timestamp_consistent(self):
        for field, value in [('frame', '2'), ('source_seconds', '9'), ('status', 'error'),
                             ('detection_count', '1')]:
            args = logs()
            args[1][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                generate_candidates(*args)

    def test_output_deterministic_and_inputs_unchanged(self):
        args = logs([track(20), track(23)])
        before = json.dumps(args)
        self.assertEqual(generate_candidates(*args), generate_candidates(*args))
        self.assertEqual(json.dumps(args), before)

    def test_all_six_heuristics_preserve_raw_evidence(self):
        r = self.generate([track(1, distance=-10, confidence=.2),
                           track(2, distance=10, x=131), track(3, distance=-10, x=131),
                           track(5, distance=10, x=131)], context=12)
        evidence = r['evidence']
        self.assertEqual(len(evidence), 13)
        self.assertEqual([e['evidence_id'] for e in evidence], list(range(1, 14)))
        self.assertEqual({e['reason'] for e in evidence},
                         {'near_line', 'low_confidence', 'tracking_id_gap',
                          'abrupt_position_change', 'repeated_side_changes', 'track_without_counted_event'})
        self.assertTrue(all(e['track_id'] == 1 for e in evidence))
        for reason, expected in (
            ('near_line', [(1, 1), (2, 2), (3, 3), (5, 5)]),
            ('low_confidence', [(1, 1)]), ('tracking_id_gap', [(3, 5)]),
            ('abrupt_position_change', [(1, 2)]),
            ('repeated_side_changes', [(2, 3), (2, 5)]),
            ('track_without_counted_event', [(1, 1), (2, 2), (3, 3), (5, 5)]),
        ):
            self.assertEqual([(e['start_frame'], e['end_frame']) for e in evidence
                              if e['reason'] == reason], expected)

    def test_duplicate_raw_items_are_not_deduplicated(self):
        r = self.generate([track(20, identity=None), track(20, identity=None)])
        self.assertEqual(len(r['evidence']), 2)
        self.assertEqual(r['windows'][0]['evidence_ids'], [1, 2])
        self.assertTrue(all(e['track_id'] is None for e in r['evidence']))

    def test_window_size_and_last_partial_window(self):
        for total in (1, 24, 25, 26, 100, 341):
            r = self.generate([track(n, identity=None) for n in range(1, total+1)], total=total)
            with self.subTest(total=total):
                self.assertEqual(len(r['windows']), (total+24)//25)
                self.assertTrue(all(1 <= w['end_frame']-w['start_frame']+1 <= 25
                                    for w in r['windows']))
                self.assertEqual(max(w['end_frame'] for w in r['windows']), total)

    def test_overlap_chain_cannot_create_long_window(self):
        r = self.generate([track(n, identity=None, confidence=.2)
                           for n in (1, 13, 25, 37, 49, 61, 73, 85, 97)], context=12)
        self.assertEqual([(w['start_frame'], w['end_frame']) for w in r['windows']],
                         [(1, 25), (26, 50), (51, 75), (76, 100)])
        self.assertEqual(len(r['evidence']), 18)

    def test_boundary_spanning_evidence_links_every_intersected_window(self):
        r = self.generate([track(25, distance=100), track(76, distance=100)], context=12)
        self.assertEqual(len(r['evidence']), 1)
        e = r['evidence'][0]
        self.assertEqual((e['start_frame'], e['end_frame']), (25, 76))
        self.assertEqual([w['window_id'] for w in r['windows']], [1, 2, 3, 4])
        self.assertTrue(all(w['evidence_ids'] == [e['evidence_id']] for w in r['windows']))

    def test_context_does_not_change_evidence_links_or_ranking(self):
        args = logs([track(25, 1), track(26, 2)])
        first = generate_candidates(*args, context_frames=0)
        second = generate_candidates(*args, context_frames=12)
        self.assertEqual(first['evidence'], second['evidence'])
        self.assertEqual(first['selected_window_ids'], second['selected_window_ids'])
        for a, b in zip(first['windows'], second['windows']):
            self.assertEqual({k: v for k, v in a.items() if not k.startswith('playback_')},
                             {k: v for k, v in b.items() if not k.startswith('playback_')})

    def test_associations_are_exact_raw_intersections(self):
        r = self.generate([track(25, distance=100), track(26, distance=100),
                           track(75, distance=100), track(100, identity=None)], context=12)
        for w in r['windows']:
            self.assertEqual(w['evidence_ids'], [e['evidence_id'] for e in r['evidence']
                             if e['start_frame'] <= w['end_frame'] and e['end_frame'] >= w['start_frame']])
        self.assertEqual({i for w in r['windows'] for i in w['evidence_ids']},
                         {e['evidence_id'] for e in r['evidence']})

    def test_top_n_retains_complete_evidence_and_windows(self):
        args = logs([track(1), track(40, 2), track(90, 3)])
        full = generate_candidates(*args)
        for limit in (0, 1, 2, 100):
            with self.subTest(limit=limit):
                selected = generate_candidates(*args, top_n=limit)
                self.assertEqual(selected['evidence'], full['evidence'])
                self.assertEqual(selected['windows'], full['windows'])
                self.assertEqual(selected['selected_window_ids'], full['selected_window_ids'][:limit])
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generate_candidates(*args, top_n=invalid)

    def test_repeated_detections_do_not_boost_ranking(self):
        r = self.generate([track(1, distance=100, confidence=.2)] +
                          [track(n, 2, distance=100, confidence=.2) for n in range(26, 51)])
        self.assertEqual([w['start_frame'] for w in r['windows']], [1, 26])
        self.assertEqual([w['priority'] for w in r['windows']], [2, 2])

    def test_raw_window_and_playback_timestamps(self):
        r = self.generate([track(25, distance=100), track(26, distance=100)], context=12)
        for item in r['windows'] + r['evidence']:
            self.assertEqual(item['start_seconds'], (item['start_frame']-1)/25)
            self.assertEqual(item['end_seconds'], (item['end_frame']-1)/25)
        for w in r['windows']:
            self.assertEqual(w['playback_start_seconds'], (w['playback_start_frame']-1)/25)
            self.assertEqual(w['playback_end_seconds'], (w['playback_end_frame']-1)/25)

    def test_cli_roundtrip_and_no_overwrite(self):
        args = logs([track(20)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'run.json').write_text(json.dumps(args[0]), encoding='utf-8')
            for name, rows in zip(REQUIRED_FIELDS, args[1:]):
                with (root/f'{name}.csv').open('w', newline='', encoding='utf-8-sig') as file:
                    writer = csv.DictWriter(file, fieldnames=sorted(REQUIRED_FIELDS[name]))
                    writer.writeheader()
                    writer.writerows(rows)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            self.assertEqual(read_run(root), args)
            stdout = io.StringIO()
            with patch('sys.argv', ['review_candidates.py', str(root), '--top-n', '0']), redirect_stdout(stdout):
                main()
            printed = json.loads(stdout.getvalue())
            self.assertEqual(printed['status'], 'REVIEW_CANDIDATES_ONLY')
            self.assertEqual(printed['selected_window_ids'], [])
            self.assertEqual(len(printed['evidence']), 2)
            self.assertEqual(len(printed['windows']), 1)
            with patch('sys.argv', ['review_candidates.py', str(root), '--output', str(root/'report.json')]):
                main()
                saved = (root/'report.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    main()
                self.assertEqual((root/'report.json').read_bytes(), saved)
            self.assertTrue(all((root/n).read_bytes() == data for n, data in before.items()))

    def test_bad_csv_headers_and_truncated_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'run.json').write_text('{}', encoding='utf-8')
            for contents in ('frame,frame\n1,1\n', ','.join(sorted(REQUIRED_FIELDS['frames']))+'\n1\n'):
                (root/'frames.csv').write_text(contents, encoding='utf-8')
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    read_run(root)


if __name__ == '__main__':
    unittest.main()
