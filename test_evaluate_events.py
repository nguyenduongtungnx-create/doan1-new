"""Synthetic event evaluation only; no project outputs, models or video reads."""
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evaluate_events import EVENT_FIELDS, GT_FIELDS, MATCH_FIELDS, evaluate, main, read_csv


def metadata(**changes):
    result = dict(schema_version=1, source_type='video', frame_count=100, fps=25,
                  axis='horizontal', timestamp_semantics='confirmation_frame', complete_video=True,
                  review_coverage=[dict(start_frame=1, end_frame=100,
                                        reviewer='B', notes='Full clip and no-event intervals reviewed')])
    result.update(changes)
    return result


def prediction(frame=30, identity=7, direction='up'):
    return dict(wall_time='2026-01-01T00:00:00', source_seconds=str(round((frame-1)/25, 3)),
                frame=str(frame), track_id=str(identity), **{'class': 'person', 'direction': direction})


def truth(event_id='G1', person_id='P1', confirmation=30, direction='up', **changes):
    result = dict(event_id=event_id, person_id=person_id, direction=direction,
                  crossing_frame=str(confirmation-2), confirmation_frame=str(confirmation),
                  status='verified', uncertainty='', reviewer='A', verified_by='B', notes='',
                  **{'class': 'person'})
    result.update(changes)
    return result


def decision(prediction_id=1, event_id='G1', verdict='matched'):
    return dict(prediction_id=str(prediction_id), event_id=event_id, verdict=verdict,
                reviewer='B', notes='Identity and event checked visually')


class EvaluateEventsTests(unittest.TestCase):
    def test_normal_match(self):
        r = evaluate([prediction()], [truth()], [decision()], metadata())
        self.assertEqual((r['TP'], r['FP'], r['FN']), (1, 0, 0))
        self.assertEqual((r['precision'], r['recall'], r['F1']), (1, 1, 1))
        self.assertEqual(r['directional_count_error']['up']['signed'], 0)
        self.assertEqual(r['scope'], 'complete_video')

    def test_missing_crossing(self):
        r = evaluate([], [truth(direction='down')], [], metadata())
        self.assertEqual((r['TP'], r['FP'], r['FN']), (0, 0, 1))
        self.assertEqual(r['missed_gt_ids'], ['G1'])
        self.assertIsNone(r['precision'])
        self.assertEqual((r['recall'], r['F1']), (0, 0))
        self.assertEqual(r['directional_count_error']['down']['signed'], -1)

    def test_duplicate_prediction_is_explicit_fp(self):
        r = evaluate([prediction(), prediction(31, 99)], [truth()],
                     [decision(2, verdict='duplicate'), decision()], metadata())
        self.assertEqual((r['TP'], r['FP'], r['FN']), (1, 1, 0))
        self.assertEqual(r['precision'], 0.5)
        self.assertAlmostEqual(r['F1'], 2/3)
        self.assertEqual(r['directional_count_error']['up']['signed'], 1)

    def test_duplicate_without_true_match_rejected(self):
        with self.assertRaisesRegex(ValueError, 'existing true match'):
            evaluate([prediction()], [truth()], [decision(verdict='duplicate')], metadata())

    def test_wrong_direction_explicit_fp_and_fn(self):
        r = evaluate([prediction(direction='down')], [truth()],
                     [decision(event_id='', verdict='false_positive')], metadata())
        self.assertEqual((r['TP'], r['FP'], r['FN']), (0, 1, 1))
        self.assertEqual(r['directional_count_error']['down']['signed'], 1)
        self.assertEqual(r['directional_count_error']['up']['signed'], -1)

    def test_direction_or_class_mismatch_cannot_match(self):
        for changes in ({'direction': 'down'}, {'class': 'car'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'class/direction'):
                evaluate([dict(prediction(), **changes)], [truth()], [decision()], metadata())

    def test_ambiguous_incomplete_pending_and_rejected_are_separate(self):
        gt = [truth('G1', status='ambiguous', uncertainty='Occluded', confirmation_frame=''),
              truth('G2', status='incomplete', uncertainty='Video ends', confirmation_frame=''),
              truth('G3', status='pending', uncertainty='Awaiting reviewer'),
              truth('G4', status='rejected', notes='Retreat inside band')]
        r = evaluate([prediction()], gt, [], metadata(complete_video=False))
        self.assertEqual((r['TP'], r['FP'], r['FN']), (0, 0, 0))
        self.assertEqual(r['pending_prediction_ids'], [1])
        self.assertEqual(r['unresolved_gt_ids'], ['G1', 'G2', 'G3'])
        self.assertEqual(r['gt_status_counts'], dict(ambiguous=1, incomplete=1, pending=1, rejected=1))

    def test_unreviewed_prediction_never_automatically_fp(self):
        for matches in ([], [decision(event_id='', verdict='pending')]):
            r = evaluate([prediction()], [], matches, metadata(complete_video=False))
            self.assertEqual(r['FP'], 0)
            self.assertEqual(r['pending_prediction_ids'], [1])

    def test_simultaneous_people_require_explicit_pairs(self):
        r = evaluate([prediction(identity=7), prediction(identity=8)],
                     [truth('G1', 'P1'), truth('G2', 'P2')],
                     [decision(1, 'G2'), decision(2, 'G1')], metadata())
        self.assertEqual(r['TP'], 2)

    def test_repeated_physical_crossings_are_retained(self):
        r = evaluate([prediction(), prediction(70)],
                     [truth(), truth('G2', 'P1', 70)],
                     [decision(), decision(2, 'G2')], metadata())
        self.assertEqual(r['TP'], 2)
        self.assertEqual(r['directional_count_error']['up']['ground_truth'], 2)

    def test_zero_denominators(self):
        r = evaluate([], [], [], metadata())
        self.assertEqual((r['precision'], r['recall'], r['F1']), (None, None, None))
        r = evaluate([prediction()], [], [decision(event_id='', verdict='false_positive')], metadata())
        self.assertEqual(r['precision'], 0)
        self.assertIsNone(r['recall'])
        self.assertEqual(r['F1'], 0)

    def test_one_to_one_enforced_both_sides(self):
        for predictions, gt, matches in (
            ([prediction()], [truth(), truth('G2')], [decision(), decision(1, 'G2')]),
            ([prediction(), prediction()], [truth()], [decision(), decision(2)]),
        ):
            with self.subTest(matches=matches), self.assertRaises(ValueError):
                evaluate(predictions, gt, matches, metadata())

    def test_duplicate_event_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate GT'):
            evaluate([], [truth(), truth()], [], metadata())

    def test_invalid_frames(self):
        for value in ('0', '-1', '101', '2.5', 'True', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate([dict(prediction(), frame=value)], [], [], metadata(complete_video=False))
        for changes in (dict(crossing_frame='0'), dict(confirmation_frame='101'),
                        dict(crossing_frame='40'), dict(confirmation_frame='')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate([], [truth(**changes)], [], metadata())

    def test_confirmation_frame_tolerance_not_crossing_frame(self):
        r = evaluate([prediction(42)], [truth(crossing_frame='1')], [decision()], metadata())
        self.assertEqual(r['TP'], 1)
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            evaluate([prediction(43)], [truth()], [decision()], metadata())
        self.assertEqual(evaluate([prediction(43)], [truth()], [decision()],
                                 metadata(tolerance_frames=13))['TP'], 1)
        self.assertEqual(evaluate([prediction(18)], [truth()], [decision()], metadata())['TP'], 1)
        with self.assertRaises(ValueError):
            evaluate([prediction(29)], [truth()], [decision()], metadata(tolerance_frames=0))

    def test_timestamp_semantics_validated(self):
        for seconds in ('1.201', 'nan', 'inf', 'bad', ''):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                evaluate([dict(prediction(), source_seconds=seconds)], [], [], metadata())
        with self.assertRaisesRegex(ValueError, 'timestamp semantics'):
            evaluate([], [], [], metadata(timestamp_semantics='crossing_frame'))

    def test_complete_video_requires_full_coverage(self):
        for coverage in ([], [dict(start_frame=2, end_frame=100, reviewer='B', notes='Partial')]):
            with self.subTest(coverage=coverage), self.assertRaisesRegex(ValueError, 'full coverage'):
                evaluate([], [], [], metadata(review_coverage=coverage))

    def test_complete_video_rejects_pending_predictions_and_gt(self):
        for pred, gt in (([prediction()], []),
                         ([], [truth(status='ambiguous', uncertainty='Occluded')]),
                         ([], [truth(status='incomplete', uncertainty='Not completed')])):
            with self.subTest(gt=gt), self.assertRaisesRegex(ValueError, 'unresolved'):
                evaluate(pred, gt, [], metadata())

    def test_partial_scope_does_not_count_uncovered_gt_as_fn(self):
        coverage = [dict(start_frame=1, end_frame=20, reviewer='B', notes='Reviewed opening')]
        r = evaluate([prediction()], [truth()], [], metadata(complete_video=False, review_coverage=coverage))
        self.assertEqual((r['scope'], r['FN'], r['FP']), ('partial', 0, 0))
        self.assertEqual(r['unresolved_gt_ids'], ['G1'])
        with self.assertRaisesRegex(ValueError, 'outside reviewed coverage'):
            evaluate([prediction()], [], [decision(event_id='', verdict='false_positive')],
                     metadata(complete_video=False, review_coverage=coverage))

    def test_coverage_must_include_whole_confirmation_interval(self):
        coverage = [dict(start_frame=30, end_frame=100, reviewer='B', notes='End reviewed')]
        r = evaluate([], [truth()], [], metadata(complete_video=False, review_coverage=coverage))
        self.assertEqual(r['FN'], 0)

    def test_coverage_validation(self):
        for coverage in (
            [dict(start_frame=1, end_frame=100, reviewer='', notes='Reviewed')],
            [dict(start_frame=20, end_frame=10, reviewer='B', notes='Reviewed')],
            [dict(start_frame=1, end_frame=100, reviewer='B', notes='Reviewed')]*2,
        ):
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                evaluate([], [], [], metadata(review_coverage=coverage))

    def test_verified_gt_requires_independent_review_and_no_uncertainty(self):
        for changes in (dict(verified_by=''), dict(verified_by='A'), dict(uncertainty='Unsure')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate([], [truth(**changes)], [], metadata())

    def test_invalid_gt_status_and_uncertainty(self):
        for changes in (dict(status='draft'), dict(status='ambiguous'), dict(status='rejected')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate([], [truth(**changes)], [], metadata())

    def test_cannot_match_unverified_gt(self):
        with self.assertRaisesRegex(ValueError, 'verified GT'):
            evaluate([prediction()], [truth(status='pending', uncertainty='Awaiting review')],
                     [decision()], metadata(complete_video=False))

    def test_invalid_decisions(self):
        for changes in (dict(prediction_id='2'), dict(verdict='automatic'), dict(reviewer=''),
                        dict(notes=''), dict(event_id='missing'), dict(verdict='pending'),
                        dict(verdict='false_positive')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate([prediction()], [truth()], [dict(decision(), **changes)], metadata())

    def test_evaluation_schema(self):
        for changes in (dict(fps=0), dict(fps=float('nan')), dict(fps=True), dict(frame_count=True),
                        dict(tolerance_frames=-1), dict(complete_video='yes'), dict(axis='diagonal'),
                        dict(source_type='webcam'), dict(extra=1), dict(schema_version=2)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate([], [], [], metadata(**changes))

    def test_vertical_directions(self):
        r = evaluate([prediction(direction='left')], [truth(direction='left')], [decision()],
                     metadata(axis='vertical'))
        self.assertEqual(r['TP'], 1)
        with self.assertRaises(ValueError):
            evaluate([prediction()], [], [], metadata(axis='vertical'))

    def test_api_does_not_mutate_inputs(self):
        inputs = ([prediction()], [truth()], [decision()], metadata())
        before = json.dumps(inputs)
        evaluate(*inputs)
        self.assertEqual(json.dumps(inputs), before)

    def test_cli_csv_json_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, headers, data in (('events', EVENT_FIELDS, [prediction()]),
                                         ('gt', GT_FIELDS, [truth(notes='Người áo sáng')]),
                                         ('matches', MATCH_FIELDS, [decision()])):
                with (root/name).open('w', encoding='utf-8-sig', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(data)
            (root/'evaluation').write_text(json.dumps(metadata()), encoding='utf-8')
            original = {p.name: p.read_bytes() for p in root.iterdir()}
            args = ['evaluate_events.py'] + [str(root/n) for n in ('events', 'gt', 'matches', 'evaluation')]
            stdout = io.StringIO()
            with patch('sys.argv', args), redirect_stdout(stdout):
                main()
            self.assertEqual(json.loads(stdout.getvalue())['TP'], 1)
            with patch('sys.argv', args + ['--output', str(root/'report')]):
                main()
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(json.loads((root/'report').read_text())['scope'], 'complete_video')
            self.assertTrue(all((root/n).read_bytes() == data for n, data in original.items()))

    def test_csv_header_and_incomplete_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.csv'
            for contents in ('bad\nvalue\n', ','.join(GT_FIELDS)+'\nG1,P1\n',
                             ','.join(GT_FIELDS)+'\n'+','.join(['x']*12)+'\n'):
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        read_csv(path, GT_FIELDS)


if __name__ == '__main__':
    unittest.main()
