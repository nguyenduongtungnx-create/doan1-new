import unittest
from counter import LineCounter


class CounterTests(unittest.TestCase):
    def test_hysteresis_and_direction_deduplication(self):
        counter = LineCounter()
        events = [counter.update(1, 'person', (50, y), i, 100, 100)
                  for i, y in enumerate([20, 49, 52, 65, 70, 20, 70])]
        self.assertEqual([e for e in events if e], ['down', 'up'])
        self.assertEqual(sum(counter.counts.values()), 2)

    def test_initially_on_line_is_not_a_crossing(self):
        counter = LineCounter()
        self.assertIsNone(counter.update(1, 'car', (0, 50), 1, 100, 100))
        self.assertIsNone(counter.update(1, 'car', (0, 80), 2, 100, 100))

    def test_vertical_and_independent_tracks(self):
        counter = LineCounter('vertical')
        for identity in [1, 2]:
            counter.update(identity, 'car', (20, 0), 1, 100, 100)
            self.assertEqual(counter.update(identity, 'car', (80, 0), 2, 100, 100), 'right')
        self.assertEqual(counter.counts['car', 'right'], 2)

    def test_stale_track_cannot_cross(self):
        counter = LineCounter(max_age=3)
        counter.update(1, 'car', (0, 20), 1, 100, 100)
        self.assertIsNone(counter.update(1, 'car', (0, 80), 9, 100, 100))
        counter.prune(20)
        self.assertFalse(counter.tracks)


class CounterEdgeCaseTests(unittest.TestCase):
    @staticmethod
    def feed(counter, positions):
        return [
            counter.update(
                1, 'person', (50, y), frame, 100, 100
            )
            for frame, y in enumerate(positions, start=1)
        ]

    def test_repeated_crossings_count_once_per_direction(self):
        counter = LineCounter()
        events = self.feed(counter, [20, 80, 20, 80, 20])

        self.assertEqual(
            events, [None, 'down', 'up', None, None]
        )
        self.assertEqual(
            dict(counter.counts),
            {('person', 'down'): 1, ('person', 'up'): 1},
        )

    def test_touching_line_and_band_boundaries_does_not_count(self):
        counter = LineCounter()
        events = self.feed(counter, [20, 42, 50, 58, 20])

        self.assertEqual(events, [None] * 5)
        self.assertFalse(counter.counts)

    def test_anchor_jitter_inside_band_does_not_count(self):
        counter = LineCounter()
        positions = [20, 43, 49, 52, 57, 48, 55, 20]
        events = self.feed(counter, positions)

        self.assertEqual(events, [None] * len(positions))
        self.assertFalse(counter.counts)

    def test_anchor_jitter_beyond_band_can_trigger_counts(self):
        # Documents a limitation: anchor movement alone can count.
        counter = LineCounter()
        events = self.feed(counter, [41, 59, 41, 59])

        self.assertEqual(
            events, [None, 'down', 'up', None]
        )
        self.assertEqual(
            dict(counter.counts),
            {('person', 'down'): 1, ('person', 'up'): 1},
        )

    def test_fast_crossing_needs_no_intermediate_observation(self):
        counter = LineCounter()
        events = self.feed(counter, [10, 90])

        self.assertEqual(events, [None, 'down'])
        self.assertEqual(
            dict(counter.counts), {('person', 'down'): 1}
        )

    def test_id_remains_active_at_exact_expiration_threshold(self):
        counter = LineCounter(max_age=90)
        counter.update(1, 'person', (50, 20), 1, 100, 100)

        event = counter.update(
            1, 'person', (50, 80), 91, 100, 100
        )

        self.assertEqual(event, 'down')
        self.assertEqual(counter.counts['person', 'down'], 1)

    def test_id_after_expiration_cannot_infer_gap_crossing(self):
        counter = LineCounter(max_age=90)
        counter.update(1, 'person', (50, 20), 1, 100, 100)

        event = counter.update(
            1, 'person', (50, 80), 92, 100, 100
        )

        self.assertIsNone(event)
        self.assertFalse(counter.counts)

        # A subsequent observed crossing can count normally.
        self.assertEqual(
            counter.update(
                1, 'person', (50, 20), 93, 100, 100
            ),
            'up',
        )

    def test_pruning_resets_per_direction_deduplication(self):
        counter = LineCounter(max_age=90)
        self.assertEqual(
            self.feed(counter, [20, 80]), [None, 'down']
        )

        counter.prune(92)  # Exactly 90 frames since last seen.
        self.assertIn(1, counter.tracks)

        counter.prune(93)  # More than 90 frames since last seen.
        self.assertNotIn(1, counter.tracks)

        self.assertIsNone(
            counter.update(
                1, 'person', (50, 20), 94, 100, 100
            )
        )
        self.assertEqual(
            counter.update(
                1, 'person', (50, 80), 95, 100, 100
            ),
            'down',
        )
        self.assertEqual(counter.counts['person', 'down'], 2)


if __name__ == '__main__':
    unittest.main()
