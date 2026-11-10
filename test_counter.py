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


if __name__ == '__main__':
    unittest.main()
