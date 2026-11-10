"""Line crossing with hysteresis; one event per ID and direction."""
from collections import Counter


class LineCounter:
    def __init__(self, axis='horizontal', position=0.5, margin=8, max_age=90):
        if axis not in ('horizontal', 'vertical') or not 0 < position < 1:
            raise ValueError('Invalid line')
        self.axis, self.position, self.margin, self.max_age = axis, position, margin, max_age
        self.tracks = {}
        self.counts = Counter()

    def update(self, identity, label, point, frame, width, height):
        coordinate, size = (point[1], height) if self.axis == 'horizontal' else (point[0], width)
        distance = coordinate - self.position * size
        side = 1 if distance > self.margin else -1 if distance < -self.margin else 0
        state = self.tracks.get(identity)
        if state is None or frame - state['seen'] > self.max_age:
            self.tracks[identity] = dict(side=side, seen=frame, counted=set())
            return None
        state['seen'] = frame
        if not side:
            return None
        previous = state['side']
        state['side'] = side
        if not previous or previous == side:
            return None
        direction = ('down' if side > 0 else 'up') if self.axis == 'horizontal' else ('right' if side > 0 else 'left')
        if direction in state['counted']:
            return None
        state['counted'].add(direction)
        self.counts[label, direction] += 1
        return direction

    def prune(self, frame):
        self.tracks = {k: v for k, v in self.tracks.items() if frame - v['seen'] <= self.max_age}
