"""Optional integration check: real inference on a generated empty video.

Run manually; requires dependencies and downloads pretrained weights on first run.
This verifies plumbing, not detection accuracy on real objects.
"""
import csv
import tempfile
import threading
from pathlib import Path


def main():
    import cv2
    import numpy as np
    from engine import run
    with tempfile.TemporaryDirectory() as temp:
        source = str(Path(temp) / 'empty.avi')
        writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
        if not writer.isOpened():
            raise RuntimeError('VideoWriter unavailable')
        for _ in range(4):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()
        result = run(dict(source=source, camera=False, model='yolo11n.pt', category='Tất cả', axis='horizontal', position=.5, confidence=.25, imgsz=320, device='cpu'), threading.Event(), lambda data: None)
        folder = Path(result['folder'])
        assert not result['counts'], result
        with (folder / 'events.csv').open(encoding='utf-8-sig') as file:
            assert list(csv.DictReader(file)) == []
        with (folder / 'performance.csv').open(encoding='utf-8-sig') as file:
            stats = list(csv.DictReader(file))[0]
            assert stats['frames'] == '4' and stats['status'] == 'completed', stats
        assert (folder / 'summary.csv').is_file()
        print('PASS: YOLO + ByteTrack + video decoding + CSV (4 empty frames).', folder)


if __name__ == '__main__':
    main()
