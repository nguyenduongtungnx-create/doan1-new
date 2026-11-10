import csv
import time
from datetime import datetime
from pathlib import Path
from counter import LineCounter

CLASSES = {'Người': [0], 'Phương tiện': [1, 2, 3, 5, 7], 'Tất cả': [0, 1, 2, 3, 5, 7]}


def run(config, stop, publish):
    import cv2
    from ultralytics import YOLO
    folder = Path(__file__).parent / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=True)
    counter = LineCounter(config['axis'], config['position'])
    cap, processed, status = None, 0, 'completed'
    started = time.perf_counter()
    try:
        publish({'status': 'Đang tải mô hình (lần đầu cần Internet)…', 'folder': str(folder)})
        model = YOLO(config['model'])
        cap = cv2.VideoCapture(int(config['source']) if config['camera'] else config['source'])
        if not cap.isOpened():
            raise RuntimeError('Không mở được video/webcam. Kiểm tra nguồn và quyền camera.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            fps = 30.0
        with (folder / 'events.csv').open('w', newline='', encoding='utf-8-sig') as file:
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
                scale = min(1.0, 1280 / frame.shape[1])
                if scale < 1:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                h, w = frame.shape[:2]
                result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=CLASSES[config['category']], conf=config['confidence'], imgsz=config['imgsz'], device=config['device'], verbose=False)[0]
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for box, identity, cls in zip(boxes.xyxy.cpu().tolist(), boxes.id.int().cpu().tolist(), boxes.cls.int().cpu().tolist()):
                        x1, y1, x2, y2 = box
                        label = model.names[cls]
                        direction = counter.update(identity, label, ((x1+x2)/2, y2), processed, w, h)
                        if direction:
                            seconds = time.perf_counter()-started if config['camera'] else (processed-1)/fps
                            writer.writerow([datetime.now().isoformat(timespec='seconds'), round(seconds, 3), processed, identity, label, direction])
                            file.flush()
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
                publish({'frame': cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 'counts': dict(counter.counts), 'status': f'Frame {processed} • {rate:.1f} FPS', 'folder': str(folder)})
            if stop.is_set():
                status = 'stopped'
    except Exception:
        status = 'error'
        raise
    finally:
        if cap is not None:
            cap.release()
        with (folder / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['class', 'direction', 'count'])
            for (label, direction), count in sorted(counter.counts.items()):
                writer.writerow([label, direction, count])
        with (folder / 'performance.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['model', 'imgsz', 'device', 'frames', 'elapsed_seconds', 'pipeline_fps', 'status'])
            elapsed = time.perf_counter()-started
            writer.writerow([config['model'], config['imgsz'], config['device'], processed, round(elapsed,3), round(processed/max(elapsed,.001),2), status])
    return {'counts': dict(counter.counts), 'folder': str(folder)}
