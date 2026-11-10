"""Vietnamese desktop UI; inference runs in a separate thread."""
import csv
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from engine import CLASSES, run


class App:
    def __init__(self, root):
        self.root = root
        root.title('AI Vision • Đếm người và phương tiện')
        root.geometry('1200x800')
        root.minsize(920, 650)
        self.events = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.worker = None
        self.folder = None
        self.closing = False
        self.source = tk.StringVar()
        self.camera = tk.BooleanVar(value=False)
        self.model = tk.StringVar(value='yolo11n.pt')
        self.category = tk.StringVar(value='Tất cả')
        self.axis = tk.StringVar(value='Ngang')
        self.position = tk.DoubleVar(value=50)
        self.confidence = tk.StringVar(value='0.25')
        self.imgsz = tk.StringVar(value='640')
        self.device = tk.StringVar(value='cpu')
        self.status = tk.StringVar(value='Chọn video hoặc bật webcam để bắt đầu.')
        top = ttk.Frame(root, padding=12)
        top.pack(fill='x')
        ttk.Label(top, text='AI VISION  /  LINE CROSSING', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(top, text='YOLO11 + ByteTrack • Theo dõi ID • Đếm hai chiều • Xuất CSV').pack(anchor='w', pady=(0, 10))
        row = ttk.Frame(top)
        row.pack(fill='x')
        ttk.Label(row, text='Nguồn:').pack(side='left')
        ttk.Entry(row, textvariable=self.source).pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(row, text='Chọn video', command=self.browse).pack(side='left')
        ttk.Checkbutton(row, text='Webcam (nhập 0, 1…)', variable=self.camera).pack(side='left', padx=10)
        settings = ttk.Frame(top)
        settings.pack(fill='x', pady=10)
        self.combo(settings, 'Mô hình', self.model, ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt'], 16)
        self.combo(settings, 'Đối tượng', self.category, list(CLASSES), 14)
        self.combo(settings, 'Vạch', self.axis, ['Ngang', 'Dọc'], 8)
        self.combo(settings, 'Ảnh inference', self.imgsz, ['320', '480', '640', '960'], 7)
        self.combo(settings, 'Thiết bị', self.device, ['cpu', '0'], 6)
        ttk.Label(settings, text='Conf:').pack(side='left')
        ttk.Entry(settings, textvariable=self.confidence, width=6).pack(side='left')
        line = ttk.Frame(top)
        line.pack(fill='x')
        ttk.Label(line, text='Vị trí vạch (% từ trên/trái):').pack(side='left')
        ttk.Scale(line, from_=5, to=95, variable=self.position).pack(side='left', fill='x', expand=True, padx=10)
        self.position_label = ttk.Label(line, width=5)
        self.position_label.pack(side='left')
        self.position.trace_add('write', lambda *_: self.position_label.config(text=f'{self.position.get():.0f}%'))
        self.position_label.config(text='50%')
        actions = ttk.Frame(top)
        actions.pack(fill='x', pady=(10, 0))
        self.start_button = ttk.Button(actions, text='▶ Bắt đầu', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(actions, text='■ Dừng', command=self.stop.set, state='disabled')
        self.stop_button.pack(side='left', padx=8)
        ttk.Button(actions, text='Biểu đồ CSV', command=self.chart).pack(side='left')
        ttk.Button(actions, text='Mở kết quả', command=self.open_folder).pack(side='left', padx=8)
        body = ttk.Panedwindow(root, orient='horizontal')
        body.pack(fill='both', expand=True, padx=12)
        self.preview = ttk.Label(body, text='Video sẽ hiển thị tại đây', anchor='center', background='#152238', foreground='white')
        body.add(self.preview, weight=4)
        side = ttk.Frame(body, padding=10)
        body.add(side, weight=1)
        self.total = ttk.Label(side, text='TỔNG LƯỢT: 0', font=('Segoe UI', 17, 'bold'))
        self.total.pack(anchor='w', pady=10)
        self.table = ttk.Treeview(side, columns=('class', 'direction', 'count'), show='headings', height=12)
        for col, title, width in [('class', 'Loại', 90), ('direction', 'Chiều', 65), ('count', 'Lượt', 50)]:
            self.table.heading(col, text=title)
            self.table.column(col, width=width, stretch=True)
        self.table.pack(fill='both', expand=True)
        ttk.Label(side, text='down/up: xuống/lên\nright/left: phải/trái\n\nMỗi ID được đếm tối đa\nmột lần cho mỗi chiều.\n\nCài đặt áp dụng ở lần\nBắt đầu tiếp theo.', justify='left').pack(anchor='w', pady=12)
        ttk.Label(root, textvariable=self.status, padding=12, wraplength=1100).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(60, self.poll)

    @staticmethod
    def combo(parent, label, variable, values, width):
        ttk.Label(parent, text=label + ':').pack(side='left', padx=(0, 4))
        ttk.Combobox(parent, textvariable=variable, values=values, width=width, state='readonly').pack(side='left', padx=(0, 10))

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv *.webm'), ('Tất cả', '*.*')])
        if path:
            self.source.set(path)
            self.camera.set(False)

    def publish(self, data):
        # A bounded queue prevents a slow preview from accumulating video frames.
        try:
            self.events.put_nowait(data)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(data)

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            source = self.source.get().strip()
            if self.camera.get():
                source = source or '0'
                if int(source) < 0:
                    raise ValueError('Chỉ số webcam phải >= 0.')
            elif not Path(source).is_file():
                raise ValueError('Hãy chọn một file video hợp lệ.')
            conf = float(self.confidence.get())
            if not 0 < conf <= 1:
                raise ValueError('Confidence phải lớn hơn 0 và không quá 1.')
            config = dict(source=source, camera=self.camera.get(), model=self.model.get(), category=self.category.get(), axis='horizontal' if self.axis.get() == 'Ngang' else 'vertical', position=self.position.get()/100, confidence=conf, imgsz=int(self.imgsz.get()), device=self.device.get())
        except ValueError as exc:
            messagebox.showerror('Cài đặt chưa hợp lệ', str(exc))
            return
        self.stop.clear()
        self.folder = None
        self.table.delete(*self.table.get_children())
        self.total.config(text='TỔNG LƯỢT: 0')
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status.set('Đang khởi động…')
        self.worker = threading.Thread(target=self.work, args=(config,), daemon=True)
        self.worker.start()

    def work(self, config):
        try:
            final = run(config, self.stop, self.publish)
            self.publish({**final, 'done': True, 'status': 'Đã dừng.' if self.stop.is_set() else 'Đã xử lý hết video.'})
        except Exception as exc:
            self.publish({'done': True, 'error': str(exc), 'status': 'Xử lý thất bại.'})

    def poll(self):
        try:
            while True:
                data = self.events.get_nowait()
                if 'folder' in data:
                    self.folder = Path(data['folder'])
                if 'status' in data:
                    self.status.set(data['status'])
                if 'counts' in data:
                    counts = data['counts']
                    self.total.config(text=f'TỔNG LƯỢT: {sum(counts.values())}')
                    self.table.delete(*self.table.get_children())
                    for (label, direction), count in sorted(counts.items()):
                        self.table.insert('', 'end', values=(label, direction, count))
                if 'frame' in data:
                    from PIL import Image, ImageTk
                    picture = Image.fromarray(data['frame'])
                    picture.thumbnail((max(self.preview.winfo_width(), 100), max(self.preview.winfo_height(), 100)))
                    self.photo = ImageTk.PhotoImage(picture)
                    self.preview.config(image=self.photo, text='')
                if data.get('done'):
                    self.start_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                if data.get('error') and not self.closing:
                    messagebox.showerror('Lỗi xử lý', data['error'])
        except queue.Empty:
            pass
        if self.closing and not (self.worker and self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(60, self.poll)

    def open_folder(self):
        folder = self.folder or Path(__file__).parent / 'outputs'
        folder.mkdir(exist_ok=True)
        os.startfile(str(folder))

    def chart(self):
        path = filedialog.askopenfilename(initialdir=str(self.folder or Path(__file__).parent), title='Chọn events.csv', filetypes=[('CSV', '*.csv')])
        if not path:
            return
        try:
            from collections import Counter
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            with open(path, encoding='utf-8-sig', newline='') as file:
                reader = csv.DictReader(file)
                if not {'class', 'direction', 'source_seconds'}.issubset(reader.fieldnames or []):
                    raise ValueError('Hãy chọn events.csv, không phải summary.csv hay performance.csv.')
                rows = list(reader)
            totals = Counter((r['class'], r['direction']) for r in rows)
            times = sorted(float(r['source_seconds']) for r in rows)
            window = tk.Toplevel(self.root)
            window.title('Thống kê lượt qua vạch')
            figure = Figure(figsize=(10, 5), tight_layout=True)
            left, right = figure.subplots(1, 2)
            left.bar([f'{c}\n{d}' for c, d in totals], list(totals.values()), color='#14b8a6')
            left.set(title='Luot theo loai va chieu', ylabel='So luot')
            right.step([0] + times, [0] + list(range(1, len(times)+1)), where='post', color='#2563eb')
            right.set(title='Tong luot tich luy', xlabel='Thoi gian nguon (giay)', ylabel='So luot')
            canvas = FigureCanvasTkAgg(figure, master=window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            NavigationToolbar2Tk(canvas, window)
            ttk.Button(window, text='Lưu PNG', command=lambda: self.save_chart(figure)).pack(pady=5)
        except Exception as exc:
            messagebox.showerror('Không mở được biểu đồ', str(exc))

    @staticmethod
    def save_chart(figure):
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            figure.savefig(path, dpi=150)

    def close(self):
        self.closing = True
        self.stop.set()
        self.status.set('Đang chờ tác vụ kết thúc và lưu CSV…')


def main():
    root = tk.Tk()
    ttk.Style().theme_use('clam')
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
