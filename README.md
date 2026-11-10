# AI Vision — Đếm người / phương tiện qua vạch

Ứng dụng desktop Python tiếng Việt dùng YOLO11 pretrained COCO và ByteTrack. Nhận video hoặc webcam, hiển thị bounding box / ID / vạch đếm, thống kê hai chiều, lưu CSV và vẽ biểu đồ có thể xuất PNG.

## Chạy trên Windows / PyCharm

Mở terminal trong thư mục dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe py.py
```

Nếu chưa có môi trường: `py -3.13 -m venv .venv`. Python 3.11–3.13 được khuyến nghị. Trong PyCharm chọn interpreter `.venv\Scripts\python.exe`, mở `py.py` và nhấn Run. Tkinter thường đi cùng bộ cài Python Windows; nếu thiếu, bật Tcl/Tk trong bộ cài Python.

Lần đầu chạy mô hình cần Internet để Ultralytics tải `yolo11n.pt` (hoặc s/m) về máy. Các lần tiếp theo có thể dùng lại trọng số đã tải. CPU chạy được; chọn thiết bị `0` chỉ khi máy có GPU NVIDIA và PyTorch hỗ trợ CUDA.

## Sử dụng

1. Chọn file video; hoặc bật Webcam và nhập chỉ số camera, mặc định 0.
2. Chọn Người, Phương tiện hoặc Tất cả. Phương tiện gồm bicycle, car, motorcycle, bus, truck (các lớp COCO).
3. Chọn vạch ngang/dọc và vị trí theo phần trăm. Bắt đầu với 50%; đặt vạch ở vùng thông thoáng mà đối tượng đi qua.
4. Nhấn Bắt đầu. Cấu hình được chụp lại tại thời điểm bắt đầu; muốn thay đổi hãy Dừng, chỉnh và chạy lại.
5. Nhấn Dừng để kết thúc và lưu kết quả. Khi đang tải mô hình hoặc inference, thao tác dừng/đóng sẽ chờ tác vụ hiện tại hoàn thành.
6. Chọn Biểu đồ CSV và mở `events.csv`. Cửa sổ biểu đồ có thống kê loại/chiều và tổng lượt tích lũy, có nút lưu PNG.

Mỗi lần chạy tạo thư mục riêng `outputs/YYYYMMDD_HHMMSS_microseconds/`:

| File | Nội dung |
|---|---|
| events.csv | Mỗi lần qua vạch: thời gian, giây trong nguồn, frame, ID, lớp, chiều |
| summary.csv | Tổng số lượt theo lớp và chiều |
| performance.csv | Mô hình, kích thước inference, thiết bị, số frame, thời gian, FPS toàn phiên, trạng thái |

`source_seconds` là `(frame-1)/FPS` với video FPS cố định; webcam dùng thời gian thực kể từ khởi động phiên. Video có FPS biến đổi có thể có sai lệch mốc thời gian. CSV dùng UTF-8 BOM để mở bằng Excel. CSV không có sự kiện chỉ chứa tiêu đề; đây là kết quả hợp lệ nếu chưa có đối tượng qua vạch.

## Quy tắc đếm

- Điểm đại diện: tâm cạnh dưới của bounding box; vạch kéo dài hết khung hình.
- Vạch ngang: down = trên xuống dưới, up = dưới lên trên. Vạch dọc: right = trái sang phải, left = phải sang trái.
- Đối tượng phải xuất hiện ở hai phía ngoài vùng đệm ±8 pixel mới được tính qua vạch. Video được thu nhỏ tối đa chiều ngang 1280 trước khi xử lý.
- Mỗi ID chỉ được đếm một lần mỗi chiều trong thời gian được theo dõi. Đây là số lượt, không phải tổng người duy nhất. ID mất dấu hơn 90 frame sẽ bị xóa; quay lại có thể được đếm tiếp.
- Mất ID, che khuất, góc quay, ánh sáng và vật thể nhỏ có thể làm sai số. Không có chức năng nhận diện danh tính người.

## Nghiên cứu và thử nghiệm đồ án

YOLO phát hiện đối tượng từng frame; ByteTrack liên kết detection thành các track ID qua thời gian. Mã gọi `model.track(..., persist=True, tracker='bytetrack.yaml')` liên tiếp trên cùng nguồn, tạo model mới khi bắt đầu phiên khác.

So sánh `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` trên cùng một video, cùng confidence, imgsz và thiết bị. Nano là điểm bắt đầu gọn nhẹ; model lớn thường cần nhiều tài nguyên hơn. Dùng `performance.csv` để báo cáo tốc độ thực đo, không dùng số FPS giả định. FPS trên màn hình đo một vòng đọc/xử lý; FPS toàn phiên bao gồm tải mô hình và khởi động nên chịu ảnh hưởng của lần tải đầu. Không giới hạn playback theo FPS gốc; video được xử lý nhanh/chậm tùy máy nhưng không chủ động bỏ frame.

Để đánh giá độ chính xác: tự ghi số lượt thực theo loại/chiều trên một video ngắn, so sánh với summary.csv; báo cáo sai số tuyệt đối và các trường hợp mất ID/đếm thiếu. Chưa có benchmark độ chính xác trên dữ liệu thực đi kèm dự án.

Tài liệu gốc:
- Ultralytics tracking: https://docs.ultralytics.com/modes/track/
- YOLO11 và bảng so sánh model: https://docs.ultralytics.com/models/yolo11/
- ByteTrack paper: https://arxiv.org/abs/2110.06864

## Kiểm tra logic đếm

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Kiểm tra bao gồm qua vạch, dao động trong vùng đệm, chống đếm lặp, hai chiều, nhiều ID, vạch dọc và ID hết hạn. Kiểm tra này không thay thế chạy inference với video/webcam thật.

Kiểm tra tích hợp tùy chọn: `.\.venv\Scripts\python.exe smoke_check.py`. Script tự tạo video 4 frame trống, chạy mô hình thật và kiểm tra CSV; cần tải trọng số lần đầu và không đo độ chính xác nhận diện.
