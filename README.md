# FrameForge

FrameForge là ứng dụng desktop local-first để tạo bộ dữ liệu object detection, huấn luyện YOLO11n và tiếp tục cải thiện model đã huấn luyện. Workspace hiện hỗ trợ nhập video/ảnh, preview frame thật, lấy mẫu theo frame/giây, Exact Center Crop đúng pixel với khung crop phủ trên frame nguồn, lọc blur/dHash, crop letterbox/grid, gán nhãn thủ công, kiểm tra/chia dataset và xuất YOLO từ dự án SQLite `.ffproj`.

Huấn luyện YOLO11n, fine-tuning, inference, benchmarking và export model là mục tiêu của sản phẩm nhưng **chưa triển khai**. Hiện ứng dụng chỉ đọc tensor/metadata của tệp ONNX bằng ONNX Runtime CPU; không tạo detection hay kết quả huấn luyện giả.

## Yêu cầu

- Node.js tương thích với Vite 8 và pnpm
- Rust stable cùng công cụ build desktop của hệ điều hành
- Python 3.10 trở lên

## Cài đặt môi trường Python

Từ thư mục repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e "engine[dev]"
pnpm install
```

Trên Windows, dùng `.venv\Scripts\python.exe` thay cho `.venv/bin/python`. Có thể đặt `FRAMEFORGE_PYTHON` để chọn một Python interpreter khác đã cài các dependency trong `engine/pyproject.toml`.

## Chạy ứng dụng

```sh
pnpm tauri dev
```

Vite sử dụng cổng `1420`. Tauri gọi tiến trình Python qua JSON-lines và kiểm tra đường dẫn/lệnh ở Rust trước khi chạy.

## Kiểm tra và build

```sh
pnpm test:engine
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo check --manifest-path src-tauri/Cargo.toml
pnpm tauri build
```

`pnpm test:engine` chạy pytest trên fixture media tổng hợp nhỏ. `pnpm build` kiểm tra TypeScript và build frontend. Các lệnh Tauri build trên hệ điều hành hiện tại; build thành công trên macOS không xác nhận Windows.

## Cấu trúc

- `src/` — React/TypeScript workspace
- `src-tauri/` — ứng dụng Tauri 2 và cầu nối Rust an toàn
- `engine/` — Python/OpenCV/ONNX Runtime, trích xuất và lưu dự án
- `docs/design/` — hợp đồng giao diện cho workspace trích xuất
- `PRODUCT.md`, `DESIGN.md`, `DEVELOPMENT.md` — phạm vi, hướng thiết kế và trạng thái phát triển

## Trạng thái

Đã triển khai media inspection/preview, trích frame có progress và hủy, Exact Center Crop, lọc dHash/Laplacian trên vùng crop, crop letterbox/grid, lưu provenance crop trong SQLite `.ffproj`, kiểm tra metadata ONNX bằng CPU, gán nhãn thủ công và quản lý/xuất dataset YOLO. Preview Exact Center Crop giữ frame thật lấy từ video (ví dụ nguồn FHD 1920×1080) và vẽ viền vuông W×W tại tâm; crop kích thước tùy chỉnh không vuông vẫn hiển thị theo đúng hình chữ nhật W×H. Ảnh crop thật được tạo khi trích xuất và xuất hiện trong gallery. Dataset validator loại ảnh trống chưa review, kiểm tra class/box/kích thước, chia nhóm theo video hoặc đoạn thời gian, giữ ảnh trùng byte cùng split; YOLO exporter sao chép ảnh không resize, ghi label chuẩn hóa, `data.yaml` và manifest vào thư mục phiên bản mới. Màn chào bắt buộc kiểm tra engine đã được bỏ; app mở thẳng workspace và tự kiểm tra Python engine ở trạng thái nhỏ trên toolbar. Video đầu vào chủ yếu là Full HD, nhưng app đọc kích thước thực tế và giữ tọa độ theo pixel. Inference, YOLO11n training/fine-tuning, benchmarking, đóng gói Python sidecar và xác nhận Windows vẫn còn ở roadmap.
