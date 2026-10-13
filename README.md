# FrameForge

FrameForge là ứng dụng desktop local-first để trích xuất ảnh từ video và chuẩn bị dữ liệu cho quy trình gán nhãn. Workspace tiếng Việt hỗ trợ nhập video/ảnh bằng dialog hoặc kéo thả, xem metadata và preview frame thật, lấy mẫu theo frame hoặc thời gian, lọc frame gần giống và ảnh mờ, crop letterbox hoặc grid, rồi lưu ảnh JPG/PNG cùng dự án SQLite `.ffproj`.

Ứng dụng có thể nạp một tệp ONNX bằng ONNX Runtime CPU để đọc thông tin tensor và metadata lớp. Đây mới là bước kiểm tra model; **inference, sinh/chỉnh sửa nhãn và xuất YOLO chưa được triển khai**. FrameForge không huấn luyện hoặc fine-tune model, và không tạo detection giả.

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

Đã triển khai media inspection/preview, trích frame có progress và hủy, lọc dHash/Laplacian, crop letterbox/grid, SQLite create/save/load và kiểm tra metadata ONNX bằng CPU. Hiện các tác vụ này có kiểm thử Python; inference YOLO, biên tập annotation, export dataset, đóng gói Python sidecar và kiểm thử Windows vẫn còn ở roadmap.
