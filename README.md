# FrameForge

Ứng dụng desktop mã nguồn mở để chuẩn bị video và dữ liệu gán nhãn. Phase 1 thiết lập nền tảng Tauri 2, React/TypeScript và Python engine chạy cục bộ. Hiện ứng dụng có health check thật qua giao thức JSON-lines; các tính năng xử lý video và gán nhãn chưa được triển khai.

## Yêu cầu

- Node.js tương thích với Vite 8 và pnpm
- Rust stable và các công cụ build desktop của hệ điều hành
- Python 3

## Chạy ứng dụng

```sh
pnpm install
pnpm tauri dev
```

Nếu Python không nằm trong `PATH`, đặt `FRAMEFORGE_PYTHON` thành đường dẫn tới trình thông dịch Python trước khi chạy ứng dụng.

## Kiểm tra và build

```sh
pnpm test:engine
pnpm build
pnpm tauri build
```

`pnpm test:engine` chạy kiểm thử giao thức bằng thư viện `unittest` có sẵn của Python. `pnpm build` kiểm tra TypeScript và build frontend. `pnpm tauri build` tạo gói desktop cho hệ điều hành hiện tại.

## Cấu trúc

- `src/` — giao diện React/TypeScript
- `src-tauri/` — ứng dụng desktop và lệnh Rust
- `engine/` — tiến trình Python và kiểm thử giao thức
- `PRODUCT.md`, `DESIGN.md`, `DEVELOPMENT.md` — phạm vi sản phẩm, thiết kế và hướng dẫn phát triển

Lệnh `engine_health` khởi chạy Python engine, gửi một dòng JSON, rồi xác thực phản hồi JSON-lines. Engine chỉ hỗ trợ lệnh `health` ở giai đoạn này và không giả lập kết quả nhận diện.
