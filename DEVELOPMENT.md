# Phát triển

## Công cụ

- Node.js và pnpm để chạy Vite/React
- Rust stable và toolchain native tương ứng để build Tauri
- Python 3 để chạy engine

Ứng dụng ưu tiên Python `python3` trên macOS/Linux và `python` trên Windows. Đặt `FRAMEFORGE_PYTHON` nếu cần dùng một interpreter ở đường dẫn khác.

## Lệnh thường dùng

```sh
pnpm install
pnpm tauri dev
pnpm build
pnpm test:engine
pnpm tauri build
```

Tauri chạy Vite tại cổng `1420`. Không đổi cổng nếu chưa cập nhật đồng thời cấu hình Tauri và Vite.

## Cầu nối engine

Rust mở `engine/frameforge_engine.py` trực tiếp dưới dạng tiến trình con, gửi JSON-lines qua stdin và đọc phản hồi từ stdout. Chẩn đoán của engine phải ghi vào stderr để không làm hỏng giao thức. Script được đóng gói cùng ứng dụng; interpreter Python là dependency ở Phase 1.

Request hiện hỗ trợ:

```json
{"id":"health","command":"health"}
```

Response thành công có `id`, `ok: true` và `data` gồm trạng thái engine, phiên bản Python và danh sách khả năng. Request sai hoặc command chưa hỗ trợ trả `ok: false` cùng mã lỗi và thông điệp.

## Hệ điều hành

Phase 1 được phát triển trên macOS. Các gói Windows chưa được build và xác nhận trên Windows; không tuyên bố hỗ trợ phát hành Windows cho đến khi hoàn tất xác nhận native.
