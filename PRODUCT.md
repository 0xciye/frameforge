# Sản phẩm

## Mục tiêu

FrameForge là ứng dụng desktop local-first hỗ trợ chuẩn bị video và dữ liệu gán nhãn. Người dùng cần một workspace rõ ràng, phản hồi nhanh và có thể mở rộng bằng engine Python.

## Phase 1

- Khởi chạy ứng dụng Tauri trên máy người dùng.
- Gọi tiến trình Python cục bộ từ Rust qua JSON-lines.
- Cho người dùng xác nhận engine có thể khởi chạy và phản hồi.
- Nêu rõ các khả năng chưa sẵn sàng.

## Chưa thuộc Phase 1

Nhập video, trích xuất frame, phát hiện đối tượng, quản lý lớp, gán nhãn, xuất dữ liệu và đồng bộ dự án. Không mô phỏng các thao tác này trong giao diện hiện tại.

## Nguyên tắc

- Xử lý dữ liệu cục bộ theo mặc định.
- Trạng thái sản phẩm phải phản ánh hành vi chạy thật.
- Giữ workspace gọn, tập trung vào video và thao tác gán nhãn.
