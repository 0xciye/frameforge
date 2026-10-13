# Hướng dẫn cho agent

- Đọc `PRODUCT.md`, `DESIGN.md` và `DEVELOPMENT.md` trước khi mở rộng sản phẩm.
- Giữ giao diện và thông báo người dùng bằng tiếng Việt.
- Không hiển thị dữ liệu phát hiện giả. Chỉ thêm khả năng khi engine thực hiện được thật.
- Giữ cầu nối Tauri an toàn: chỉ thêm lệnh cần thiết, kiểm tra dữ liệu và không chạy lệnh qua shell.
- Giao thức engine là JSON-lines UTF-8: mỗi request và response chiếm một dòng.
- Ưu tiên thay đổi nhỏ phù hợp giai đoạn; cập nhật tài liệu khi thay đổi phạm vi hoặc cách chạy.
