# Hướng dẫn cho agent

- Đọc `PRODUCT.md`, `DESIGN.md` và `DEVELOPMENT.md` trước khi mở rộng sản phẩm.
- Giữ giao diện và thông báo người dùng bằng tiếng Việt.
- Kiểm tra `git status` trước khi sửa và giữ nguyên thay đổi có sẵn.
- Không hiển thị preview, metadata model hoặc detection giả. Chỉ bật hành động khi engine thực hiện được thật.
- Giữ cầu nối Tauri an toàn: allowlist command, xác thực đường dẫn/argument và gọi tiến trình trực tiếp; không chạy lệnh qua shell.
- Giao thức engine là JSON-lines UTF-8: mỗi request và response chiếm một dòng.
- Chạy kiểm thử phù hợp sau thay đổi; không ghi kết quả kiểm thử chưa chạy.
- Ưu tiên thay đổi nhỏ theo roadmap; cập nhật `README.md`, `agent.md`, `DEVELOPMENT.md` và tài liệu sản phẩm/thiết kế liên quan.
- Không khẳng định hỗ trợ Windows trước khi build/test Windows; DirectML và RX 6600 còn chờ máy Windows.
