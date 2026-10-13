# Hướng dẫn cho agent

- Đọc `PRODUCT.md`, `DESIGN.md` và `DEVELOPMENT.md` trước khi mở rộng sản phẩm.
- Giữ giao diện và thông báo người dùng bằng tiếng Việt.
- Kiểm tra `git status` trước khi sửa và giữ nguyên thay đổi có sẵn.
- Không hiển thị preview, metadata model hoặc detection giả. Chỉ bật hành động khi engine thực hiện được thật.
- Preview Exact Center Crop luôn giữ frame thật được giải mã từ video và vẽ khung vùng crop đúng tọa độ/kích thước lên frame; không thay frame bằng ảnh crop khi người dùng xem preview.
- Gán nhãn thủ công hoạt động độc lập với inference; không tạo hoặc gợi ý kết quả AI giả. Khi chỉnh sửa annotation, yêu cầu dự án đã lưu và bảo toàn tọa độ pixel của ảnh.
- Dataset chỉ xuất ảnh có box hợp lệ hoặc ảnh trống đã được review; giữ cùng video/đoạn và ảnh trùng byte trong cùng split, ghi nhãn YOLO chuẩn hóa và manifest provenance vào phiên bản mới.
- Video người dùng chủ yếu ở Full HD (1920×1080); vẫn đọc kích thước media thật, không hard-code độ phân giải hoặc tự resize crop chính xác.
- App mở thẳng workspace. Không khôi phục màn chào bắt buộc kiểm tra engine; health check chạy nền và chỉ hiện trạng thái gọn trên toolbar.
- Giữ cầu nối Tauri an toàn: allowlist command, xác thực đường dẫn/argument và gọi tiến trình trực tiếp; không chạy lệnh qua shell.
- Giao thức engine là JSON-lines UTF-8: mỗi request và response chiếm một dòng.
- Chạy kiểm thử phù hợp sau thay đổi; không ghi kết quả kiểm thử chưa chạy.
- Sau mỗi lượt hoàn thành, cập nhật `README.md`, `AGENTS.md`, `agent.md` và `DEVELOPMENT.md`; đồng bộ tài liệu sản phẩm/thiết kế và changelog khi có thay đổi liên quan.
- Không khẳng định hỗ trợ Windows trước khi build/test Windows; DirectML và RX 6600 còn chờ máy Windows.
