# Nhật ký thay đổi

## Chưa phát hành — Workspace trích xuất

- Thêm tab Dataset: xác thực ảnh/nhãn/class, loại ảnh trống chưa review, chia theo video hoặc đoạn thời gian, giữ ảnh byte-identical cùng split, thống kê class và cảnh báo mất cân bằng.
- Thêm YOLO export phiên bản mới: sao chép ảnh không resize, ghi label chuẩn hóa, nhãn rỗng cho reviewed negative, `data.yaml` và manifest integrity/provenance.
- Thêm Exact Center Crop làm chế độ mặc định: crop đúng W×H tại tâm frame, preview giữ frame nguồn và phủ khung vùng crop; không resize hoặc padding.
- Thêm preset 192/256/320/416/512/640 cùng kích thước tùy chỉnh, kiểm tra crop vượt biên và lưu provenance crop trong dự án SQLite schema v2.
- Tính lọc blur và near-duplicate trên chính crop nguồn; giữ nguyên các mode Letterbox và Grid.
- Bỏ màn chào buộc kiểm tra Python engine; app mở thẳng workspace và tự kiểm tra engine ở trạng thái nhỏ trên toolbar.
- Thêm import media/thư mục, đọc metadata/preview, timeline, sampling theo frame/giây và progress/hủy.
- Thêm lọc dHash/Laplacian, crop letterbox/grid, cỡ ảnh, màu viền, JPG/PNG và lựa chọn xử lý filename trùng.
- Thêm tạo/mở/lưu dự án SQLite `.ffproj` và ngăn lưu đè tệp không phải dự án FrameForge.
- Thêm kiểm tra metadata model ONNX bằng ONNX Runtime CPU; chưa chạy inference.
- Thêm thumbnails thật cho gallery; cập nhật trạng thái roadmap và hướng dẫn cài Python.
- Thêm trình gán nhãn thủ công: Head (0), Body (1), thao tác box, undo/redo, review và autosave vào dự án `.ffproj`.
- Giữ kiểm tra Python engine chạy nền trên toolbar; không hiện lại màn chào buộc người dùng bấm kiểm tra mỗi lần mở app.
- Bỏ chế độ chuyển preview sang ảnh crop; khung crop hiển thị trực tiếp trên frame video, viền nằm trong hình học vùng crop.
- Đổi khung crop preview sang viền cam nét đứt để vùng W×H dễ phân biệt trên frame nguồn.

## 0.1.0 — Nền tảng

- Khởi tạo ứng dụng desktop Tauri 2 với React, TypeScript và Tailwind CSS.
- Thêm Python engine chạy cục bộ theo JSON-lines và health check.
- Thêm cấu hình CSP, capability Tauri và tài liệu khởi đầu.
