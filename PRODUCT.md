# Sản phẩm

## Mục tiêu

FrameForge là ứng dụng desktop local-first giúp người làm dataset đi từ video/ảnh đến dữ liệu gán nhãn YOLO. Sản phẩm không huấn luyện model. Mọi preview, metadata, nhãn và file đầu ra phải đến từ dữ liệu thật của người dùng.

## Đã triển khai

- Workspace desktop Tauri 2, React và TypeScript bằng tiếng Việt.
- Nhập video/ảnh bằng dialog, kéo thả hoặc quét thư mục; đọc metadata và preview frame thật.
- Điều hướng frame, lấy mẫu theo số frame hoặc thời gian, lọc near-duplicate bằng dHash và độ mờ bằng variance of Laplacian.
- Crop letterbox/grid, chọn kích thước, màu viền, JPG/PNG, xử lý tên file trùng, progress và hủy.
- Lưu/mở dự án SQLite `.ffproj` gồm nguồn, cài đặt, ảnh và annotation schema.
- Nạp ONNX Runtime CPU để xem tensor/metadata model; chưa suy luận.

## Chưa khả dụng

- Adapter YOLO, inference, bounding box thật, class mapping đầy đủ.
- Canvas sửa nhãn, review workflow, undo/redo.
- Export YOLO và kiểm tra tương ứng image/label.
- Đóng gói Python engine thành sidecar standalone.

Các hành động chưa triển khai được khóa rõ trong giao diện. Không thay bằng demo detection hoặc dữ liệu giả.

## Hướng trải nghiệm

Người dùng nhập media → xác nhận preview và thông số thật → thiết lập sampling/filter/crop → trích ảnh → nạp model đã huấn luyện → chạy inference → rà soát/chỉnh box → lưu và xuất dataset. Workspace giữ nguồn, preview, thiết lập và gallery trong cùng cửa sổ.

## Nguyên tắc

- Xử lý cục bộ theo mặc định, không upload video/model.
- Không sửa video gốc hoặc ghi đè im lặng file đã tồn tại.
- Phân biệt rõ chức năng đã triển khai, đã kiểm thử và còn chờ Windows.
- Không tuyên bố hỗ trợ mọi ONNX model; cần adapter cụ thể cho từng tensor layout.
