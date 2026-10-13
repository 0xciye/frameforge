# Sản phẩm

## Mục tiêu

FrameForge là ứng dụng desktop local-first giúp người dùng tạo dữ liệu object detection từ video/ảnh, gán nhãn, huấn luyện YOLO11n, fine-tune checkpoint tương thích, so sánh benchmark và export model triển khai. Mọi preview, metadata, nhãn, metric và file đầu ra phải đến từ dữ liệu hoặc phép tính thật.

## Đã triển khai

- Workspace desktop Tauri 2, React và TypeScript bằng tiếng Việt.
- Nhập video/ảnh bằng dialog, kéo thả hoặc quét thư mục; đọc metadata và preview frame thật.
- Điều hướng frame, lấy mẫu theo số frame hoặc thời gian, lọc near-duplicate bằng dHash và độ mờ bằng variance of Laplacian.
- Exact Center Crop cắt đúng vùng W×H từ tâm frame không resize/padding; preview giữ frame nguồn và phủ overlay vùng crop đúng pixel; full-frame letterbox và grid vẫn khả dụng.
- Lọc blur và near-duplicate đo vùng crop nguồn; provenance gồm source identity, frame/timestamp, crop rectangle, output size và settings.
- Lưu/mở dự án SQLite `.ffproj`, hỗ trợ nâng schema hiện có từ v1 lên v2.
- Nạp ONNX Runtime CPU để xem tensor/metadata model; chưa suy luận.

## Chưa khả dụng

- Adapter YOLO và inference.
- Canvas gán/sửa bounding box, review workflow, undo/redo.
- Quản lý/kiểm tra dataset và export YOLO.
- YOLO11n training trên MPS/CPU, hàng đợi resolution, checkpoint discovery và fine-tuning.
- Benchmark/experiment tracking và export ONNX/TensorRT/OpenVINO.
- Đóng gói Python engine thành sidecar standalone.

Các hành động chưa triển khai phải khóa rõ trong giao diện. Không thay bằng demo detection, metric hoặc training progress giả.

## Hướng trải nghiệm

Người dùng nhập media → xác nhận Exact Center Crop → lọc ảnh → gán nhãn thủ công hoặc hỗ trợ bởi inference → quản lý và chia dataset → huấn luyện YOLO11n → fine-tune checkpoint → benchmark → export model. Mọi bước chạy cục bộ theo mặc định; remote training là tùy chọn sau này.

## Nguyên tắc

- Xử lý cục bộ theo mặc định, không upload video/model.
- Không sửa video gốc hoặc ghi đè im lặng file đã tồn tại.
- Phân biệt rõ chức năng đã triển khai, đã kiểm thử và còn chờ Windows.
- Không tuyên bố hỗ trợ mọi ONNX model; cần adapter cụ thể cho từng tensor layout.
