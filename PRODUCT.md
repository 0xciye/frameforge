# Sản phẩm

## Mục tiêu

FrameForge là ứng dụng desktop local-first giúp người dùng tạo dữ liệu object detection từ video/ảnh, gán nhãn, huấn luyện YOLO11n, fine-tune checkpoint tương thích, so sánh benchmark và export model triển khai. Mọi preview, metadata, nhãn, metric và file đầu ra phải đến từ dữ liệu hoặc phép tính thật.

## Đã triển khai

- Workspace desktop Tauri 2, React và TypeScript bằng tiếng Việt.
- Nhập video/ảnh bằng dialog, kéo thả hoặc quét thư mục; đọc metadata và preview frame thật. Chỉ nhận video tối đa 1920×1080; video lớn hơn bị từ chối thay vì scale.
- Điều hướng frame, lấy mẫu theo số frame hoặc thời gian, lọc near-duplicate bằng dHash và độ mờ bằng variance of Laplacian.
- Exact Center Crop cắt đúng vùng W×H từ tâm frame nguồn, không resize/padding vùng crop; preview phủ overlay đúng pixel. Video lớn hơn 1920×1080 bị từ chối. Full-frame letterbox và grid vẫn khả dụng; kích thước crop/output tối đa 1920×1080.
- Lọc blur và near-duplicate đo vùng crop nguồn; provenance gồm source identity, frame/timestamp, crop rectangle, output size và settings.
- Lưu/mở dự án SQLite `.ffproj`, hỗ trợ nâng schema v1/v2/v3 lên v4.
- Gán nhãn box thủ công với class ổn định, review, undo/redo và autosave vào dự án đã lưu.
- Kiểm tra dataset, loại ảnh trống chưa review, chia theo video/đoạn với chống rò rỉ ảnh trùng byte, và export YOLO vào thư mục phiên bản mới.
- Quét thư mục dự án ở chế độ chỉ đọc để tìm dataset YOLO, run Ultralytics, metric CSV, checkpoint, model export và project FrameForge; giới hạn độ sâu/số file, bỏ symlink, báo progress và hỗ trợ hủy.
- Nạp ONNX Runtime CPU để xem tensor/metadata model; chưa suy luận.
- Huấn luyện YOLO11n trên CPU hoặc backend GPU mà PyTorch báo khả dụng; kiểm tra dataset trước train, queue nhiều imgsz độc lập, theo dõi progress/metrics thật, dừng tại ranh giới epoch và lưu registry/checkpoint. Fine-tune chỉ nhận checkpoint YOLO11n detection nội bộ đã đăng ký và checksum còn khớp.
- Engine inference nhận crop images, xác minh checkpoint và class schema, rồi trả box pixel, class ID và confidence từ Ultralytics. Tab Gán nhãn chạy inference cho ảnh hiện tại hoặc cả project: nạp model một lần, xử lý batch/chunk có giới hạn, kiểm fingerprint trước khi lưu và giữ các chunk đã hoàn tất khi hủy. Box AI được lưu ở trạng thái `needs_review`; ảnh không có detection trở thành candidate negative có provenance/fingerprint và chỉ được export sau khi người dùng xác nhận ảnh trống.
- Dataset export chỉ đưa box thủ công, gợi ý AI đã duyệt và candidate negative đã xác nhận vào YOLO labels. Gợi ý/candidate đang chờ hoặc bị loại không được export.
- Dataset có thể đóng băng validation/test khi toàn bộ ảnh đã review. Phiên bản con xác minh manifest/checksum parent, giữ nguyên ảnh và labels của hai split cố định, chia dữ liệu mới theo nhóm và hỗ trợ replay train `all`, `balanced` hoặc `new`; manifest và training metadata ghi lineage.
- Training preflight xử lý dataset lớn theo luồng, giữ `cache=False`, hỗ trợ hơn 20.000 ảnh và cấu hình hơn 300 epoch cho mỗi imgsz. Fixture tổng hợp 20.003 ảnh cùng hai job 301 epoch đã được kiểm tra bằng fake trainer; đây là kiểm tra cấu hình, không phải benchmark hoặc train thật ở quy mô đó. Patience = 0 tắt early stopping.
- Lưu metadata run đã quét vào registry chỉ sau thao tác của người dùng; checkpoint ngoài giữ trạng thái chưa xác minh và không được đưa vào lựa chọn fine-tune.

## Chưa khả dụng

- Xác minh an toàn/khả năng tương thích và nạp checkpoint `.pt` từ nguồn ngoài.
- Tự khôi phục phiên làm việc sau khi mở lại ứng dụng.
- Đánh giá model trên fixed validation/test set, quality targets và so sánh/promote Champion/Challenger.
- Resume training sau khi tiến trình bị gián đoạn.
- Benchmark model, experiment tracking và export ONNX/TensorRT/OpenVINO.
- Train YOLO thật trên hơn 20.000 ảnh trong hơn 300 epoch; hiện chỉ xác nhận plumbing bằng fake trainer và smoke nhỏ.
- Đóng gói Python engine thành sidecar standalone.

Các hành động chưa triển khai phải khóa rõ trong giao diện. Không thay bằng demo detection, metric hoặc training progress giả.

## Hướng trải nghiệm

Người dùng nhập media → xác nhận Exact Center Crop → lọc ảnh → gán nhãn thủ công hoặc hỗ trợ bởi inference → quản lý và chia dataset → huấn luyện YOLO11n → fine-tune checkpoint → benchmark → export model. Mọi bước chạy cục bộ theo mặc định; remote training là tùy chọn sau này.

## Nguyên tắc

- Xử lý cục bộ theo mặc định, không upload video/model.
- Không sửa video gốc hoặc ghi đè im lặng file đã tồn tại.
- Phân biệt rõ chức năng đã triển khai, đã kiểm thử và còn chờ Windows.
- Không tuyên bố hỗ trợ mọi ONNX model; cần adapter cụ thể cho từng tensor layout.
