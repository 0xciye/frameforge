# Sản phẩm

## Mục tiêu

FrameForge là ứng dụng desktop local-first giúp người dùng tạo dữ liệu object detection từ video/ảnh, gán nhãn, huấn luyện YOLO11n, fine-tune checkpoint tương thích, so sánh benchmark và export model triển khai. Mọi preview, metadata, nhãn, metric và file đầu ra phải đến từ dữ liệu hoặc phép tính thật.

## Đã triển khai

- Workspace desktop Tauri 2, React và TypeScript bằng tiếng Việt.
- Nhập video/ảnh bằng dialog, kéo thả hoặc quét thư mục; đọc metadata và preview frame thật. Image I/O hỗ trợ đường dẫn Unicode trên Windows; checksum label được tính trên đúng byte đã ghi. Chỉ nhận video tối đa 1920×1080; video lớn hơn bị từ chối thay vì scale.
- Điều hướng frame, lấy mẫu theo số frame hoặc thời gian, lọc near-duplicate bằng dHash và độ mờ bằng variance of Laplacian. dHash là dấu vân tay cảm nhận 64-bit; mặc định ngưỡng Hamming ≤1 chỉ bỏ crop gần như trùng, so với tối đa 4.096 crop được giữ gần nhất. Ngưỡng cao hơn lọc mạnh hơn; gallery báo số crop thực tế bị lọc.
- Exact Center Crop cắt đúng vùng W×H từ tâm frame nguồn, không resize/padding vùng crop; preview phủ overlay đúng pixel. Video lớn hơn 1920×1080 bị từ chối. Full-frame letterbox và grid vẫn khả dụng; kích thước crop/output tối đa 1920×1080.
- Lọc blur và near-duplicate đo vùng crop nguồn; provenance gồm source identity, frame/timestamp, crop rectangle, output size và settings.
- Lưu/mở dự án SQLite `.ffproj`, hỗ trợ nâng schema v1/v2/v3 lên v4.
- Gán nhãn box thủ công với class ổn định, review, undo/redo và autosave vào dự án đã lưu.
- Kiểm tra dataset, loại ảnh trống chưa review, chia theo video/đoạn với chống rò rỉ ảnh trùng byte, và export YOLO vào thư mục phiên bản mới.
- Quét thư mục dự án ở chế độ chỉ đọc để tìm dataset YOLO, run Ultralytics, metric CSV, checkpoint, model export và project FrameForge; giới hạn độ sâu/số file, bỏ symlink, báo progress và hỗ trợ hủy.
- Nhập thư mục làm việc đã chuyển từ máy khác: đối chiếu lại đường dẫn trong `data.yaml` với tệp thật bên trong thư mục đã chọn, kể cả khi YAML nằm ngoài dataset; chỉ tự nối path cũ khi duy nhất một thư mục khớp mọi split. Ghi cấu hình local riêng và chuẩn hóa file list trong App Data mà không sửa hoặc sao chép ảnh/nhãn. Preflight xác minh nguồn còn tồn tại, không nằm ngoài scope và không đi qua symlink. Checkpoint phát hiện vẫn cần trust flow riêng.
- Nạp ONNX Runtime CPU để xem tensor/metadata model; chưa suy luận.
- Huấn luyện YOLO11n trên CPU hoặc backend GPU mà PyTorch báo khả dụng; kiểm tra dataset trước train, queue nhiều imgsz độc lập, theo dõi progress/metrics thật, dừng tại ranh giới epoch và lưu registry/checkpoint. Không chọn checkpoint fine-tune thì tạo run mới từ YOLO11n pretrained; chọn checkpoint đã xác minh thì điền sẵn `imgsz` từ checkpoint hoặc `args.yaml` cùng run. Nếu cả hai thiếu metadata, hiển thị rõ và dùng mặc định Ultralytics 640; registry cũ không ghi nguồn `imgsz` phải được đánh dấu chưa xác định. Fine-tune nhận checkpoint YOLO11n detection đã đăng ký, còn checksum hợp lệ và class schema khớp dataset. Có thể nhập `.pt` tương thích sau xác nhận tin cậy; FrameForge giới hạn 2 GiB, xác minh rồi nạp bản sao trong vùng riêng, kiểm tra task/kiến trúc/class và xác minh checksum lần nữa trước khi đăng ký cùng provenance. Tệp được quét từ thư mục ngoài vẫn chưa tin cậy cho đến thao tác nhập này.
- Engine inference nhận crop images, xác minh checkpoint và class schema, rồi trả box pixel, class ID và confidence từ Ultralytics. Có thể chọn checkpoint YOLO11n đã huấn luyện hoặc đã nhập qua trust flow. Box chạm sát biên crop bị bỏ qua vì object có thể bị cắt hoặc lớn hơn khung; nếu toàn bộ box bị bỏ qua thì không tạo negative candidate. Tab Gán nhãn chạy inference cho ảnh hiện tại hoặc cả project: nạp model một lần, xử lý batch/chunk có giới hạn, kiểm fingerprint trước khi lưu và giữ các chunk đã hoàn tất khi hủy. Box AI còn lại được lưu ở trạng thái `needs_review`; ảnh thực sự không có detection trở thành candidate negative có provenance/fingerprint và chỉ được export sau khi người dùng xác nhận ảnh trống.
- Dataset export chỉ đưa box thủ công đã review, gợi ý AI đã duyệt và candidate negative đã xác nhận vào YOLO labels. Ảnh còn box thủ công chưa review hoặc gợi ý/candidate AI đang chờ hay bị loại sẽ không được export.
- Negative sampling mặc định giữ tối đa 10% ảnh negative đã review, có preset 0/5/10/25% và tùy chỉnh 0–100%; seed giúp lấy mẫu tái lập, dHash ≤1 loại ảnh gần như trùng. Negative trong fixed validation/test của parent được giữ nguyên và report cảnh báo nếu vượt quota.
- Dataset có thể đóng băng validation/test khi toàn bộ ảnh đã review. Phiên bản con xác minh manifest/checksum parent, giữ nguyên ảnh và labels của hai split cố định, chia dữ liệu mới theo nhóm và hỗ trợ replay train `all`, `balanced` hoặc `new`; manifest và training metadata ghi lineage.
- Training preflight xử lý dataset lớn theo luồng, giữ `cache=False`, hỗ trợ hơn 20.000 ảnh và cấu hình hơn 300 epoch cho mỗi imgsz. Fixture tổng hợp 20.003 ảnh cùng hai job 301 epoch đã được kiểm tra bằng fake trainer; đây là kiểm tra cấu hình, không phải benchmark hoặc train thật ở quy mô đó. Patience = 0 tắt early stopping.
- Đánh giá YOLO11n chỉ trên fixed validation/test của manifest FrameForge và checkpoint nội bộ có SHA-256 hợp lệ. Lưu metric tổng/per-class, checksum, dataset/split, cấu hình inference, thiết bị/framework, cảnh báo về độ tin cậy và lịch sử. Mục tiêu để trống mặc định, hỗ trợ theo class; latency, FPS, throughput và memory hiển thị chưa đo tới khi có benchmark triển khai.
- Champion/Challenger dùng validation cố định để so sánh cùng tập và điều kiện; xác minh checksum, target chất lượng và regression tổng/per-class. Promote do người dùng xác nhận, lưu lịch sử và hỗ trợ rollback. Engine đã qua fake-runtime tests; kiểm tra bằng checkpoint thật trên máy này còn thiếu.
- Lưu metadata run đã quét vào registry chỉ sau thao tác của người dùng; checkpoint ngoài vẫn chưa xác minh và không được đưa vào fine-tune/inference cho đến khi nhập riêng.
- Tự lưu snapshot workspace cục bộ và tự khôi phục khi mở app mà không hiện hộp thoại. Nếu dự án gốc mất hoặc checksum thay đổi, mở snapshot như dự án chưa lưu để không ghi đè dữ liệu gốc.

## Chưa khả dụng

- Resume training sau khi tiến trình bị gián đoạn.
- Benchmark latency/FPS/throughput/memory theo runtime triển khai, experiment comparison và export ONNX/TensorRT/OpenVINO.
- Soak test extraction trên nhiều giờ video FHD thực vẫn chưa có. Đã chạy một batch tổng cộng bốn giờ nguồn bằng bốn bản sao của cùng fixture tổng hợp một giờ; chưa đại diện nội dung và codec thực.
- Train YOLO thật trên hơn 20.000 ảnh trong hơn 300 epoch; hiện chỉ xác nhận plumbing bằng fake trainer và smoke nhỏ.
- Đóng gói Python engine thành sidecar standalone.
- Production release v1.0 chưa sẵn sàng: cài đặt engine/dependencies trên máy sạch, ký/notarize macOS, ký Windows MSI và xác minh Windows end-to-end còn chờ. Giữ phiên bản tiền phát hành 0.1.0 cho tới khi đóng các cổng này.

Các hành động chưa triển khai phải khóa rõ trong giao diện. Không thay bằng demo detection, metric hoặc training progress giả.

## Hướng trải nghiệm

Người dùng nhập media → xác nhận Exact Center Crop → lọc ảnh → gán nhãn thủ công hoặc hỗ trợ bởi inference → quản lý và chia dataset → huấn luyện YOLO11n → fine-tune checkpoint → benchmark → export model. Mọi bước chạy cục bộ theo mặc định; remote training là tùy chọn sau này.

## Nguyên tắc

- Xử lý cục bộ theo mặc định, không upload video/model.
- Không sửa video gốc hoặc ghi đè im lặng file đã tồn tại.
- Phân biệt rõ chức năng đã triển khai, đã kiểm thử và còn chờ Windows.
- Không tuyên bố hỗ trợ mọi ONNX model; cần adapter cụ thể cho từng tensor layout.
