# Nhật ký thay đổi

## Chưa phát hành — Workspace trích xuất

- Bỏ qua detection box chạm sát biên crop vì object có thể bị cắt hoặc lớn hơn khung; báo số box bị bỏ qua và không tạo negative candidate nếu tất cả box đều bị bỏ qua.
- Làm rõ dHash ngay dưới thanh trượt: giá trị 5 là số bit khác tối đa giữa hai hash 64-bit, không phải số ảnh; tổng kết gallery báo số crop gần giống thực tế bị lọc.
- Thêm Champion/Challenger: so sánh evaluation trên cùng fixed validation và điều kiện, kiểm tra checksum/target/regression toàn cục và theo class, lưu lịch sử compare/promote/rollback; promote cần xác nhận. Test engine dùng fake runtime, chưa xác minh bằng checkpoint thật.
- Thêm đánh giá YOLO11n trên fixed validation/test: chỉ dùng checkpoint nội bộ đã xác minh; kiểm tra manifest, hash ảnh/nhãn, class schema và chống leakage; lưu metric tổng/per-class, mục tiêu người dùng, điều kiện, cảnh báo và lịch sử. Có thể huỷ giữa các batch mà không lưu kết quả dở dang. Latency/FPS/throughput/RAM tiếp tục hiện `not_measured` đến khi có benchmark deployment.
- Thêm dataset lineage: đóng băng val/test sau review, tạo phiên bản con xác minh parent và checksum, giữ nguyên ảnh/label evaluation, replay dữ liệu train theo `all`/`balanced`/`new`; truyền lineage vào training queue/run.
- Giữ giới hạn video đầu vào tối đa FHD 1920×1080; nguồn lớn hơn bị từ chối ở import, preview và extraction, không có lựa chọn độ phân giải video cao hơn và không tự scale.
- Benchmark một batch gồm bốn bản sao fixture H.264 FHD một giờ: xử lý tổng cộng 4 giờ nguồn/432.000 frame, xuất 14.400 crop trong 519,16 giây, không lỗi; RSS đỉnh 892,3 MiB. Đây là fixture tổng hợp, không phải kiểm thử nhiều codec hoặc video thực.
- Thêm inference toàn project: nạp model một lần, chạy theo batch/chunk có giới hạn, kiểm fingerprint khi lưu và giữ phần đã hoàn tất khi hủy.
- Thêm candidate negative kèm model/checkpoint provenance và fingerprint ảnh; chỉ export sau khi người dùng xác nhận ảnh trống. Nâng SQLite project lên schema v4, migrate từ v1/v2/v3.
- Mở rộng training preflight cho dataset trên 20.000 ảnh và cấu hình trên 300 epoch mỗi imgsz; ghi rõ Patience = 0 để tắt early stopping. Test 20.003 ảnh/301 epoch dùng fake trainer, chưa xác nhận hiệu năng train YOLO thật.
- Nối inference vào tab Gán nhãn cho ảnh đang mở: chọn checkpoint YOLO11n nội bộ, lưu gợi ý AI kèm model ID/checkpoint SHA-256/confidence, duyệt/chỉnh sửa/loại; chỉ gợi ý đã duyệt vào YOLO export mặc định. Có thể chạy lại cùng checkpoint sau khi loại toàn bộ gợi ý cũ trên ảnh.
- Lưu provenance annotation AI qua project SQLite và migrate schema v1/v2/v3 lên v4 khi ghi; autosave giữ nguyên gợi ý AI, ảnh còn gợi ý chờ không được dùng làm negative.
- Thêm engine YOLO11n inference theo batch cho ảnh crop; chỉ nạp checkpoint nội bộ đã xác minh, kiểm tra class schema và trả box pixel/confidence từ Ultralytics; hỗ trợ hủy giữa các batch.
- Khóa fine-tune vào checkpoint YOLO11n detection nội bộ đã đăng ký, còn trong thư mục run và khớp SHA-256; từ chối checkpoint `.pt` bất kỳ hoặc được phát hiện từ nguồn ngoài.
- Giới hạn đầu vào video tối đa 1920×1080; từ chối video lớn hơn ở import, preview và extraction, không tự scale; hiển thị lý do từ chối.
- Đo một lượt extraction trên fixture H.264 FHD dài một giờ: 3.600 crop JPG 320×320 trong 132,27 giây, không lỗi; RSS cao nhất đo được 807,3 MiB. Điều kiện phép đo được ghi trong DEVELOPMENT.md.
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
- Dùng viền cam liền nét cho khung Exact Center Crop để vùng W×H dễ nhận ra trên frame nguồn.
- Xác nhận bằng UI rằng preset vuông đổi đúng kích thước overlay trên cùng frame nguồn; video 1920×1080 với crop 320×320 cho tọa độ x=800, y=380.
- Giữ frame video nguồn làm preview và vẽ ô vuông đúng kích thước crop tại tâm; bỏ màn chào buộc kiểm tra engine, app mở thẳng workspace.
- Thêm tab Huấn luyện YOLO11n: kiểm tra dataset/dependency, queue nhiều độ phân giải, progress/metrics thật, dừng an toàn theo epoch, registry và fine-tune từ checkpoint đã lưu.
- Thêm Smart Folder Discovery: quét có giới hạn/progress/hủy, đọc dataset YOLO và run Ultralytics, tái dựng metric CSV, nhận diện checkpoint/model export mà không nạp `.pt`; lưu run ngoài vào registry theo yêu cầu người dùng.
- Đóng gói đầy đủ các module Python được engine import, gồm Smart Folder Discovery, session và training; sửa lỗi app bundle thiếu `frameforge.discovery`.

## 0.1.0 — Nền tảng

- Khởi tạo ứng dụng desktop Tauri 2 với React, TypeScript và Tailwind CSS.
- Thêm Python engine chạy cục bộ theo JSON-lines và health check.
- Thêm cấu hình CSP, capability Tauri và tài liệu khởi đầu.
