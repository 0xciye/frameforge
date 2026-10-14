# Thiết kế

## Hướng hình ảnh

Workspace desktop tối, gọn và thiên về thao tác: nền than chì nhiều lớp, separator mảnh, chữ xám sáng, xanh dương dành cho lựa chọn và hành động chính. Màu xanh lá/vàng/đỏ chỉ báo trạng thái. Bố cục và mật độ lấy từ screenshot FrameForge người dùng cung cấp.

## Bố cục

- Thanh công cụ trên cùng: dự án mới/mở/lưu, kiểm tra ONNX và trạng thái engine.
- Tabs quy trình: Trích xuất, Gán nhãn, Dataset và Huấn luyện. Inference được gọi ngay trong tab Gán nhãn cho ảnh đang mở.
- Cột trái: import, kéo thả và chọn nguồn media; drag overlay phản hồi khi người dùng kéo tệp vào cửa sổ.
- Trung tâm: preview frame thật đến tối đa 1920×1080, điều khiển frame/timeline; video lớn hơn FHD bị từ chối.
- Cột phải: sampling, filter, chế độ crop, kích thước crop/output, định dạng và thư mục đầu ra.
- Gallery ngang ở dưới: thumbnail của ảnh thật vừa trích; trạng thái tổng kết ở thanh cuối.

## Dataset

- Tab Dataset cho phép chọn split theo video nguồn hoặc đoạn thời gian, đặt tỷ lệ và seed, xem số ảnh/box/class cùng cảnh báo trước khi xuất.
- Điều khiển negative sampling có preset 0/5/10/25% và tỷ lệ tùy chỉnh; report nêu số ảnh negative được giữ/bỏ cùng tỷ lệ thực tế. Giữ nguyên negative trong fixed validation/test của parent.
- Ảnh trống chưa review không được export; candidate negative do inference tạo ra chỉ thành negative đã review sau khi người dùng xác nhận ảnh trống. Negative đã xác nhận có nhãn TXT rỗng. Ảnh được sao chép nguyên pixel, còn labels chuẩn hóa theo kích thước nguồn.
- Box AI dùng nét đứt và nhãn confidence để phân biệt với nhãn thủ công. Mỗi gợi ý có thao tác Duyệt/Loại; chỉnh sửa box AI cũng xác nhận gợi ý. Gợi ý chưa duyệt/bị loại không vào export mặc định; ảnh chỉ có gợi ý chờ review không được tính negative.
- Chọn checkpoint nội bộ đã xác minh hoặc nhập `.pt` YOLO11n tương thích qua xác nhận tin cậy; danh sách hiển thị tên file đã nhập và chỉ dùng model khớp class schema. Inference áp dụng vào ảnh đang mở hoặc toàn bộ project theo batch có giới hạn. Hiển thị trạng thái chạy/dừng, lỗi, kết quả và thao tác làm mới registry. Hủy giữ các chunk đã lưu; ảnh thay đổi trong lúc chạy bị bỏ qua khi commit.
- Mỗi lần export tạo thư mục phiên bản mới với `data.yaml` và manifest; thao tác không sửa dự án hay ảnh nguồn.
- Có thể đóng băng val/test khi cả hai split đều có ảnh và mọi ảnh đã được review. Khi tạo child version, chọn parent manifest và replay policy train (`all`, `balanced`, `new`); ảnh và label val/test được giữ nguyên, còn nhóm của chúng không đi vào train. Report hiển thị ảnh mới, replay và fixed-set counts.

## Lọc frame gần giống

- Thanh dHash biểu thị khoảng cách Hamming tối đa giữa hash cảm nhận 64-bit. Mặc định 1 nghĩa là chỉ bỏ crop gần như trùng (khoảng cách ≤1) so với một trong tối đa 4.096 crop đã giữ gần nhất; ngưỡng cao hơn lọc mạnh hơn, không phải số lượng crop cần xóa.
- Engine chỉ so sánh với tối đa 4.096 hash của crop được giữ gần nhất. Sau khi extraction xong, gallery hiển thị số crop gần giống thực tế đã bị lọc.

## Huấn luyện

- Tab Huấn luyện kiểm tra dependency, thiết bị khả dụng và dataset local trước khi bật queue. Mỗi imgsz chạy thành run riêng từ cùng checkpoint nền; fine-tune chỉ bắt đầu từ checkpoint YOLO11n đã đăng ký, được người dùng chọn, còn checksum hợp lệ và khớp class schema. Model nhập hiển thị tên file trong lựa chọn checkpoint.
- Progress, loss, validation metrics, runtime và khả năng dừng được lấy từ callback training thật. Checkpoint, config và log lưu trong thư mục output; model pretrained cache trong app data. Giữ `cache=False` cho dataset lớn; nhập Patience = 0 để tắt early stopping và chạy đủ epoch đã đặt.
- Card “Đánh giá model trên fixed set” chọn checkpoint registry, validation/test đã đóng băng, `imgsz`, batch, confidence, NMS IoU và thiết bị. Chỉ bật khi split hiện tại có fixed-set ID. Đánh giá có thể hủy giữa các batch; tác vụ dở dang không được ghi thành kết quả.
- Hiển thị Precision/Recall/F1/mAP50/mAP50–95 tổng thể và theo class cùng mục tiêu người dùng đặt. Để trống là không đặt ngưỡng. Kèm dataset/fixed-set/checkpoint fingerprint, thiết bị/framework và cảnh báo tập nhỏ/thiếu class; latency/FPS/throughput/RAM ghi “Chưa đo” cho đến khi có benchmark triển khai.
- Hiển thị Champion hiện tại, Challenger mới nhất, bảng delta metric, blockers/target và lịch sử. Chỉ dùng fixed validation có cùng fingerprint/schema/điều kiện; mặc định không chấp nhận regression. Promote và rollback đều cần xác nhận rõ, model cũ được giữ trong registry.

## Quét thư mục dự án

- Chọn một thư mục cha để quét cấu trúc, dataset YAML, cấu hình run, metric CSV, checkpoint và model export; hiện progress, giới hạn quét và nút hủy.
- Kết quả run ngoài là metadata chỉ đọc. Lưu vào registry cần thao tác riêng; checkpoint `.pt` chưa được nạp hoặc xác minh tương thích nên không xuất hiện làm model fine-tune.

Ba vùng làm việc giữ đồng thời trên màn hình. Splitter giữa source/preview/settings và trên gallery kéo được bằng chuột hoặc phím mũi tên; kích thước lưu ở local storage. Ở cửa sổ hẹp hơn 1240px, panel co theo CSS trong khi nội dung settings/gallery cuộn độc lập. Cửa sổ Tauri khởi tạo ở 1440×960, tối thiểu 1024×700.

## Exact Center Crop

- Đây là chế độ crop mặc định, dùng hình chữ nhật W×H cố định ở tâm frame video nguồn tối đa 1920×1080. Từ chối video lớn hơn FHD; không scale video. Tọa độ bắt đầu dùng `floor((frame - crop) / 2)` và vùng lấy theo half-open slicing, không resize hoặc padding vùng crop.
- Preview giữ frame nguồn đã giải mã và vẽ viền cam liền nét tại tâm theo đúng W×H pixel, co giãn cùng frame khi preview đổi kích thước. Ảnh crop đầu ra được xem trong gallery sau khi trích xuất.
- Preset vuông 192, 256, 320, 416, 512 và 640 chỉ là lối tắt; hai trường W/H vẫn nhận kích thước nguyên tùy chỉnh độc lập.
- Crop/output nhận kích thước nguyên tối đa rộng 1920, cao 1080 px; crop lớn hơn frame bị báo lỗi trước khi trích. Letterbox/Grid có cùng giới hạn tối đa. Blur và near-duplicate ở chế độ center đo trên crop của frame video nguồn.
- Full Frame/Letterbox và Grid là thuật toán riêng; chỉ các mode đó mới resize hoặc padding.

## Tương tác và tiếp cận

- Chỉ số media do OpenCV đọc từ đường dẫn người dùng chọn. Không có hình demo hay detection giả.
- Dialog native chọn nguồn, dự án, model và thư mục đầu ra.
- Các điều khiển có nhãn, focus ring nhìn thấy được, disabled/loading/error/empty states và hỗ trợ `prefers-reduced-motion`.
- Màu không phải tín hiệu duy nhất; trạng thái có chữ và icon.
- Timeline seek chạy qua tiến trình nền; request cũ không được cập nhật preview sau request mới.
