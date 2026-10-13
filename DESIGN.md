# Thiết kế

## Hướng hình ảnh

Workspace desktop tối, gọn và thiên về thao tác: nền than chì nhiều lớp, separator mảnh, chữ xám sáng, xanh dương dành cho lựa chọn và hành động chính. Màu xanh lá/vàng/đỏ chỉ báo trạng thái. Bố cục và mật độ lấy từ screenshot FrameForge người dùng cung cấp.

## Bố cục

- Thanh công cụ trên cùng: dự án mới/mở/lưu, kiểm tra ONNX và trạng thái engine.
- Tabs quy trình: Trích xuất, Gán nhãn và Dataset; các nút inference/training vẫn khóa đến khi có chức năng thật.
- Cột trái: import, kéo thả và chọn nguồn media; drag overlay phản hồi khi người dùng kéo tệp vào cửa sổ.
- Trung tâm: preview frame thật, độ phân giải, điều khiển frame/timeline.
- Cột phải: sampling, filter, chế độ crop, kích thước crop/output, định dạng và thư mục đầu ra.
- Gallery ngang ở dưới: thumbnail của ảnh thật vừa trích; trạng thái tổng kết ở thanh cuối.

## Dataset

- Tab Dataset cho phép chọn split theo video nguồn hoặc đoạn thời gian, đặt tỷ lệ và seed, xem số ảnh/box/class cùng cảnh báo trước khi xuất.
- Ảnh trống chưa review không được export; negative đã review có nhãn TXT rỗng. Ảnh được sao chép nguyên pixel, còn labels chuẩn hóa theo kích thước nguồn.
- Mỗi lần export tạo thư mục phiên bản mới với `data.yaml` và manifest; thao tác không sửa dự án hay ảnh nguồn.

Ba vùng làm việc giữ đồng thời trên màn hình. Splitter giữa source/preview/settings và trên gallery kéo được bằng chuột hoặc phím mũi tên; kích thước lưu ở local storage. Ở cửa sổ hẹp hơn 1240px, panel co theo CSS trong khi nội dung settings/gallery cuộn độc lập. Cửa sổ Tauri khởi tạo ở 1440×960, tối thiểu 1024×700.

## Exact Center Crop

- Đây là chế độ crop mặc định, dùng hình chữ nhật W×H cố định ở tâm từng frame đã giải mã. Tọa độ bắt đầu dùng `floor((frame - crop) / 2)` và vùng lấy theo half-open slicing; không resize hoặc padding.
- Preview luôn giữ frame nguồn thật và vẽ khung màu cam tại tâm theo đúng W×H pixel, co giãn cùng frame khi preview đổi kích thước. Ảnh crop đầu ra được xem trong gallery sau khi trích xuất.
- Preset vuông 192, 256, 320, 416, 512 và 640 chỉ là lối tắt; hai trường W/H vẫn nhận kích thước nguyên tùy chỉnh độc lập.
- Center crop nhận kích thước nguyên 1–4096 px; crop lớn hơn frame bị báo lỗi trước khi trích. Letterbox/Grid giữ giới hạn hiện có 16–4096 px. Blur và near-duplicate ở chế độ center đo trên crop nguồn.
- Full Frame/Letterbox và Grid là thuật toán riêng; chỉ các mode đó mới resize hoặc padding.

## Tương tác và tiếp cận

- Chỉ số media do OpenCV đọc từ đường dẫn người dùng chọn. Không có hình demo hay detection giả.
- Dialog native chọn nguồn, dự án, model và thư mục đầu ra.
- Các điều khiển có nhãn, focus ring nhìn thấy được, disabled/loading/error/empty states và hỗ trợ `prefers-reduced-motion`.
- Màu không phải tín hiệu duy nhất; trạng thái có chữ và icon.
- Timeline seek chạy qua tiến trình nền; request cũ không được cập nhật preview sau request mới.
