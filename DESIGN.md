# Thiết kế

## Hướng hình ảnh

Workspace desktop tối, gọn và thiên về thao tác: nền than chì nhiều lớp, separator mảnh, chữ xám sáng, xanh dương dành cho lựa chọn và hành động chính. Màu xanh lá/vàng/đỏ chỉ báo trạng thái. Bố cục và mật độ lấy từ screenshot FrameForge người dùng cung cấp.

## Bố cục

- Thanh công cụ trên cùng: dự án mới/mở/lưu, kiểm tra ONNX và trạng thái engine.
- Tabs quy trình: Trích xuất khả dụng; Gán nhãn và Xuất dữ liệu bị khóa đến khi có chức năng thật.
- Cột trái: import, kéo thả và chọn nguồn media; drag overlay phản hồi khi người dùng kéo tệp vào cửa sổ.
- Trung tâm: preview frame thật, độ phân giải, điều khiển frame/timeline.
- Cột phải: sampling, filter, crop, kích thước, định dạng và thư mục đầu ra.
- Gallery ngang ở dưới: thumbnail của ảnh thật vừa trích; trạng thái tổng kết ở thanh cuối.

Ba vùng làm việc giữ đồng thời trên màn hình. Splitter giữa source/preview/settings và trên gallery kéo được bằng chuột hoặc phím mũi tên; kích thước lưu ở local storage. Ở cửa sổ hẹp hơn 1240px, panel co theo CSS trong khi nội dung settings/gallery cuộn độc lập. Cửa sổ Tauri khởi tạo ở 1440×960, tối thiểu 1024×700.

## Tương tác và tiếp cận

- Chỉ số media do OpenCV đọc từ đường dẫn người dùng chọn. Không có hình demo hay detection giả.
- Dialog native chọn nguồn, dự án, model và thư mục đầu ra.
- Các điều khiển có nhãn, focus ring nhìn thấy được, disabled/loading/error/empty states và hỗ trợ `prefers-reduced-motion`.
- Màu không phải tín hiệu duy nhất; trạng thái có chữ và icon.
- Timeline seek chạy qua tiến trình nền; request cũ không được cập nhật preview sau request mới.
