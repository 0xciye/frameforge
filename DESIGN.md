# Thiết kế

## Hướng hình ảnh

Một workspace desktop tối, ít nhiễu, lấy cảm hứng từ công cụ hậu kỳ: nền than chì, chữ xám sáng và xanh acid làm màu nhấn cho trạng thái sẵn sàng/hành động chính. Bản Phase 1 dùng cùng ngôn ngữ hình ảnh trên màn hình kiểm tra engine.

## Bố cục hiện tại

- Thanh trên cùng nhận diện FrameForge và phiên bản.
- Nội dung trung tâm giới thiệu workspace và trạng thái Python engine.
- Thanh dưới hiển thị kết nối hiện tại.
- Trên cửa sổ hẹp, hành động chuyển xuống dưới phần mô tả để tránh chật ngang.

## Thành phần

- Một nút chính duy nhất để chạy health check.
- Trạng thái được phân biệt bằng nhãn và chấm màu, không chỉ dựa vào màu.
- Kết quả thành công cho biết phiên bản Python và khả năng mà engine công bố.
- Lỗi khởi chạy hiển thị ngay trong thẻ engine.

## Tương tác và tiếp cận

- Nút có trạng thái chờ và vô hiệu hóa trong lúc kiểm tra.
- Có viền focus nhìn thấy được khi điều hướng bàn phím.
- Tôn trọng `prefers-reduced-motion`.
- Ngôn ngữ giao diện là tiếng Việt.
