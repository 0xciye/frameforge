# Thiết kế

## Preview detection và chất lượng frame — 2026-10-26

- Giữ bố cục hiện tại; bật Hiện detection và chọn model sẽ tự vẽ box khi xem/tua/phát video. Nút Cập nhật box chạy lại frame hiện tại; có trạng thái bận/lỗi/thiếu model và số vật thể/vùng crop.
- Box xanh viền 2 px có class/confidence, crop cam nét đứt có số thứ tự; tọa độ theo nguồn và co cùng ảnh. Giữ box cùng ảnh đang hiển thị trong lúc xử lý frame kế tiếp, thay cả hai đồng thời; đổi model/nguồn xóa box cũ.
- Zoom tab Gán nhãn dùng nội suy mượt (`image-rendering: auto`) ở mọi tỷ lệ; không thay ảnh nguồn, geometry SVG hoặc tọa độ nhãn.
- Crop theo vật thể nằm cạnh Center/Letterbox/Grid; mô tả detect trước rồi crop, giữ pixel và bỏ object lớn hơn khung. Lấy frame có độ nét nguồn và công tắc xác nhận trùng bằng nội dung.

## Runtime desktop

- Bản portable nạp frontend đã build trong `frontendDist` qua Tauri custom protocol; chỉ `tauri dev` mới trỏ tới Vite `devUrl`/localhost. Người dùng mở executable/app portable trực tiếp, không khởi chạy server web riêng.

## Hướng hình ảnh

Workspace desktop tối, gọn và thiên về thao tác: nền than chì nhiều lớp, separator mảnh, chữ xám sáng, xanh dương dành cho lựa chọn và hành động chính. Màu xanh lá/vàng/đỏ chỉ báo trạng thái. Bố cục và mật độ lấy từ screenshot FrameForge người dùng cung cấp.

## Bố cục

- Thanh công cụ trên cùng: dự án mới/mở/lưu, Auto label, Xuất YOLO và trạng thái engine.
- Tabs quy trình theo pipeline: Trích xuất (nhập ảnh/video), Gán nhãn (nhập model, gợi ý AI, duyệt/gán lại) và Dataset (lưu dataset). Không có tab huấn luyện. Nút Auto label trên toolbar mở tab Gán nhãn; inference ảnh hiện tại hoặc cả project được chạy từ khu vực AI trong tab này.
- Cột trái: import, kéo thả và chọn nguồn media; drag overlay phản hồi khi người dùng kéo tệp vào cửa sổ.
- Trung tâm: preview frame thật đến tối đa 1920×1080, điều khiển frame/timeline; video lớn hơn FHD bị từ chối.
- Cột phải: sampling, filter, **Lọc bằng model** (chọn model khớp class hoặc nút "Nhập .pt", ngưỡng confidence %, mặc định 35, công tắc "Giữ cả crop không có vật thể"), chế độ crop, kích thước crop/output, định dạng và thư mục đầu ra. Gallery báo số crop không có vật thể đã bỏ và số box AI chờ duyệt.
- Mục Lọc bằng model có công tắc "Sử dụng model để lọc" độc lập với model đang chọn, nút "Xóa model đang chọn" dùng xác nhận hiện có; tắt lọc không quên lựa chọn model. Khi bật lọc, công tắc "Sử dụng GPU" mặc định bật và hiển thị tên GPU thực tế hoặc trạng thái chưa khả dụng. Điều khiển model bị khóa trong khi trích xuất/nhập/xóa; thiết lập lọc chi tiết ẩn khi tắt lọc.
- Lấy frame có "Chọn frame nét nhất mỗi khoảng" với ghi chú thời gian giải mã; lọc mờ có mức chặt cho dữ liệu train. Định dạng hiển thị PNG không mất dữ liệu cố định, bỏ chất lượng JPEG. Summary gallery xuống dòng, không che số đếm; hiển thị nguồn/lấy mẫu/không chọn/chưa xử lý bên cạnh các lý do loại crop.
- Gallery ngang ở dưới: hiển thị mọi ảnh trong dự án (mới nhất trước), chỉ render vùng đang cuộn tới và tải thumbnail thật theo lô; ảnh đã review có ghi chú. Trạng thái tổng kết ở thanh cuối.

## Dataset

- Tab Dataset mặc định chia theo từng ảnh đủ điều kiện sau negative sampling; tỷ lệ tính theo số ảnh, giữ ảnh trùng byte và frame gần trùng cùng split. Có thể chọn theo video nguồn hoặc đoạn thời gian, đặt tỷ lệ và seed, xem số ảnh/box/class cùng cảnh báo trước khi xuất.
- Điều khiển negative sampling có preset 0/5/10/25% và tỷ lệ tùy chỉnh; report nêu số ảnh negative được giữ/bỏ cùng tỷ lệ thực tế.
- Ảnh trống chưa review không được export; candidate negative do inference tạo ra chỉ thành negative đã review sau khi người dùng xác nhận ảnh trống. Negative đã xác nhận có nhãn TXT rỗng. Ảnh được sao chép nguyên pixel, còn labels chuẩn hóa theo kích thước nguồn.
- Box AI dùng nét đứt và nhãn confidence để phân biệt với nhãn thủ công. Mỗi gợi ý có thao tác Duyệt/Loại; chỉnh sửa box AI cũng xác nhận gợi ý. Gợi ý chưa duyệt/bị loại không vào export mặc định; ảnh chỉ có gợi ý chờ review không được tính negative.
- Nút “Nhập model .pt” trong khu vực Gợi ý AI nhập checkpoint YOLO11n tương thích qua xác nhận tin cậy; khi dự án chưa có ảnh, tab Gán nhãn vẫn hiện panel Model để nhập trước. Danh sách hiển thị mọi checkpoint hợp lệ, đánh dấu “class khác dự án”; khi dự án chưa có nhãn có nút “Dùng class của model cho dự án”. Trường “Ngưỡng confidence (%)” 1–99, mặc định 25. Danh sách box có “Duyệt tất cả (n)” và “Loại tất cả” cho gợi ý đang chờ; danh sách hiển thị tên file đã nhập và chỉ dùng model khớp class schema. Inference áp dụng vào ảnh đang mở hoặc toàn bộ project theo batch có giới hạn. Hiển thị trạng thái chạy/dừng, lỗi, kết quả và thao tác làm mới registry. Hủy giữ các chunk đã lưu; ảnh thay đổi trong lúc chạy bị bỏ qua khi commit.
- Mỗi lần export tạo thư mục phiên bản mới với `data.yaml` và manifest; thao tác không sửa dự án hay ảnh nguồn.

## Lọc frame gần giống

- Thanh dHash biểu thị khoảng cách Hamming tối đa giữa hash cảm nhận 64-bit. Mặc định 1 nghĩa là chỉ bỏ crop gần như trùng (khoảng cách ≤1) so với một trong tối đa 4.096 crop đã giữ gần nhất; ngưỡng cao hơn lọc mạnh hơn, không phải số lượng crop cần xóa.
- Engine chỉ so sánh với tối đa 4.096 hash của crop được giữ gần nhất. Sau khi extraction xong, gallery hiển thị số crop gần giống thực tế đã bị lọc.

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
