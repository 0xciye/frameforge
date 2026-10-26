# FrameForge

<div align="center">
  <p><strong>Tạo dataset object detection từ video và ảnh — chạy cục bộ trên máy.</strong></p>
  <p>Tauri 2 · React · TypeScript · Python · OpenCV · YOLO11n</p>
</div>

FrameForge tạo dataset object detection bằng một pipeline duy nhất: **nhập model → nhập ảnh/video → model gắn nhãn → người dùng duyệt/gán lại → lưu dataset**. FrameForge không huấn luyện hay đánh giá model. Bản portable mở trực tiếp cửa sổ FrameForge với giao diện được đóng gói bên trong app; không cần mở trình duyệt hoặc chạy server localhost.

## Quy trình chính

| Bước | Khả năng |
|---|---|
| 1. Nhập model | Trong tab Gán nhãn (kể cả khi chưa có ảnh), “Nhập model .pt” nhập checkpoint YOLO11n detection sau xác nhận tin cậy; FrameForge kiểm tra kiến trúc, class và checksum. Dự án chưa có nhãn có thể dùng class của model. |
| 2. Nhập ảnh/video | Tab Trích xuất: preview frame thật, Exact Center Crop theo pixel, lọc blur và ảnh gần trùng bằng dHash. Mục **Lọc bằng model** dùng checkpoint YOLO11n đã nhập để bỏ crop không có vật thể và lưu box tìm được thành gợi ý AI chờ duyệt ngay khi trích xuất (cần dự án đã lưu). |
| 3. Model gắn nhãn | Nút **Auto label** mở tab Gán nhãn; chọn ngưỡng confidence rồi chạy gợi ý cho ảnh hiện tại hoặc cả project. Box AI ở trạng thái chờ duyệt. |
| 4. Duyệt/gán lại | Duyệt hoặc loại từng gợi ý hoặc tất cả gợi ý của ảnh, vẽ/sửa box thủ công, xác nhận ảnh trống. Nút **Nhập ảnh** trên toolbar thêm một hoặc nhiều thư mục ảnh có sẵn (quét cả thư mục con) vào dự án đã lưu, ảnh dùng tại chỗ; ảnh đã có JSON vào dự án cùng nhãn của nó. Chỉ khi có JSON của công cụ khác app mới hỏi đánh dấu ảnh có box là đã review. Nhãn lưu trong file JSON LabelMe/X-AnyLabeling cạnh từng ảnh (`anh.png` → `anh.json`, user request 2026-10-25); `.ffproj` chỉ giữ danh sách ảnh, cài đặt và bản sao nhãn để truy vấn nhanh. FrameForge ghi JSON sau mỗi thay đổi nhãn và đọc lại khi mở dự án, kiểm tra hoặc xuất dataset, chạy auto label; file bị sửa bên ngoài (so SHA-256 với lần ghi/đọc cuối) thắng. `shapes` chỉ chứa nhãn thật (box thủ công, box AI đã duyệt); gợi ý AI chờ duyệt/bị loại, trạng thái review và candidate âm tính nằm trong khóa `frameforge`, `checked` = ảnh đã review. Shape khác loại hoặc nhãn ngoài class dự án được giữ nguyên trong file. Phím tắt class: `1`–`9` theo thứ tự class ID và chữ cái đầu tên class khi không trùng (mặc định `1`/`H` = head, `2`/`B` = body); đang chọn box thì đổi class box đó, không thì chọn class để vẽ box tiếp theo. Danh sách ảnh lọc được theo Tất cả / Chưa xong / Đã review / Có gợi ý AI chờ duyệt / Có box của từng class; phím `A`/`D` (hoặc PageUp/PageDown) ảnh trước/sau trong danh sách đang lọc, `N` ảnh chưa xong kế tiếp, `Enter` đánh dấu đã review rồi sang ảnh chưa xong kế tiếp; chữ `A`, `D`, `N` không dùng làm phím class. Trích xuất video không cần chọn thư mục khi dự án đã lưu: mặc định ghi vào `<tên dự án>_frames` cạnh file `.ffproj` (engine tạo thư mục; allowlist Rust nhận thư mục chưa tồn tại nếu thư mục cha có sẵn), vẫn đổi được bằng nút chọn thư mục. |
| 5. Lưu dataset | Tab Dataset kiểm tra dữ liệu, chia theo video/đoạn và xuất YOLO có provenance cùng negative sampling. |

## Giới hạn và hành vi

- **Video đầu vào tối đa FHD 1920×1080.** Video lớn hơn bị từ chối, không scale; video nhỏ hơn giữ nguyên độ phân giải.
- **Lọc ảnh gần trùng:** dHash là dấu vân tay cảm nhận 64-bit. Mặc định chỉ loại crop lệch tối đa **1 bit** so với một trong 4.096 crop được giữ gần nhất; ngưỡng cao hơn lọc mạnh hơn. Đây là ngưỡng giống nhau, không phải số ảnh cần giữ. Gallery báo số crop thực tế bị loại.
- **Negative sampling** mặc định nhắm tỷ lệ tối đa 10% trên dataset cuối và cũng bỏ ảnh gần trùng (dHash ≤1).
- Checkpoint `.pt` ngoài cần xác nhận tin cậy, kiểm tra YOLO11n detection, class schema và checksum trước khi dùng. Gợi ý AI cần review trước khi vào dataset.
- Checkpoint không ghi `imgsz` sẽ đọc `args.yaml` cùng run; nếu cả hai thiếu metadata, dùng mặc định 640 và báo rõ.

## Model dùng để gợi ý box

FrameForge chỉ gán box: thủ công, hoặc bằng checkpoint YOLO11n nhập qua “Nhập model .pt”. Việc train, tối ưu và xuất model nằm ở công cụ riêng `C:\Users\CiyE\Desktop\yolo-train` (README riêng, không thuộc repo này). Công cụ đó dùng môi trường `.venv-train` trong thư mục repo này (đã nằm trong `.gitignore`), nên đừng xóa `.venv-train`.

## Bắt đầu

Cần Node.js, pnpm, Rust stable/toolchain desktop và Python 3.10+. Từ thư mục repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e "engine[dev]"
pnpm install
pnpm tauri dev
```

Để chạy inference, cài thêm `.venv/bin/python -m pip install -e "engine[inference]"` (PyTorch + Ultralytics). Có thể đặt `FRAMEFORGE_PYTHON` để chọn interpreter khác. Trên Windows, dùng `.venv\Scripts\python.exe`; hỗ trợ Windows/DirectML/RX 6600 chưa được xác minh.

## Phát triển

```sh
pnpm test:engine
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo check --manifest-path src-tauri/Cargo.toml
pnpm tauri build
```

UI dùng React/TypeScript được đóng gói vào app Tauri; chỉ chế độ phát triển `pnpm tauri dev` mới chạy Vite tại localhost. Tauri gọi Python engine qua JSON-lines. Bản Windows portable nằm tại `src-tauri/target/release/FrameForge/`; mở `frameforge.exe` trong thư mục đó. Xem trạng thái và giới hạn xác minh trong [DEVELOPMENT.md](DEVELOPMENT.md).

## Trạng thái

Pipeline nhập model, trích xuất media, gợi ý AI, duyệt nhãn và xuất dataset đã được triển khai; các phần huấn luyện, đánh giá model, Champion/Challenger, lineage dataset và Smart Folder Discovery đã được gỡ. Benchmark extraction hiện có dùng fixture H.264 FHD tổng hợp lặp lại: 14.400 crop từ 4 giờ nguồn trong 519 giây, RSS đỉnh khoảng 892 MiB. Chưa click-through toàn pipeline bằng checkpoint YOLO11n thật. Xem [DEVELOPMENT.md](DEVELOPMENT.md) để biết phạm vi kiểm chứng và giới hạn.

## Tình trạng phát hành v1.0

FrameForge vẫn là bản tiền phát hành `0.1.0`, chưa được xác nhận production-ready. App macOS portable đã bundle Python, PyTorch và Ultralytics cho inference; mở trực tiếp `FrameForge.app`, không cần cài Python riêng. Trên máy Windows phát triển, bản portable mới đã build và mở được thành cửa sổ `FrameForge` mà không cần server localhost; DirectML/RX 6600 chưa được kiểm chứng. Ký/notarize, kiểm tra máy sạch, click-through inference với model/dataset thật và soak video thực vẫn là cổng phát hành. Xem [DEVELOPMENT.md](DEVELOPMENT.md).

## Giấy phép

Mã nguồn FrameForge được cấp phép theo GNU AGPL phiên bản 3 trở lên; xem [LICENSE](LICENSE). Ứng dụng tích hợp Ultralytics cho inference. Trước khi phân phối bản thương mại/đóng nguồn hoặc model YOLO, hãy kiểm tra [điều khoản cấp phép hiện hành của Ultralytics](https://www.ultralytics.com/license); hướng dẫn của họ nêu AGPL-3.0 hoặc giấy phép Enterprise cho các trường hợp sử dụng tương ứng. Dependency bên thứ ba vẫn giữ giấy phép riêng.

## Tài liệu

[Phạm vi sản phẩm](PRODUCT.md) · [Thiết kế](DESIGN.md) · [Trạng thái phát triển](DEVELOPMENT.md) · [Hướng dẫn agent](AGENTS.md)
