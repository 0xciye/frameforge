# FrameForge

Dọn output/cache có thể tạo lại bằng `node scripts/python-runner.mjs scripts/cleanup_builds.py`. Lệnh giữ portable mới nhất trong `src-tauri/target/release/FrameForge`, source, model, dataset và các `.venv*`. Chỉ dùng `--keep-installed` khi đã kiểm tra có bản đầy đủ tại `Desktop/FrameForge-Portable`.

## Cập nhật lấy frame và crop theo vật thể (2026-10-26)

- Chọn model đã nhập, bật lọc bằng model và chọn **Crop theo vật thể (YOLO11n)**. Detect frame nguồn một lần rồi crop W × H giữ pixel; chuyển box sang crop, không detect lại. Bỏ vật thể quá lớn và gộp vùng crop trùng tọa độ.
- Bật **Hiện detection trên preview** và chọn model trong **Lọc bằng model**: app tự vẽ box/class/confidence và vùng crop trực tiếp lên ảnh khi đổi frame, tua hoặc phát video. **Cập nhật box** chỉ dùng để chạy lại frame hiện tại.
- Zoom tab Gán nhãn hiển thị mượt khi phóng to, không ép thành ô pixel vuông; giữ ảnh PNG đủ độ phân giải và tọa độ nhãn gốc.
- **Lấy frame** hiển thị độ nét toàn frame nguồn; bộ lọc mờ khi xuất đo trên crop. Xác nhận trùng bằng nội dung mặc định bật, kiểm tra màu/chi tiết cục bộ/vật thể trong tối đa 4.096 ảnh đã ghi của lượt hiện tại.
- Play chờ giải mã, dùng lại decoder, seek khi tua và phát lại từ đầu khi bấm ở cuối. Preview theo frame không phát âm thanh.

<div align="center">
  <p><strong>Tạo dataset object detection từ video và ảnh — chạy cục bộ trên máy.</strong></p>
  <p>Tauri 2 · React · TypeScript · Python · OpenCV · YOLO11n</p>
</div>

FrameForge tạo dataset object detection bằng một pipeline duy nhất: **nhập model → nhập ảnh/video → model gắn nhãn → người dùng duyệt/gán lại → lưu dataset**. FrameForge không huấn luyện hay đánh giá model. Bản portable mở trực tiếp cửa sổ FrameForge với giao diện được đóng gói bên trong app; không cần mở trình duyệt hoặc chạy server localhost.

Build portable Windows thành công sẽ tự dọn bản backup và output/cache trung gian, chỉ giữ bản portable mới nhất tại `src-tauri/target/release/FrameForge`. Sau khi cập nhật và kiểm chứng bản Desktop, chạy `node scripts/python-runner.mjs scripts/cleanup_builds.py --keep-installed` để chỉ giữ bản tại `Desktop/FrameForge-Portable`. Lần build sau phải biên dịch lại do cache build đã được dọn.

## Quy trình chính

| Bước | Khả năng |
|---|---|
| 1. Nhập model | Trong tab Gán nhãn (kể cả khi chưa có ảnh), “Nhập model .pt” nhập checkpoint YOLO11n detection sau xác nhận tin cậy; FrameForge kiểm tra kiến trúc, class và checksum. Dự án chưa có nhãn có thể dùng class của model. |
| 2. Nhập ảnh/video | Tab Trích xuất: preview frame thật, Exact Center Crop theo pixel, lọc blur và ảnh gần trùng bằng dHash. Mục **Lọc bằng model** dùng checkpoint YOLO11n đã nhập để bỏ crop không có vật thể và lưu box tìm được thành gợi ý AI chờ duyệt ngay khi trích xuất (cần dự án đã lưu). |
| 3. Model gắn nhãn | Nút **Auto label** mở tab Gán nhãn; chọn ngưỡng confidence rồi chạy gợi ý cho ảnh hiện tại hoặc cả project. Box AI ở trạng thái chờ duyệt. |
| 4. Duyệt/gán lại | Duyệt hoặc loại từng gợi ý hoặc tất cả gợi ý của ảnh, vẽ/sửa box thủ công, xác nhận ảnh trống. Nút **Nhập ảnh** trên toolbar thêm một hoặc nhiều thư mục ảnh có sẵn (quét cả thư mục con) vào dự án đã lưu, ảnh dùng tại chỗ; ảnh đã có JSON vào dự án cùng nhãn của nó. Chỉ khi có JSON của công cụ khác app mới hỏi đánh dấu ảnh có box là đã review. Nhãn lưu trong file JSON LabelMe/X-AnyLabeling cạnh từng ảnh (`anh.png` → `anh.json`, user request 2026-10-25); `.ffproj` chỉ giữ danh sách ảnh, cài đặt và bản sao nhãn để truy vấn nhanh. FrameForge ghi JSON sau mỗi thay đổi nhãn và đọc lại khi mở dự án, kiểm tra hoặc xuất dataset, chạy auto label; file bị sửa bên ngoài (so SHA-256 với lần ghi/đọc cuối) thắng. `shapes` chỉ chứa nhãn thật (box thủ công, box AI đã duyệt); gợi ý AI chờ duyệt/bị loại, trạng thái review và candidate âm tính nằm trong khóa `frameforge`, `checked` = ảnh đã review. Shape khác loại hoặc nhãn ngoài class dự án được giữ nguyên trong file. Phím tắt class: `1`–`9` theo thứ tự class ID và chữ cái đầu tên class khi không trùng (mặc định `1`/`H` = head, `2`/`B` = body); đang chọn box thì đổi class box đó, không thì chọn class để vẽ box tiếp theo. Danh sách ảnh lọc được theo Tất cả / Chưa xong / Đã review / Có gợi ý AI chờ duyệt / Có box của từng class; phím `A`/`D` (hoặc PageUp/PageDown) ảnh trước/sau trong danh sách đang lọc, `N` ảnh chưa xong kế tiếp, `Enter` đánh dấu đã review rồi sang ảnh chưa xong kế tiếp; chữ `A`, `D`, `N` không dùng làm phím class. Trích xuất video không cần chọn thư mục khi dự án đã lưu: mặc định ghi vào `<tên dự án>_frames` cạnh file `.ffproj` (engine tạo thư mục; allowlist Rust nhận thư mục chưa tồn tại nếu thư mục cha có sẵn), vẫn đổi được bằng nút chọn thư mục. |
| 5. Lưu dataset | Tab Dataset kiểm tra dữ liệu, mặc định chia theo số ảnh (ảnh trùng byte và frame gần trùng luôn cùng split, không bỏ ảnh nào), và xuất YOLO có provenance cùng negative sampling. Có thể chọn chia theo video/đoạn. |

## Giới hạn và hành vi

- Bản portable Desktop vừa build: `C:\Users\CiyE\Desktop\FrameForge-Portable\frameforge.exe`; shortcut FrameForge cũ trên Desktop trỏ tới bản này. Giữ `engine-runtime` cạnh executable; đóng cửa sổ app cũ rồi mở lại shortcut để dùng bản mới.
- **Trích xuất chỉ ghi PNG không mất dữ liệu** (yêu cầu 2026-10-26); dự án cũ chọn JPG cũng ghi PNG cho lượt mới. Ảnh JPG cũ vẫn đọc được. PNG giữ pixel của frame giải mã, không sửa được chi tiết đã mất do nén/chuyển động trong video.
- **Ưu tiên dữ liệu nét:** mặc định chọn frame rõ chi tiết nhất trong mỗi khoảng lấy mẫu và bật lọc mờ chặt; kiểm tra thêm vùng mục tiêu khi có model. Chọn frame trong cả khoảng tốn thêm thời gian giải mã. Lọc trùng mới xác nhận màu, nội dung và mục tiêu sau dHash; chỉ ảnh thực sự đã ghi mới ảnh hưởng lượt lọc sau. Ngưỡng mới mặc định 6 bit, giữ ngưỡng đã lưu của dự án cũ.
- **Số frame không bị mất:** video Radiant có 99.917 frame. Mỗi 10 frame chỉ chọn khoảng 9.992 frame, 89.925 frame không được chọn. Ví dụ cũ: 1.011 xuất + 430 gần giống + 1.794 mờ + 6.757 không vật thể = 9.992. Gallery hiện số frame nguồn/lấy mẫu/bỏ theo khoảng, tệp cũ giữ nguyên và phần chưa xử lý/ngoài giới hạn.
- Tên nguồn có dấu chấm (ví dụ `SnapYT.App.mp4`) giữ nguyên trong tên ảnh kèm số frame/tile, tránh trùng tên. Lấy mẫu chỉ chuyển frame được chọn sang BGR; benchmark 120 crop JPG 416×416, mỗi 30 frame trên video Radiant FHD 60 FPS: trung bình 14,41 → 4,20 giây (nhanh khoảng 3,44 lần, chưa bật lọc/model).
- **Video đầu vào tối đa FHD 1920×1080.** Video lớn hơn bị từ chối, không scale; video nhỏ hơn giữ nguyên độ phân giải.
- **Lọc bằng model:** có công tắc bật/tắt riêng (tắt vẫn nhớ model), nút xóa model đang chọn và công tắc GPU/CPU. Xóa chỉ bỏ bản sao model trong app, giữ checkpoint gốc và nhãn đã gán. GPU được yêu cầu mà engine không có GPU sẽ báo lỗi, không âm thầm chạy CPU. Runtime RX 6600 trên máy này dùng ROCm cộng đồng, MIOpen tắt để tránh NaN.
- **Lọc ảnh gần trùng:** dHash là dấu vân tay cảm nhận 64-bit. Mặc định chỉ loại crop lệch tối đa **1 bit** so với một trong 4.096 crop được giữ gần nhất; ngưỡng cao hơn lọc mạnh hơn. Đây là ngưỡng giống nhau, không phải số ảnh cần giữ. Gallery báo số crop thực tế bị loại.
- **Negative sampling** mặc định nhắm tỷ lệ tối đa 10% trên dataset cuối và cũng bỏ ảnh gần trùng (dHash ≤1).
- Checkpoint `.pt` ngoài cần xác nhận tin cậy, kiểm tra YOLO11n detection, class schema và checksum trước khi dùng. Gợi ý AI cần review trước khi vào dataset.
- Checkpoint không ghi `imgsz` sẽ đọc `args.yaml` cùng run; nếu cả hai thiếu metadata, dùng mặc định 640 và báo rõ.

## Model dùng để gợi ý box

FrameForge chỉ gán box: thủ công, hoặc bằng checkpoint YOLO11n nhập qua “Nhập model .pt”. Việc train, tối ưu và xuất model nằm ở công cụ riêng `C:\Users\CiyE\Desktop\yolo-train` (README riêng, không thuộc repo này). Model để gán nhãn là teacher 640 `yolo-train\models\teacher\best.pt`; model chạy trong SysX là gói ONNX FP16 256/320 trong `yolo-train\models\deploy\`. Công cụ đó dùng môi trường `.venv-train` trong thư mục repo này (đã nằm trong `.gitignore`), nên đừng xóa `.venv-train`.

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
