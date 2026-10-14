# FrameForge

<div align="center">
  <p><strong>Tạo dataset object detection từ video và ảnh — chạy cục bộ trên máy.</strong></p>
  <p>Tauri 2 · React · TypeScript · Python · OpenCV · YOLO11n</p>
</div>

FrameForge đưa quy trình từ media nguồn tới dataset và model vào một ứng dụng desktop: trích crop, gán nhãn, kiểm tra/xuất YOLO và huấn luyện YOLO11n.

## Quy trình chính

| Bước | Khả năng |
|---|---|
| Trích xuất | Preview frame thật, Exact Center Crop theo pixel, lọc blur và ảnh gần trùng bằng dHash. |
| Gán nhãn | Vẽ và review bounding box; lưu nhãn cùng dự án SQLite `.ffproj`. |
| Dataset | Kiểm tra dữ liệu, chia theo video/đoạn, xuất YOLO có provenance, negative sampling và lineage. |
| Model | Train/fine-tune YOLO11n, suy luận có xác minh checkpoint, đánh giá trên fixed set và quản lý Champion/Challenger. |

## Giới hạn và hành vi

- **Video đầu vào tối đa FHD 1920×1080.** Video lớn hơn bị từ chối, không scale; video nhỏ hơn giữ nguyên độ phân giải. `imgsz` là kích thước riêng của bước training.
- **Lọc ảnh gần trùng:** dHash là dấu vân tay cảm nhận 64-bit. Mặc định chỉ loại crop lệch tối đa **1 bit** so với một trong 4.096 crop được giữ gần nhất; ngưỡng cao hơn lọc mạnh hơn. Đây là ngưỡng giống nhau, không phải số ảnh cần giữ. Gallery báo số crop thực tế bị loại.
- **Negative sampling** mặc định nhắm tỷ lệ tối đa 10% trên dataset cuối và cũng bỏ ảnh gần trùng (dHash ≤1). Negative thuộc fixed validation/test luôn được giữ; tỷ lệ thực tế có thể vượt quota.
- Checkpoint `.pt` ngoài cần xác nhận tin cậy, kiểm tra YOLO11n detection, class schema và checksum trước khi dùng. Gợi ý AI cần review trước khi vào dataset.

## Bắt đầu

Cần Node.js, pnpm, Rust stable/toolchain desktop và Python 3.10+. Từ thư mục repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e "engine[dev]"
pnpm install
pnpm tauri dev
```

Để bật training, cài thêm `.venv/bin/python -m pip install -e "engine[train]"`. Có thể đặt `FRAMEFORGE_PYTHON` để chọn interpreter khác. Trên Windows, dùng `.venv\Scripts\python.exe`; hỗ trợ Windows/DirectML/RX 6600 chưa được xác minh.

## Phát triển

```sh
pnpm test:engine
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo check --manifest-path src-tauri/Cargo.toml
pnpm tauri build
```

UI dùng React/TypeScript; Tauri gọi Python engine qua JSON-lines. Python runtime và dependencies cần cài riêng, standalone sidecar chưa có.

## Trạng thái

Media, gán nhãn, dataset, inference và workflow YOLO11n đã được triển khai. Training quy mô lớn chưa được xác minh bằng dataset/model thực; benchmark hiện có dùng fixture H.264 FHD tổng hợp lặp lại: 14.400 crop từ 4 giờ nguồn trong 519 giây, RSS đỉnh khoảng 892 MiB. Chưa có deployment benchmark/export ONNX, TensorRT, OpenVINO hoặc resume training. Xem [DEVELOPMENT.md](DEVELOPMENT.md) để biết phạm vi kiểm chứng và giới hạn.

## Giấy phép

Mã nguồn FrameForge được cấp phép theo GNU AGPL phiên bản 3 trở lên; xem [LICENSE](LICENSE). Ứng dụng tích hợp Ultralytics trong workflow huấn luyện và inference tùy chọn. Trước khi phân phối bản thương mại/đóng nguồn hoặc model YOLO, hãy kiểm tra [điều khoản cấp phép hiện hành của Ultralytics](https://www.ultralytics.com/license); hướng dẫn của họ nêu AGPL-3.0 hoặc giấy phép Enterprise cho các trường hợp sử dụng tương ứng. Dependency bên thứ ba vẫn giữ giấy phép riêng.

## Tài liệu

[Phạm vi sản phẩm](PRODUCT.md) · [Thiết kế](DESIGN.md) · [Trạng thái phát triển](DEVELOPMENT.md) · [Hướng dẫn agent](AGENTS.md)
