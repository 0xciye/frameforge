# Nhật ký thay đổi

## Chưa phát hành — Lọc và gán nhãn bằng model khi trích xuất

- Cài đặt trích xuất có mục **Lọc bằng model**: chọn model YOLO11n đã nhập (hoặc nhập `.pt` ngay tại đó). Crop không có vật thể bị bỏ, không ghi ra đĩa; box tìm được lưu thành gợi ý AI chờ duyệt, nên tab Gán nhãn chỉ còn duyệt/sửa.
- Tùy chọn "Giữ cả crop không có vật thể" ghi mọi crop; crop trống thành ảnh trống chờ xác nhận.

## Chưa phát hành — Hiệu năng dự án lớn và zoom ảnh

- Mở/lưu dự án và chuyển ảnh nhanh hơn nhiều với hàng chục–trăm nghìn ảnh (engine chạy nền, index, đồng bộ JSON chỉ chạm file đổi, danh sách ảnh ảo hóa, phiên nhẹ).
- Zoom ảnh bằng con lăn quanh con trỏ tới 3200%, kéo ảnh bằng Space/chuột giữa, phím `+`/`-`/`0`.
- Gợi ý AI không còn hai box cùng class trên một vật (box cùng class trùng IoU ≥ 0,5 hoặc nằm trong nhau ≥ 80% chỉ giữ box confidence cao nhất); confidence mặc định 35%.
- Gợi ý lại: ảnh đã có box có thể gợi ý lại (xóa box cũ, có hỏi trước); tùy chọn gợi ý lại hàng loạt cả ảnh đã có box. Nút xóa model đã nhập. Bỏ các đoạn hướng dẫn dài trong tab Gán nhãn.
- Mở dự án chép từ máy/thư mục khác: ảnh không còn ở đường dẫn cũ được tìm lại cạnh file dự án (theo tên file và thư mục cha), hoặc trong thư mục bạn chọn; nhãn đi theo JSON cạnh ảnh.
- Xuất dataset loại cả ảnh còn gợi ý AI chờ duyệt (trước đây ảnh có box đã duyệt và box AI còn chờ vẫn được xuất với nhãn thiếu).
- Chọn box bị đè: click chọn box nhỏ nhất dưới con trỏ; click lại chỗ chồng nhau chọn lần lượt các box bên dưới; kéo vẫn di chuyển box đang chọn.
- Tab Gán nhãn: nút **Duyệt toàn bộ ảnh** cho các ảnh đang hiện theo bộ lọc: duyệt mọi box AI chờ duyệt (ghi vào JSON) và đánh dấu đã review ảnh có box; ảnh không có box giữ nguyên.
- Tab Dataset: tùy chọn **Chỉ xuất ảnh có đủ class** (ví dụ head + body); ảnh chỉ có một trong các class đã chọn, hoặc không có box, không được xuất.

## Chưa phát hành — Nhập nhãn LabelMe/X-AnyLabeling và phím tắt class

- Gán nhãn nhanh hơn: lọc danh sách ảnh, `A`/`D` chuyển ảnh, `N` ảnh chưa xong kế tiếp, `Enter` duyệt và sang ảnh tiếp.
- Trích xuất video ghi mặc định vào `<tên dự án>_frames` cạnh dự án, không cần chọn thư mục.

- Nhãn lưu trong file JSON LabelMe/X-AnyLabeling cạnh từng ảnh thay vì chỉ trong dự án; LabelMe, X-AnyLabeling và yolo-train đọc trực tiếp được, file sửa bên ngoài được đọc lại khi mở dự án.
- Nút **Nhập ảnh**: thêm thư mục ảnh có sẵn (kể cả thư mục con) vào dự án, ảnh dùng tại chỗ, đọc nhãn từ JSON cạnh ảnh.
- Phím `1`/`H` = head, `2`/`B` = body (tổng quát: `1`–`9` và chữ cái đầu tên class) để đổi class box đang chọn hoặc chọn class để vẽ.

## Chưa phát hành — Tách công cụ train khỏi repo

- Chuyển `train_yolo.py`, `optimize_model.py`, `build_model.py`, `labelme_to_yolo.py`, `config.yml`, `train_rocm.cmd`/`.ps1` sang công cụ riêng `C:\Users\CiyE\Desktop\yolo-train` (README riêng, kết quả trong `runs/` của thư mục đó). Gỡ các file này và hướng dẫn train khỏi repo FrameForge.
- FrameForge chỉ gán box (thủ công hoặc bằng model đã nhập). `.venv-train/` vẫn ở repo (đã ignore) vì công cụ train dùng nó; bỏ `runs/` khỏi `.gitignore` của repo.

## Chưa phát hành — Gộp train, tối ưu và xuất model; class ID trong config.yml

- Thêm `scripts/build_model.py`: tune (tùy chọn) → train → tối ưu (NMS IoU, confidence F1 tốt nhất, latency) → xuất `.pt` mới kèm `.json` thông số, `config.yml` và `build_summary.json`. Launcher `train_rocm.cmd` có mode `build`/`build-finetune`, `-MaxLatencyMs`, `-TuneIterations`, `-Config`.
- Thêm `scripts/config.yml` (`0: head`, `1: body`); `labelme_to_yolo.py`, `train_yolo.py`, `optimize_model.py` đọc class ID từ đó và chặn `data.yaml` lệch ID.
- Các script là công cụ riêng, không gắn với app: bỏ giả định NMS IoU 0,7 và đo latency trên CPU; FrameForge chỉ dùng để gán box (thủ công hoặc bằng model đã nhập).
- Sửa đo latency khi số ảnh ít hơn số lần warmup (trước đây lỗi do danh sách thời gian rỗng) và chép file ONNX/OpenVINO đã export sang thư mục model.

## Chưa phát hành — Script tối ưu latency và độ chính xác

- Thêm `scripts/optimize_model.py`: `evaluate`, `benchmark`, `all`, `tune`; báo cáo `report.md`/`report.json`, khuyến nghị imgsz, NMS IoU, ngưỡng confidence cho app và format nhanh nhất giữ được độ chính xác.
- `train_yolo.py` thêm `--hyp` để train với hyperparameter đã tune; tách hàm `resolve_data_path` dùng chung.
- Kiểm tra bằng Ultralytics/torch giả; chưa đo trên model và GPU thật.

## Chưa phát hành — Script train cho imgsz 640 và 256

- Thêm `scripts/labelme_to_yolo.py` và cho `train_yolo.py`/`train_rocm.cmd` nhận thẳng thư mục ảnh + JSON LabelMe/X-AnyLabeling (head = 0, body = 1); ảnh chưa có JSON được bỏ qua như chưa gán nhãn, split theo thời gian chụp, cảnh báo box quá nhỏ ở imgsz 256. Launcher tách đúng danh sách `-Imgsz`/`-Weights`/`-Extra` khi chạy qua `.cmd` và có `-PreparedDir`, `-ValRatio`.

- Thêm `scripts/train_rocm.cmd`/`train_rocm.ps1`: chạy train/fine-tune bằng PyTorch ROCm trong `.venv-train`, kiểm tra bản ROCm và GPU trước (mode `check`), báo lỗi rõ khi thiếu torch, torch không phải ROCm, không thấy GPU, thiếu `-Data` hoặc `-Weights`. Thêm `runs/` vào `.gitignore`.

- `scripts/train_yolo.py` nhận `--imgsz 640 256` (mặc định cả hai), chạy mỗi kích thước thành run riêng `<tên>-<imgsz>` với batch mặc định 16 (640) và 64 (256) trên GPU 8 GB.
- Fine-tune nhận `--weights SIZE=PATH` cho từng kích thước hoặc một checkpoint dùng chung; mọi checkpoint được kiểm tra trước khi train và có cảnh báo khi imgsz của checkpoint khác kích thước đích.
- Train mới dùng `yolo11n.pt` (nhận cả `yolov11n.pt`). Thêm thanh tiến độ theo epoch cho từng run và tổng, thời gian epoch/đã chạy/ETA, mAP, tổng thời gian cuối cùng và log `frameforge_progress.log` trong thư mục run.
- Đã kiểm tra parse tham số, chọn batch, báo lỗi và callback tiến độ (trainer giả, có dừng sớm) bằng Python không có torch; chưa train thật.

## Chưa phát hành — Gallery toàn bộ ảnh và script train

- Gallery hiển thị mọi ảnh trong dự án thay vì 500 thumbnail của lượt trích gần nhất; danh sách được ảo hóa và thumbnail tải dần qua lệnh `image_thumbnails`.
- Thêm `scripts/train_yolo.py` để train/fine-tune YOLO11n ở imgsz 640 bằng GPU (CUDA/ROCm) ngoài app, mặc định theo RX 6600 8 GB / i3-12100F / 16 GB RAM. Chưa chạy train thật vì máy chưa có PyTorch GPU cho RX 6600.
- Engine pytest 70 passed; frontend build, Rust fmt/check/test pass.

## Chưa phát hành — Sửa các lỗi audit pipeline

- Model có class khác `head`/`body` dùng được: khi dự án chưa có nhãn, có thể đổi class dự án theo model (`set_project_classes`), kể cả ngay lúc nhập. Panel Model hiện cả khi dự án chưa có ảnh.
- Extraction trả metadata của mọi ảnh đã ghi; trước đây chỉ 500 ảnh đầu được thêm vào dự án. Gallery vẫn chỉ giữ 500 thumbnail.
- Ảnh bị ghi đè khi trích lại mất nhãn, candidate và lượt inference cũ thay vì giữ nhãn của nội dung cũ.
- Thêm ngưỡng confidence 1–99% và “Duyệt tất cả”/“Loại tất cả” gợi ý cho ảnh hiện tại.
- Ảnh có mọi box chạm biên crop được ghi lượt inference theo checkpoint + fingerprint; batch không chạy lại vô ích.
- Bỏ “Đã review” đưa box thủ công về chưa review, nên ảnh không còn được xuất ngầm.
- Inference truyền `imgsz` đã đăng ký của checkpoint vào Ultralytics.
- Engine pytest 69 passed; frontend build, Rust fmt/check/test pass. Chưa click-through với checkpoint thật.

## Chưa phát hành — Thu gọn về pipeline gán nhãn

- Sản phẩm chỉ còn pipeline: nhập model → nhập ảnh/video → model gắn nhãn → người dùng duyệt/gán lại → lưu dataset.
- Gỡ tab Huấn luyện cùng train/fine-tune YOLO11n, đánh giá model trên fixed set, Champion/Challenger, Smart Folder Discovery/nhập workspace chuyển máy và kiểm tra metadata ONNX khỏi UI, Tauri allowlist và engine.
- Gỡ lineage/parent manifest, replay và đóng băng val/test khỏi tab Dataset và exporter; manifest không còn `version`, `parent_*`, `fixed_validation_set`, `frozen_test_set`.
- Registry và trust flow nhập checkpoint YOLO11n chuyển sang `engine/frameforge/models.py`; giữ đường dẫn registry cũ để model đã nhập vẫn dùng được. Tab Gán nhãn có nút “Nhập model .pt”.
- Extra Python `train` đổi thành `inference` (PyTorch + Ultralytics); bỏ `onnxruntime`, `psutil`, `lap`. Engine pytest 64 passed, frontend build và Rust fmt/check/test pass; chưa build lại portable app.

## Chưa phát hành — Sửa app Windows phụ thuộc localhost

- Bật Tauri `custom-protocol` mặc định để release executable nạp frontend đã đóng gói thay vì cố kết nối Vite ở `localhost:1420`.
- Build lại portable app bằng `pnpm build:portable`; engine bundle qua health/training self-check, frontend production build thành công và cửa sổ `FrameForge` mở được trên máy Windows này.
- App được mở trực tiếp từ `src-tauri/target/release/FrameForge/frameforge.exe`; không có tiến trình server lắng nghe cổng 1420. Chưa kiểm chứng máy sạch, DirectML hoặc RX 6600.
- Khôi phục tên project mojibake trong snapshot legacy (có thể bị mã hóa sai nhiều lượt) trước khi hiển thị; không sửa dự án nguồn. Regression test bao phủ phục hồi và giữ nguyên chuỗi tiếng Việt hợp lệ; sau khi mở bản portable mới, snapshot hiện có đã lưu lại tên đúng.

## Chưa phát hành — Khôi phục lối vào Auto label

- Nút Auto label trên toolbar trước đây bị disable với tooltip “Suy luận chưa được triển khai”, gây hiểu nhầm dù inference thật đã có trong tab Gán nhãn.
- Nối nút này tới workflow Gán nhãn; người dùng chọn checkpoint đã xác minh và chạy inference ảnh/batch trong tab đó.
- Đóng gói Python/PyTorch/Ultralytics vào app portable macOS và thư mục portable Windows, không dùng installer. App macOS hiện tại 1,44 GiB; engine self-check, app launch, 110 engine tests, frontend build và Rust fmt/check đã pass.
- Bổ sung resource path rỗng trước Rust check trong CI; runner Linux/macOS cần thư mục tồn tại trước khi PyInstaller tạo runtime.
- CI run `38723939528` pass trên Ubuntu, macOS và Windows; hai runner desktop đã build portable app và smoke-check engine. Chưa xác minh cài/chạy Windows trên máy thật.

## Chưa phát hành — Rà soát production v1.0

- Nhập checkpoint `.pt` qua bản sao trong vùng quản lý riêng; giới hạn kích thước cả khi đọc/sao chép, xác minh nguồn và bản sao trước khi nạp, rồi xác minh lại checksum sau khi nạp.
- Giới hạn log stderr giữ trong Rust ở 256 KiB/request và vẫn đọc hết pipe, tránh giữ log training dài vô hạn trong RAM.
- Mở rộng GitHub Actions sang Ubuntu/macOS/Windows, thêm Tauri `cargo check` và build macOS app/Windows MSI trên runner tương ứng.
- Ghi label YOLO bằng byte UTF-8 ổn định để SHA-256 manifest khớp trên Windows; đọc/ghi ảnh qua I/O Unicode-aware để hỗ trợ tên file Unicode.
- Tạo được macOS `FrameForge.app` 4,50 MiB; engine trong bundle trả health `status: ok` với Python `.venv` hiện có. Engine tests local 110/110, Rust unit test 1/1, frontend build, Rust compile và bundle health thành công.
- CI run `38564201938`: Ubuntu/macOS/Windows đều thành công; Windows runner chạy engine tests, Rust checks và build MSI. MSI chưa được cài/click-through trên máy Windows thật.
- Chưa phát hành v1.0. Runtime/dependencies Python chưa được bundle; ký/notarize chưa cấu hình. Xem các cổng trong `DEVELOPMENT.md`.

## Chưa phát hành — Khởi tạo train và kế thừa imgsz

- Khi không chọn checkpoint để fine-tune, bắt đầu run mới từ YOLO11n pretrained `yolo11n.pt`.
- Khi chọn checkpoint đã xác minh, tự chọn `imgsz` ghi trong checkpoint hoặc đọc `args.yaml` cùng run bằng YAML an toàn; nếu cả hai thiếu, đánh dấu mặc định Ultralytics 640 và báo điều đó trong giao diện.
- Registry nhập checkpoint đời cũ không có provenance `imgsz` được báo là nguồn chưa ghi, tránh nhầm giá trị fallback với metadata thật.
- Hiển thị kích thước được nhận diện sau khi nhập checkpoint; thêm regression cho metadata `imgsz` có và thiếu.

## Chưa phát hành — Nhập thư mục làm việc từ máy khác

- Đổi Smart Folder Discovery thành luồng nhập thư mục làm việc; có thể khôi phục đường dẫn dataset từ `data.yaml` cũ, kể cả khi YAML nằm ngoài dataset, khi duy nhất một thư mục khớp tất cả split.
- Từ chối nối path khi có nhiều thư mục khớp hoặc tìm kiếm vượt giới hạn; thêm regression cho trường hợp YAML đặt riêng trong thư mục cấu hình và checkpoint vẫn chờ trust flow.
- Tạo `data.yaml` local và chuẩn hóa file list trong App Data, không sửa hoặc sao chép ảnh/nhãn nguồn. Giới hạn file list và chặn đường dẫn ngoài scope.
- Preflight xác minh workspace nguồn còn tồn tại và từ chối ảnh/nhãn đi qua symlink; checkpoint `.pt` vẫn chỉ được nhập sau trust flow.
- Dataset nhập vào không tự đặt output vào vùng staging; người dùng chọn riêng nơi lưu training runs.

## Chưa phát hành — Rà soát repository

- Thêm giấy phép GNU AGPL-3.0-or-later, phù hợp với tích hợp Ultralytics; ghi chú điều kiện cấp phép Ultralytics trong README.
- Chuyển project legacy có ngưỡng dHash mặc định 5 về 1 một lần; version marker giữ nguyên lựa chọn người dùng sau migration và provenance crop cũ.
- Thêm GitHub Actions cho frontend build, engine tests và Rust formatting.
- Xóa `agent.md` và `TASKS.md` trùng nội dung với `AGENTS.md` và `DEVELOPMENT.md`.

## Chưa phát hành — Workspace trích xuất

- Giảm ngưỡng dHash mặc định từ 5 xuống 1 bit để chỉ lọc crop gần như trùng; áp dụng tương tự khi negative sampling. Quota negative 10% là quy tắc riêng, không phải ngưỡng dHash.
- Thêm negative sampling cho dataset: mặc định 10%, preset 0/5/10/25% hoặc tùy chỉnh 0–100%, chọn lặp lại theo seed và bỏ ảnh gần như trùng dHash ≤1. Giữ nguyên negative trong fixed validation/test của parent và cảnh báo nếu vượt quota.
- Tự lưu snapshot project và trạng thái workspace; tự khôi phục khi mở app mà không hiện hộp thoại. Nếu `.ffproj` nguồn mất hoặc checksum thay đổi, khôi phục thành dự án chưa lưu để tránh ghi đè.
- Dataset validation/export loại ảnh còn box thủ công `unreviewed`; báo số ảnh bị loại. Replay parent bỏ ảnh train cũ chưa review và từ chối dùng nhãn thủ công chưa review trong fixed validation/test.
- Thêm nhập checkpoint `.pt` YOLO11n detection có xác nhận tin cậy; kiểm tra task/kiến trúc/class, giới hạn 2 GiB, SHA-256 và lưu bản sao quản lý riêng. Model đã nhập xuất hiện trong inference/fine-tune theo class schema; discovered checkpoint vẫn chưa tin cậy. Đã kiểm tra bằng fake runtime, chưa click-through với checkpoint thật.
- Bỏ qua detection box chạm sát biên crop vì object có thể bị cắt hoặc lớn hơn khung; báo số box bị bỏ qua và không tạo negative candidate nếu tất cả box đều bị bỏ qua.
- Làm rõ dHash ngay dưới thanh trượt: giải thích ngưỡng là số bit Hamming tối đa giữa hai hash 64-bit, không phải số ảnh; tổng kết gallery báo số crop bị lọc thực tế.
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
