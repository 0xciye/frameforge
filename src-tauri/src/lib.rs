use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{AppHandle, Emitter, Manager};

const FHD_WIDTH: u64 = 1920;
const FHD_HEIGHT: u64 = 1080;

#[derive(Clone, Default)]
struct EngineJobs(Arc<Mutex<HashMap<String, JobHandle>>>);

#[derive(Clone)]
struct JobHandle {
    child: Arc<Mutex<Child>>,
    cancelled: Arc<AtomicBool>,
    cancellation_marker: Option<PathBuf>,
}

#[tauri::command]
async fn engine_request(
    app: AppHandle,
    jobs: tauri::State<'_, EngineJobs>,
    request: Value,
) -> Result<Value, String> {
    let jobs = jobs.inner().clone();
    tauri::async_runtime::spawn_blocking(move || run_engine_request(app, jobs, request))
        .await
        .map_err(|error| format!("Tác vụ engine bị gián đoạn: {error}"))?
}

#[tauri::command]
fn cancel_engine_request(
    jobs: tauri::State<'_, EngineJobs>,
    request_id: String,
) -> Result<bool, String> {
    let handle = jobs
        .0
        .lock()
        .map_err(|_| "Không truy cập được danh sách tác vụ engine.".to_string())?
        .get(&request_id)
        .cloned();

    let Some(handle) = handle else {
        return Ok(false);
    };
    if let Some(marker) = handle.cancellation_marker {
        if let Some(parent) = marker.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("Không tạo được thư mục hủy an toàn: {error}"))?;
        }
        std::fs::write(marker, b"cancel")
            .map_err(|error| format!("Không ghi được yêu cầu dừng an toàn: {error}"))?;
        return Ok(true);
    }
    handle.cancelled.store(true, Ordering::SeqCst);
    let mut child = handle
        .child
        .lock()
        .map_err(|_| "Không truy cập được tiến trình engine.".to_string())?;
    match child.kill() {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => Ok(false),
        Err(error) => Err(format!("Không hủy được tác vụ: {error}")),
    }
}

fn run_engine_request(app: AppHandle, jobs: EngineJobs, request: Value) -> Result<Value, String> {
    let mut request = request;
    let request_id = request
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Yêu cầu engine thiếu mã request.".to_string())?
        .to_owned();
    let command = request
        .get("command")
        .and_then(Value::as_str)
        .ok_or_else(|| "Yêu cầu engine thiếu tên lệnh.".to_string())?
        .to_owned();
    if request_id.len() > 128 {
        return Err("Mã request không được dài quá 128 ký tự.".to_string());
    }
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Không xác định được thư mục dữ liệu FrameForge: {error}"))?;
    let mut cancellation_marker = None;
    if matches!(
        command.as_str(),
        "train_yolo_queue"
            | "evaluate_yolo_checkpoint"
            | "scan_project_folder"
            | "predict_crops"
            | "auto_label_project"
    ) {
        let marker_dir = app_data_dir
            .join(match command.as_str() {
                "train_yolo_queue" | "evaluate_yolo_checkpoint" => "training",
                "scan_project_folder" => "discovery",
                _ => "inference",
            })
            .join("cancel");
        std::fs::create_dir_all(&marker_dir)
            .map_err(|error| format!("Không tạo được thư mục điều khiển tác vụ: {error}"))?;
        let safe_id: String = request_id
            .bytes()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let marker = marker_dir.join(format!("{safe_id}.cancel"));
        let arguments = request
            .get_mut("arguments")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| "Arguments phải là một JSON object.".to_string())?;
        arguments.insert("cancel_file".to_string(), json!(marker));
        cancellation_marker = Some(marker);
    }
    validate_request(&command, request.get("arguments").unwrap_or(&Value::Null))?;
    let request_bytes = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
    if request_bytes.len() > 64 * 1024 * 1024 {
        return Err("Yêu cầu engine vượt giới hạn 64 MiB.".to_string());
    }
    let cancellable = matches!(
        command.as_str(),
        "extract_frames"
            | "train_yolo_queue"
            | "evaluate_yolo_checkpoint"
            | "scan_project_folder"
            | "predict_crops"
            | "auto_label_project"
    );
    if cancellable
        && jobs
            .0
            .lock()
            .map_err(|_| "Không truy cập được danh sách tác vụ engine.".to_string())?
            .contains_key(&request_id)
    {
        return Err("Mã tác vụ đang được sử dụng.".to_string());
    }

    let script = engine_script_path(&app)?;
    let python = engine_program(&script);
    let mut child = Command::new(python)
        .arg(script)
        .env("FRAMEFORGE_APP_DATA", &app_data_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "Không chạy được Python engine. Hãy cài Python 3.10 trở lên hoặc đặt FRAMEFORGE_PYTHON. ({error})"
            )
        })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Không mở được kênh nhận dữ liệu từ engine.".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Không mở được kênh chẩn đoán của engine.".to_string())?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Không mở được kênh gửi lệnh tới engine.".to_string())?;
    if let Err(error) = stdin
        .write_all(&request_bytes)
        .and_then(|_| stdin.write_all(b"\n"))
    {
        let _ = child.kill();
        return Err(format!("Không gửi được yêu cầu tới engine: {error}"));
    }
    drop(stdin);

    let child = Arc::new(Mutex::new(child));
    let cancelled = Arc::new(AtomicBool::new(false));
    if cancellable {
        let mut running = jobs
            .0
            .lock()
            .map_err(|_| "Không đăng ký được tác vụ engine.".to_string())?;
        if running.contains_key(&request_id) {
            if let Ok(mut process) = child.lock() {
                let _ = process.kill();
                let _ = process.wait();
            }
            return Err("Mã tác vụ đang được sử dụng.".to_string());
        }
        running.insert(
            request_id.clone(),
            JobHandle {
                child: child.clone(),
                cancelled: cancelled.clone(),
                cancellation_marker: cancellation_marker.clone(),
            },
        );
    }

    let result = (|| -> Result<Value, String> {
        let stderr_reader = thread::spawn(move || {
            let mut text = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut text);
            text
        });

        let mut final_response = None;
        let mut read_error = None;
        for line in BufReader::new(stdout).lines() {
            match line {
                Ok(line) if line.trim().is_empty() => continue,
                Ok(line) => match serde_json::from_str::<Value>(&line) {
                    Ok(response) => {
                        let response_id = response.get("id").and_then(Value::as_str);
                        if response_id != Some(request_id.as_str()) {
                            read_error = Some("Mã phản hồi không khớp với yêu cầu.".to_string());
                            break;
                        }
                        if let Some(event) = response.get("event") {
                            let _ = app.emit(
                                "engine-progress",
                                json!({"id": request_id, "progress": event}),
                            );
                        } else {
                            final_response = Some(response);
                        }
                    }
                    Err(error) => {
                        read_error = Some(format!("Phản hồi JSON-lines không hợp lệ: {error}"));
                        break;
                    }
                },
                Err(error) => {
                    read_error = Some(format!("Không đọc được phản hồi từ engine: {error}"));
                    break;
                }
            }
        }

        let status = child
            .lock()
            .map_err(|_| "Không truy cập được tiến trình engine.".to_string())?
            .wait()
            .map_err(|error| format!("Không chờ được tiến trình engine: {error}"))?;
        let diagnostics = stderr_reader.join().unwrap_or_default();
        if cancelled.load(Ordering::SeqCst) {
            return Err("Đã hủy tác vụ.".to_string());
        }
        if let Some(error) = read_error {
            return Err(error);
        }
        if !status.success() {
            let detail = diagnostics.trim();
            return Err(if detail.is_empty() {
                format!("Python engine thoát với mã {:?}.", status.code())
            } else {
                format!("Python engine gặp lỗi: {detail}")
            });
        }

        let response = final_response.ok_or_else(|| {
            if diagnostics.trim().is_empty() {
                "Python engine không trả dữ liệu.".to_string()
            } else {
                format!("Python engine không trả dữ liệu: {}", diagnostics.trim())
            }
        })?;
        if response.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err(response
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Python engine từ chối yêu cầu.")
                .to_string());
        }
        response
            .get("data")
            .cloned()
            .ok_or_else(|| "Phản hồi thành công thiếu trường data.".to_string())
    })();

    if cancellable {
        if let Ok(mut running) = jobs.0.lock() {
            running.remove(&request_id);
        }
    }
    if let Some(marker) = cancellation_marker {
        let _ = std::fs::remove_file(marker);
    }
    result
}

fn validate_request(command: &str, arguments: &Value) -> Result<(), String> {
    let arguments = arguments
        .as_object()
        .ok_or_else(|| "Arguments phải là một JSON object.".to_string())?;
    match command {
        "health" => Ok(()),
        "training_capabilities"
        | "list_training_runs"
        | "get_evaluation_targets"
        | "get_model_promotion_state"
        | "rollback_model_promotion"
        | "load_session" => Ok(()),
        "compare_model_challenger" | "promote_model_challenger" => {
            let tolerance = match arguments.get("max_regression") {
                None => 0.0,
                Some(value) => value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or("Tolerance regression phải là số hữu hạn từ 0 đến 1.")?,
            };
            if !(0.0..=1.0).contains(&tolerance) {
                return Err("Tolerance regression phải nằm trong khoảng 0–1.".to_string());
            }
            Ok(())
        }
        "list_model_evaluations" => {
            if let Some(run_id) = arguments.get("run_id").filter(|value| !value.is_null()) {
                if run_id
                    .as_str()
                    .is_none_or(|value| value.is_empty() || value.len() > 128)
                {
                    return Err("Mã training run không hợp lệ.".to_string());
                }
            }
            Ok(())
        }
        "save_session" => {
            if arguments
                .get("project")
                .and_then(Value::as_object)
                .is_none()
                || arguments
                    .get("session")
                    .and_then(Value::as_object)
                    .is_none()
            {
                return Err("Thiếu project hoặc metadata phiên làm việc.".to_string());
            }
            Ok(())
        }
        "register_discovered_run" => {
            let root = canonical_existing_path(
                arguments
                    .get("discovery_root")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu thư mục quét đã xác nhận.")?,
            )?;
            if !root.is_dir() {
                return Err("Thư mục quét không còn tồn tại.".to_string());
            }
            let run = arguments
                .get("run")
                .and_then(Value::as_object)
                .ok_or("Thiếu metadata training run.")?;
            let source = canonical_existing_path(
                run.get("source_dir")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu thư mục training run.")?,
            )?;
            if !source.is_dir() || !source.starts_with(&root) {
                return Err("Training run nằm ngoài thư mục đã quét.".to_string());
            }
            Ok(())
        }
        "scan_project_folder" => {
            let path = canonical_existing_path(
                arguments
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu thư mục cần quét.")?,
            )?;
            if !path.is_dir() {
                return Err("Hãy chọn thư mục dự án để quét.".to_string());
            }
            let max_depth = arguments
                .get("max_depth")
                .map(Value::as_u64)
                .flatten()
                .unwrap_or(8);
            if arguments.get("max_depth").is_some()
                && arguments.get("max_depth").and_then(Value::as_u64).is_none()
            {
                return Err("Độ sâu quét phải là số nguyên từ 1–16.".to_string());
            }
            if !(1..=16).contains(&max_depth) {
                return Err("Độ sâu quét phải nằm trong khoảng 1–16.".to_string());
            }
            let max_files = arguments
                .get("max_files")
                .map(Value::as_u64)
                .flatten()
                .unwrap_or(50_000);
            if arguments.get("max_files").is_some()
                && arguments.get("max_files").and_then(Value::as_u64).is_none()
            {
                return Err("Giới hạn file phải là số nguyên từ 1–200000.".to_string());
            }
            if !(1..=200_000).contains(&max_files) {
                return Err("Giới hạn file phải nằm trong khoảng 1–200000.".to_string());
            }
            if let Some(ignored_dirs) = arguments.get("ignored_dirs") {
                let ignored_dirs = ignored_dirs
                    .as_array()
                    .filter(|items| items.len() <= 64)
                    .ok_or("Danh sách thư mục bỏ qua không hợp lệ.")?;
                for name in ignored_dirs {
                    let name = name
                        .as_str()
                        .filter(|name| {
                            !name.is_empty()
                                && name.len() <= 128
                                && !name.contains('/')
                                && !name.contains('\\')
                        })
                        .ok_or("Tên thư mục bỏ qua không hợp lệ.")?;
                    if name == "." || name == ".." {
                        return Err("Tên thư mục bỏ qua không hợp lệ.".to_string());
                    }
                }
            }
            Ok(())
        }
        "inspect_training_dataset" => validate_yaml_file(
            arguments
                .get("data_yaml")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn data.yaml.")?,
        ),
        "evaluate_yolo_checkpoint" => {
            let checkpoint = canonical_existing_path(
                arguments
                    .get("checkpoint")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu checkpoint cần đánh giá.")?,
            )?;
            if !checkpoint.is_file()
                || !checkpoint
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("pt"))
            {
                return Err("Chọn checkpoint `.pt` đã đăng ký trong FrameForge.".to_string());
            }
            validate_yaml_file(
                arguments
                    .get("data_yaml")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu data.yaml của fixed set.")?,
            )?;
            let split = arguments
                .get("split")
                .and_then(Value::as_str)
                .unwrap_or("val");
            if !["val", "test"].contains(&split) {
                return Err("Chỉ đánh giá split validation hoặc test.".to_string());
            }
            let image_size = arguments
                .get("imgsz")
                .and_then(Value::as_u64)
                .ok_or("imgsz đánh giá phải là số nguyên từ 1 đến 2048 px.")?;
            if !(1..=2048).contains(&image_size) {
                return Err("imgsz đánh giá phải nằm trong khoảng 1–2048 px.".to_string());
            }
            let batch = match arguments.get("batch") {
                None => 8,
                Some(value) => value
                    .as_u64()
                    .ok_or("Batch đánh giá phải là số nguyên từ 1 đến 512.")?,
            };
            if !(1..=512).contains(&batch) {
                return Err("Batch đánh giá phải nằm trong khoảng 1–512.".to_string());
            }
            for (key, default) in [("confidence_threshold", 0.001), ("iou_threshold", 0.7)] {
                let value = match arguments.get(key) {
                    None => default,
                    Some(value) => value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or("Confidence và NMS IoU phải là số hữu hạn.")?,
                };
                if !(0.0..=1.0).contains(&value) {
                    return Err("Confidence và NMS IoU phải nằm trong khoảng 0–1.".to_string());
                }
            }
            let device = arguments
                .get("device")
                .and_then(Value::as_str)
                .unwrap_or("auto");
            if !["auto", "cpu", "mps", "cuda", "cuda:0"].contains(&device) {
                return Err("Thiết bị đánh giá không hợp lệ.".to_string());
            }
            Ok(())
        }
        "save_evaluation_targets" => {
            validate_yaml_file(
                arguments
                    .get("data_yaml")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu data.yaml để kiểm tra class mục tiêu.")?,
            )?;
            let allowed = [
                "precision",
                "recall",
                "f1",
                "map50",
                "map50_95",
                "avg_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "fps",
                "throughput",
                "max_model_size_mb",
                "max_memory_mb",
            ];
            for key in ["targets", "per_class"] {
                let values = arguments
                    .get(key)
                    .and_then(Value::as_object)
                    .filter(|values| values.len() <= 64)
                    .ok_or("Cấu hình chỉ số mục tiêu không hợp lệ.")?;
                if key == "targets" {
                    for (metric, target) in values {
                        if !allowed.contains(&metric.as_str()) {
                            return Err("Tên chỉ số mục tiêu không được hỗ trợ.".to_string());
                        }
                        if let Some(number) = target.as_f64() {
                            if !number.is_finite()
                                || (allowed[..5].contains(&metric.as_str())
                                    && !(0.0..=1.0).contains(&number))
                                || (allowed[..5].contains(&metric.as_str()) == false
                                    && number <= 0.0)
                            {
                                return Err(
                                    "Giá trị mục tiêu phải hữu hạn và nằm trong phạm vi hợp lệ."
                                        .to_string(),
                                );
                            }
                        } else if !target.is_null() {
                            return Err("Mỗi chỉ số mục tiêu phải là số hoặc để trống.".to_string());
                        }
                    }
                } else {
                    for (class_name, class_values) in values {
                        if class_name.is_empty() || class_name.len() > 128 {
                            return Err("Tên class mục tiêu không hợp lệ.".to_string());
                        }
                        let class_values = class_values
                            .as_object()
                            .filter(|items| items.len() <= 5)
                            .ok_or("Mục tiêu theo class không hợp lệ.")?;
                        for (metric, target) in class_values {
                            if !allowed[..5].contains(&metric.as_str()) {
                                return Err(
                                    "Mục tiêu theo class chỉ nhận metric chất lượng.".to_string()
                                );
                            }
                            if let Some(number) = target.as_f64() {
                                if !number.is_finite() || !(0.0..=1.0).contains(&number) {
                                    return Err("Mục tiêu chất lượng phải nằm trong khoảng 0–1."
                                        .to_string());
                                }
                            } else if !target.is_null() {
                                return Err(
                                    "Mỗi mục tiêu class phải là số hoặc để trống.".to_string()
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        }
        "train_yolo_queue" => {
            validate_yaml_file(
                arguments
                    .get("data_yaml")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu đường dẫn data.yaml.")?,
            )?;
            let output = canonical_existing_path(
                arguments
                    .get("output_dir")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu thư mục đầu ra training.")?,
            )?;
            if !output.is_dir() {
                return Err("Thư mục đầu ra training không hợp lệ.".to_string());
            }
            let sizes = arguments
                .get("sizes")
                .and_then(Value::as_array)
                .filter(|sizes| !sizes.is_empty() && sizes.len() <= 12)
                .ok_or("Chọn từ 1 đến 12 kích thước train.")?;
            let mut unique_sizes = std::collections::HashSet::new();
            for size in sizes {
                let size = size
                    .as_u64()
                    .filter(|size| (1..=2048).contains(size))
                    .ok_or("imgsz phải là số nguyên từ 1 đến 2048 px.")?;
                if !unique_sizes.insert(size) {
                    return Err("Danh sách imgsz không được trùng kích thước.".to_string());
                }
            }
            let device = arguments
                .get("device")
                .and_then(Value::as_str)
                .unwrap_or("auto");
            if !["auto", "cpu", "mps", "cuda", "cuda:0"].contains(&device) {
                return Err("Thiết bị training không hợp lệ.".to_string());
            }
            let on_error = arguments
                .get("on_error")
                .and_then(Value::as_str)
                .unwrap_or("stop");
            if !["stop", "continue"].contains(&on_error) {
                return Err("Chính sách khi job lỗi phải là stop hoặc continue.".to_string());
            }
            let base_model = arguments
                .get("base_model")
                .and_then(Value::as_str)
                .unwrap_or("yolo11n.pt");
            if base_model != "yolo11n.pt" {
                let checkpoint = canonical_existing_path(base_model)?;
                if !checkpoint.is_file()
                    || !checkpoint
                        .extension()
                        .and_then(|value| value.to_str())
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("pt"))
                {
                    return Err("Checkpoint fine-tune phải là tệp .pt hợp lệ.".to_string());
                }
            }
            Ok(())
        }
        "predict_crops" => {
            let checkpoint = canonical_existing_path(
                arguments
                    .get("checkpoint")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu checkpoint inference.")?,
            )?;
            if !checkpoint.is_file()
                || !checkpoint
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("pt"))
            {
                return Err("Chọn checkpoint `.pt` đã đăng ký trong FrameForge.".to_string());
            }
            let images = arguments
                .get("images")
                .and_then(Value::as_array)
                .filter(|images| !images.is_empty() && images.len() <= 500)
                .ok_or("Inference nhận từ 1 đến 500 ảnh mỗi lượt.")?;
            for image in images {
                let image = validate_media_path(
                    image
                        .as_str()
                        .ok_or("Đường dẫn ảnh inference không hợp lệ.")?,
                    false,
                )?;
                let extension = image
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                if !["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"]
                    .contains(&extension.as_str())
                {
                    return Err("Inference chỉ nhận ảnh đã trích xuất.".to_string());
                }
            }
            for (key, default) in [("confidence_threshold", 0.25), ("iou_threshold", 0.7)] {
                let value = match arguments.get(key) {
                    None => default,
                    Some(value) => value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or("Confidence threshold và NMS IoU phải là số hữu hạn.")?,
                };
                if !(0.0..=1.0).contains(&value) {
                    return Err(
                        "Confidence threshold và NMS IoU phải nằm trong khoảng 0–1.".to_string()
                    );
                }
            }
            if let Some(classes) = arguments.get("class_ids").filter(|value| !value.is_null()) {
                let classes = classes
                    .as_array()
                    .filter(|classes| classes.len() <= 256)
                    .ok_or("Bộ lọc class inference không hợp lệ.")?;
                let mut unique = std::collections::HashSet::new();
                for class_id in classes {
                    let class_id = class_id
                        .as_u64()
                        .filter(|class_id| *class_id <= u32::MAX as u64)
                        .ok_or("Class ID phải là số nguyên không âm.")?;
                    if !unique.insert(class_id) {
                        return Err("Danh sách class ID không được trùng.".to_string());
                    }
                }
            }
            Ok(())
        }
        "auto_label_project" => {
            let project = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(project)?;
            let checkpoint = canonical_existing_path(
                arguments
                    .get("checkpoint")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu checkpoint inference.")?,
            )?;
            if !checkpoint.is_file()
                || !checkpoint
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("pt"))
            {
                return Err("Chọn checkpoint `.pt` đã đăng ký trong FrameForge.".to_string());
            }
            for (key, default) in [("confidence_threshold", 0.25), ("iou_threshold", 0.7)] {
                let value = match arguments.get(key) {
                    None => default,
                    Some(value) => value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or("Confidence threshold và NMS IoU phải là số hữu hạn.")?,
                };
                if !(0.0..=1.0).contains(&value) {
                    return Err(
                        "Confidence threshold và NMS IoU phải nằm trong khoảng 0–1.".to_string()
                    );
                }
            }
            Ok(())
        }
        "inspect_sources" => {
            let paths = arguments
                .get("paths")
                .and_then(Value::as_array)
                .filter(|paths| !paths.is_empty() && paths.len() <= 500)
                .ok_or_else(|| "Chọn từ 1 đến 500 nguồn media.".to_string())?;
            for path in paths {
                validate_media_path(path.as_str().ok_or("Đường dẫn nguồn không hợp lệ.")?, true)?;
            }
            Ok(())
        }
        "preview_frame" => {
            validate_media_path(
                arguments
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu đường dẫn media.")?,
                false,
            )?;
            if arguments
                .get("frame_index")
                .and_then(Value::as_u64)
                .is_none()
            {
                return Err("Chỉ số frame phải là số nguyên không âm.".to_string());
            }
            Ok(())
        }
        "preview_center_crop" => {
            validate_media_path(
                arguments
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu đường dẫn media.")?,
                false,
            )?;
            if arguments
                .get("frame_index")
                .and_then(Value::as_u64)
                .is_none()
            {
                return Err("Chỉ số frame phải là số nguyên không âm.".to_string());
            }
            for key in ["width", "height"] {
                let size = arguments
                    .get(key)
                    .and_then(Value::as_u64)
                    .ok_or_else(|| format!("Thiếu kích thước crop {key}."))?;
                let maximum = if key == "width" {
                    FHD_WIDTH
                } else {
                    FHD_HEIGHT
                };
                if !(1..=maximum).contains(&size) {
                    return Err(format!(
                        "Kích thước crop tối đa là {} × {} px (FHD).",
                        FHD_WIDTH, FHD_HEIGHT
                    ));
                }
            }
            Ok(())
        }
        "inspect_model" => {
            let path = canonical_existing_path(
                arguments
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu đường dẫn model ONNX.")?,
            )?;
            if path.is_file()
                && path
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("onnx"))
            {
                Ok(())
            } else {
                Err("Hãy chọn một tệp model .onnx hợp lệ.".to_string())
            }
        }
        "extract_frames" => {
            let paths = arguments
                .get("paths")
                .and_then(Value::as_array)
                .filter(|paths| !paths.is_empty() && paths.len() <= 500)
                .ok_or_else(|| "Chọn ít nhất một nguồn media để trích xuất.".to_string())?;
            for path in paths {
                validate_media_path(path.as_str().ok_or("Đường dẫn nguồn không hợp lệ.")?, true)?;
            }
            let output = arguments
                .get("output_dir")
                .and_then(Value::as_str)
                .ok_or("Thiếu thư mục đầu ra.")?;
            let output_path = canonical_existing_path(output)?;
            if !output_path.is_dir() {
                return Err("Thư mục đầu ra không hợp lệ.".to_string());
            }
            let options = arguments
                .get("options")
                .and_then(Value::as_object)
                .ok_or("Thiếu cài đặt trích xuất.")?;
            let mode = options
                .get("crop_mode")
                .and_then(Value::as_str)
                .unwrap_or("letterbox");
            if !["center", "letterbox", "grid"].contains(&mode) {
                return Err("Chế độ crop chưa khả dụng.".to_string());
            }
            let minimum_size = if mode == "center" { 1 } else { 16 };
            for key in ["output_width", "output_height"] {
                let size = options
                    .get(key)
                    .and_then(Value::as_u64)
                    .ok_or_else(|| format!("Thiếu kích thước đầu ra {key}."))?;
                let maximum = if key == "output_width" {
                    FHD_WIDTH
                } else {
                    FHD_HEIGHT
                };
                if !(minimum_size..=maximum).contains(&size) {
                    return Err(format!(
                        "Kích thước đầu ra phải từ {minimum_size} đến {maximum} px."
                    ));
                }
            }
            Ok(())
        }
        "create_project" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_new_project_path(path)
        }
        "save_project" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_new_project_path(path)
        }
        "save_annotations" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            let image_path = arguments
                .get("image_path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn ảnh.")?;
            let image_path = validate_media_path(image_path, false)?;
            let extension = image_path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"].contains(&extension.as_str()) {
                return Err("Chỉ lưu nhãn cho tệp ảnh.".to_string());
            }
            if arguments
                .get("annotations")
                .and_then(Value::as_array)
                .is_none_or(|annotations| annotations.len() > 10000)
            {
                return Err("Danh sách nhãn phải có tối đa 10000 box.".to_string());
            }
            let fingerprint = arguments
                .get("image_fingerprint")
                .and_then(Value::as_str)
                .ok_or("Thiếu fingerprint ảnh đã preview.")?;
            if fingerprint.len() != 64 || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err("Fingerprint ảnh không hợp lệ.".to_string());
            }
            if !matches!(
                arguments.get("status").and_then(Value::as_str),
                Some("ready" | "reviewed")
            ) {
                return Err("Trạng thái ảnh không hợp lệ.".to_string());
            }
            Ok(())
        }
        "save_ai_suggestions" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            let image_path = arguments
                .get("image_path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn ảnh.")?;
            let image_path = validate_media_path(image_path, false)?;
            let extension = image_path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"].contains(&extension.as_str()) {
                return Err("Chỉ lưu gợi ý cho tệp ảnh.".to_string());
            }
            let checkpoint = canonical_existing_path(
                arguments
                    .get("checkpoint")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu checkpoint inference.")?,
            )?;
            if !checkpoint.is_file()
                || !checkpoint
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.eq_ignore_ascii_case("pt"))
            {
                return Err("Chọn checkpoint `.pt` đã đăng ký trong FrameForge.".to_string());
            }
            if arguments
                .get("predictions")
                .and_then(Value::as_array)
                .is_none_or(|predictions| predictions.len() > 300)
            {
                return Err("Danh sách gợi ý phải có tối đa 300 box.".to_string());
            }
            let fingerprint = arguments
                .get("image_fingerprint")
                .and_then(Value::as_str)
                .ok_or("Thiếu fingerprint ảnh đã preview.")?;
            if fingerprint.len() != 64 || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err("Fingerprint ảnh không hợp lệ.".to_string());
            }
            Ok(())
        }
        "review_ai_negative_candidate" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            let candidate_id = arguments
                .get("candidate_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty() && value.len() <= 128)
                .ok_or("Mã candidate không hợp lệ.")?;
            let _ = candidate_id;
            let action = arguments
                .get("action")
                .and_then(Value::as_str)
                .ok_or("Thiếu thao tác review candidate.")?;
            if !matches!(action, "confirmed_negative" | "rejected") {
                return Err("Thao tác review candidate không hợp lệ.".to_string());
            }
            if action == "confirmed_negative" {
                let fingerprint = arguments
                    .get("image_fingerprint")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu fingerprint ảnh đã preview.")?;
                if fingerprint.len() != 64
                    || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit())
                {
                    return Err("Fingerprint ảnh không hợp lệ.".to_string());
                }
            }
            Ok(())
        }
        "validate_dataset" | "export_yolo_dataset" => {
            let project = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(project)?;
            let mode = arguments
                .get("split_mode")
                .and_then(Value::as_str)
                .unwrap_or("source");
            if !["source", "segment"].contains(&mode) {
                return Err("Cách chia dataset không hợp lệ.".to_string());
            }
            for key in ["train_ratio", "val_ratio", "test_ratio"] {
                let ratio = arguments
                    .get(key)
                    .and_then(Value::as_f64)
                    .unwrap_or(match key {
                        "train_ratio" => 0.8,
                        _ => 0.1,
                    });
                if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
                    return Err("Tỷ lệ train/val/test phải từ 0 đến 1.".to_string());
                }
            }
            let train = arguments
                .get("train_ratio")
                .and_then(Value::as_f64)
                .unwrap_or(0.8);
            let val = arguments
                .get("val_ratio")
                .and_then(Value::as_f64)
                .unwrap_or(0.1);
            let test = arguments
                .get("test_ratio")
                .and_then(Value::as_f64)
                .unwrap_or(0.1);
            if train <= 0.0 || (train + val + test - 1.0).abs() > 1e-6 {
                return Err("Tổng tỷ lệ phải bằng 100% và train phải lớn hơn 0%.".to_string());
            }
            let seed = arguments.get("seed").and_then(Value::as_u64).unwrap_or(42);
            if seed > u32::MAX as u64 {
                return Err("Seed phải nằm trong khoảng 0 đến 4294967295.".to_string());
            }
            let segment_seconds = arguments
                .get("segment_seconds")
                .and_then(Value::as_f64)
                .unwrap_or(30.0);
            if !segment_seconds.is_finite() || !(1.0..=3600.0).contains(&segment_seconds) {
                return Err("Độ dài đoạn video phải từ 1 đến 3600 giây.".to_string());
            }
            let replay_strategy = arguments
                .get("replay_strategy")
                .and_then(Value::as_str)
                .unwrap_or("all");
            if !["all", "balanced", "new"].contains(&replay_strategy) {
                return Err("Chính sách replay dataset không hợp lệ.".to_string());
            }
            let freeze_evaluation_set = arguments
                .get("freeze_evaluation_set")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if arguments
                .get("freeze_evaluation_set")
                .is_some_and(|value| !value.is_boolean())
            {
                return Err("Cờ đóng băng validation/test không hợp lệ.".to_string());
            }
            if arguments
                .get("replay_strategy")
                .is_some_and(|value| !value.is_string())
            {
                return Err("Chính sách replay dataset không hợp lệ.".to_string());
            }
            let parent_manifest = match arguments.get("parent_manifest_path") {
                Some(Value::String(value)) if !value.trim().is_empty() => {
                    let path = canonical_existing_path(value)?;
                    if !path.is_file()
                        || path.file_name().and_then(|name| name.to_str()) != Some("manifest.json")
                    {
                        return Err(
                            "Chọn manifest.json của một phiên bản FrameForge đã xuất.".to_string()
                        );
                    }
                    Some(path)
                }
                Some(Value::Null) | None | Some(Value::String(_)) => None,
                Some(_) => return Err("Đường dẫn parent manifest không hợp lệ.".to_string()),
            };
            if parent_manifest.is_some() && freeze_evaluation_set {
                return Err(
                    "Parent đã có val/test cố định; không thể tạo bộ mới trong cùng lượt xuất."
                        .to_string(),
                );
            }
            if parent_manifest.is_some() && test != 0.0 {
                return Err(
                    "Khi dùng parent, đặt Test (%) của dữ liệu mới bằng 0 để giữ nguyên test set."
                        .to_string(),
                );
            }
            if command == "export_yolo_dataset" {
                let output = arguments
                    .get("output_dir")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu thư mục export dataset.")?;
                if !canonical_existing_path(output)?.is_dir() {
                    return Err("Thư mục export dataset không hợp lệ.".to_string());
                }
            }
            Ok(())
        }
        "load_project" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            let path = canonical_existing_path(path)?;
            if path.is_file()
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("ffproj"))
            {
                Ok(())
            } else {
                Err("Hãy chọn một tệp dự án .ffproj.".to_string())
            }
        }
        _ => Err("Lệnh engine không được phép.".to_string()),
    }
}

fn validate_media_path(value: &str, allow_directory: bool) -> Result<PathBuf, String> {
    let path = canonical_existing_path(value)?;
    if path.is_dir() && allow_directory {
        return Ok(path);
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let allowed = [
        "avi", "bmp", "jpeg", "jpg", "mkv", "mov", "mp4", "png", "tif", "tiff", "webp", "webm",
    ];
    if path.is_file() && allowed.contains(&extension.as_str()) {
        Ok(path)
    } else {
        Err("Nguồn media không tồn tại hoặc có định dạng không hỗ trợ.".to_string())
    }
}

fn canonical_existing_path(value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("Đường dẫn phải là đường dẫn tuyệt đối.".to_string());
    }
    path.canonicalize()
        .map_err(|error| format!("Đường dẫn không hợp lệ: {error}"))
}

fn validate_yaml_file(value: &str) -> Result<(), String> {
    let path = canonical_existing_path(value)?;
    if path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                matches!(extension.to_ascii_lowercase().as_str(), "yaml" | "yml")
            })
    {
        Ok(())
    } else {
        Err("Hãy chọn tệp data.yaml hoặc .yml có thật.".to_string())
    }
}

fn validate_new_project_path(value: &str) -> Result<(), String> {
    let path = PathBuf::from(value);
    if !path.is_absolute()
        || !path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("ffproj"))
    {
        return Err("Tệp dự án phải là đường dẫn tuyệt đối có đuôi .ffproj.".to_string());
    }
    let parent = path
        .parent()
        .filter(|parent| parent.is_dir())
        .ok_or_else(|| "Thư mục dự án không tồn tại.".to_string())?;
    parent
        .canonicalize()
        .map_err(|error| format!("Không xác định được thư mục dự án: {error}"))?;
    Ok(())
}

fn validate_existing_project_path(value: &str) -> Result<(), String> {
    let path = canonical_existing_path(value)?;
    if path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("ffproj"))
    {
        Ok(())
    } else {
        Err("Hãy chọn một tệp dự án .ffproj hợp lệ.".to_string())
    }
}

fn engine_script_path(app: &AppHandle) -> Result<PathBuf, String> {
    let bundled_path = app
        .path()
        .resource_dir()
        .map_err(|error| format!("Không xác định được thư mục tài nguyên: {error}"))?
        .join("engine")
        .join("frameforge_engine.py");
    if bundled_path.is_file() {
        return Ok(bundled_path);
    }

    let development_path = project_root().join("engine").join("frameforge_engine.py");
    if development_path.is_file() {
        return Ok(development_path);
    }
    Err("Không tìm thấy Python engine trong gói ứng dụng.".to_string())
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn engine_program(script: &Path) -> PathBuf {
    if let Some(program) = std::env::var_os("FRAMEFORGE_PYTHON") {
        return PathBuf::from(program);
    }
    let root = project_root();
    let venv = if cfg!(target_os = "windows") {
        root.join(".venv").join("Scripts").join("python.exe")
    } else {
        root.join(".venv").join("bin").join("python")
    };
    if venv.is_file() {
        return venv;
    }
    if cfg!(target_os = "windows") {
        PathBuf::from("python")
    } else {
        let _ = script;
        PathBuf::from("python3")
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(EngineJobs::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            engine_request,
            cancel_engine_request
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
