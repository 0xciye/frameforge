use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{AppHandle, Emitter, Manager};

const FHD_WIDTH: u64 = 1920;
const FHD_HEIGHT: u64 = 1080;
const MAX_ENGINE_DIAGNOSTIC_BYTES: usize = 256 * 1024;

fn read_engine_diagnostics<R: Read>(mut reader: R) -> String {
    let mut retained = Vec::with_capacity(MAX_ENGINE_DIAGNOSTIC_BYTES);
    let mut buffer = [0; 8192];
    let mut truncated = false;
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                let available = MAX_ENGINE_DIAGNOSTIC_BYTES.saturating_sub(retained.len());
                let keep = available.min(count);
                retained.extend_from_slice(&buffer[..keep]);
                truncated |= keep < count;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    let mut diagnostics = String::from_utf8_lossy(&retained).into_owned();
    if truncated {
        diagnostics.push_str("\n[Đã rút gọn log engine vượt 256 KiB.]");
    }
    diagnostics
}

#[derive(Clone, Default)]
struct EngineJobs(Arc<Mutex<HashMap<String, JobHandle>>>);

/// Engines kept running between short requests: starting the packaged Python engine costs about
/// 0.7 s, which every image switch, autosave and thumbnail batch otherwise paid. Cancellable jobs
/// (extraction, inference) still get their own process so cancelling can stop it.
#[derive(Clone, Default)]
struct EnginePool(Arc<Mutex<Vec<PooledEngine>>>);

const MAX_IDLE_ENGINES: usize = 3;

struct PooledEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    diagnostics: Arc<Mutex<String>>,
}

impl PooledEngine {
    fn kill(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_pooled_engine(app: &AppHandle, app_data_dir: &Path) -> Result<PooledEngine, String> {
    let (engine_program, engine_script) = engine_invocation(app)?;
    let mut command = Command::new(engine_program);
    if let Some(script) = engine_script {
        command.arg(script);
    }
    command
        .env("FRAMEFORGE_APP_DATA", app_data_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_engine_command(&mut command);
    let mut child = command.spawn().map_err(|error| {
        format!("Không chạy được engine tích hợp. Hãy đóng rồi mở lại ứng dụng. ({error})")
    })?;
    let stdin = child
        .stdin
        .take()
        .ok_or("Không mở được kênh gửi lệnh tới engine.")?;
    let stdout = child
        .stdout
        .take()
        .ok_or("Không mở được kênh nhận dữ liệu từ engine.")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("Không mở được kênh chẩn đoán của engine.")?;
    // Drain stderr for the engine's whole life, keeping only the latest part for error messages.
    let diagnostics = Arc::new(Mutex::new(String::new()));
    let sink = diagnostics.clone();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Ok(mut text) = sink.lock() {
                if text.len() > MAX_ENGINE_DIAGNOSTIC_BYTES {
                    let cut = text.len() - MAX_ENGINE_DIAGNOSTIC_BYTES / 2;
                    let cut = (cut..text.len())
                        .find(|index| text.is_char_boundary(*index))
                        .unwrap_or(text.len());
                    text.drain(..cut);
                }
                text.push_str(&line);
                text.push('\n');
            }
        }
    });
    Ok(PooledEngine {
        child,
        stdin,
        stdout: BufReader::new(stdout),
        diagnostics,
    })
}

/// Sends one request to a pooled engine and reads its progress events and final response.
fn exchange(
    engine: &mut PooledEngine,
    app: &AppHandle,
    request_id: &str,
    request_bytes: &[u8],
) -> Result<Value, String> {
    engine
        .stdin
        .write_all(request_bytes)
        .and_then(|_| engine.stdin.write_all(b"\n"))
        .and_then(|_| engine.stdin.flush())
        .map_err(|error| format!("Không gửi được yêu cầu tới engine: {error}"))?;
    let mut line = String::new();
    loop {
        line.clear();
        let read = engine
            .stdout
            .read_line(&mut line)
            .map_err(|error| format!("Không đọc được phản hồi từ engine: {error}"))?;
        if read == 0 {
            return Err("engine-exited".to_string());
        }
        if line.trim().is_empty() {
            continue;
        }
        let response: Value = serde_json::from_str(&line)
            .map_err(|error| format!("Phản hồi JSON-lines không hợp lệ: {error}"))?;
        if response.get("id").and_then(Value::as_str) != Some(request_id) {
            return Err("Mã phản hồi không khớp với yêu cầu.".to_string());
        }
        if let Some(event) = response.get("event") {
            let _ = app.emit(
                "engine-progress",
                json!({"id": request_id, "progress": event}),
            );
            continue;
        }
        return Ok(response);
    }
}

fn run_pooled_request(
    app: &AppHandle,
    pool: &EnginePool,
    app_data_dir: &Path,
    request_id: &str,
    request_bytes: &[u8],
) -> Result<Value, String> {
    let mut attempts = 0;
    let response = loop {
        attempts += 1;
        let idle = pool.0.lock().ok().and_then(|mut idle| idle.pop());
        let reused = idle.is_some();
        let mut engine = match idle {
            Some(engine) => engine,
            None => spawn_pooled_engine(app, app_data_dir)?,
        };
        match exchange(&mut engine, app, request_id, request_bytes) {
            Ok(response) => {
                let mut idle = pool
                    .0
                    .lock()
                    .map_err(|_| "Không truy cập được engine.".to_string())?;
                if idle.len() < MAX_IDLE_ENGINES {
                    idle.push(engine);
                } else {
                    drop(idle);
                    engine.kill();
                }
                break response;
            }
            // An idle engine that died meanwhile: retry once on a fresh one.
            Err(error) if error == "engine-exited" && reused && attempts == 1 => {
                engine.kill();
                continue;
            }
            Err(error) => {
                let detail = engine
                    .diagnostics
                    .lock()
                    .map(|text| text.trim().to_string())
                    .unwrap_or_default();
                engine.kill();
                return Err(if error == "engine-exited" {
                    if detail.is_empty() {
                        "Python engine dừng trước khi trả dữ liệu.".to_string()
                    } else {
                        format!("Python engine gặp lỗi: {detail}")
                    }
                } else {
                    error
                });
            }
        }
    };
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
}

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
    pool: tauri::State<'_, EnginePool>,
    request: Value,
) -> Result<Value, String> {
    let jobs = jobs.inner().clone();
    let pool = pool.inner().clone();
    tauri::async_runtime::spawn_blocking(move || run_engine_request(app, jobs, pool, request))
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

fn run_engine_request(
    app: AppHandle,
    jobs: EngineJobs,
    pool: EnginePool,
    request: Value,
) -> Result<Value, String> {
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
    if matches!(command.as_str(), "predict_crops" | "auto_label_project") {
        let marker_dir = app_data_dir.join("inference").join("cancel");
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
        "extract_frames" | "predict_crops" | "auto_label_project"
    );
    if !cancellable {
        return run_pooled_request(&app, &pool, &app_data_dir, &request_id, &request_bytes);
    }
    if jobs
        .0
        .lock()
        .map_err(|_| "Không truy cập được danh sách tác vụ engine.".to_string())?
        .contains_key(&request_id)
    {
        return Err("Mã tác vụ đang được sử dụng.".to_string());
    }

    let (engine_program, engine_script) = engine_invocation(&app)?;
    let mut engine_command = Command::new(engine_program);
    if let Some(script) = engine_script {
        engine_command.arg(script);
    }
    engine_command
        .env("FRAMEFORGE_APP_DATA", &app_data_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_engine_command(&mut engine_command);
    let mut child = engine_command.spawn().map_err(|error| {
        format!("Không chạy được engine tích hợp. Hãy đóng rồi mở lại ứng dụng. ({error})")
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
        let stderr_reader = thread::spawn(move || read_engine_diagnostics(stderr));

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

fn validate_checkpoint_argument(value: Option<&Value>) -> Result<(), String> {
    let checkpoint = canonical_existing_path(
        value
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
    Ok(())
}

fn validate_request(command: &str, arguments: &Value) -> Result<(), String> {
    let arguments = arguments
        .as_object()
        .ok_or_else(|| "Arguments phải là một JSON object.".to_string())?;
    match command {
        "health" => Ok(()),
        "inference_capabilities" | "list_models" | "load_session" => Ok(()),
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
            if arguments
                .get("replace")
                .is_some_and(|value| !value.is_boolean())
            {
                return Err("Tùy chọn gợi ý lại không hợp lệ.".to_string());
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
        "image_thumbnails" => {
            let paths = arguments
                .get("paths")
                .and_then(Value::as_array)
                .filter(|paths| !paths.is_empty() && paths.len() <= 64)
                .ok_or("Mỗi lần chỉ tải từ 1 đến 64 thumbnail.")?;
            for path in paths {
                let path = PathBuf::from(path.as_str().ok_or("Đường dẫn thumbnail không hợp lệ.")?);
                if !path.is_absolute() {
                    return Err("Đường dẫn thumbnail phải là đường dẫn tuyệt đối.".to_string());
                }
                let extension = path
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                if !["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"]
                    .contains(&extension.as_str())
                {
                    return Err("Thumbnail chỉ dành cho tệp ảnh.".to_string());
                }
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
            if arguments
                .get("lossless")
                .is_some_and(|value| !value.is_boolean())
            {
                return Err("Tùy chọn preview không hợp lệ.".to_string());
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
        "import_yolo_checkpoint" => {
            let path = canonical_existing_path(
                arguments
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu đường dẫn checkpoint YOLO11n.")?,
            )?;
            if !path.is_file()
                || !path
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("pt"))
            {
                return Err("Chọn một checkpoint YOLO `.pt` hợp lệ.".to_string());
            }
            if arguments.get("trust_source").and_then(Value::as_bool) != Some(true) {
                return Err(
                    "Hãy xác nhận bạn tin cậy nguồn checkpoint trước khi nạp tệp `.pt`."
                        .to_string(),
                );
            }
            Ok(())
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
            // A new folder is created by the engine; its parent must already exist.
            let requested = PathBuf::from(output);
            let output_path = if requested.exists() {
                canonical_existing_path(output)?
            } else {
                let parent = requested
                    .parent()
                    .filter(|parent| requested.is_absolute() && parent.is_dir())
                    .ok_or("Thư mục đầu ra không hợp lệ.")?;
                canonical_existing_path(&parent.to_string_lossy())?.join(
                    requested
                        .file_name()
                        .ok_or("Thư mục đầu ra không hợp lệ.")?,
                )
            };
            if output_path.exists() && !output_path.is_dir() {
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
            if let Some(detection) = options.get("detection").filter(|value| !value.is_null()) {
                let detection = detection
                    .as_object()
                    .ok_or("Cài đặt lọc bằng model không hợp lệ.")?;
                validate_checkpoint_argument(detection.get("checkpoint"))?;
                if detection
                    .get("confidence_threshold")
                    .and_then(Value::as_f64)
                    .is_none_or(|value| !(value > 0.0 && value < 1.0))
                {
                    return Err("Confidence threshold phải nằm trong khoảng 0–1.".to_string());
                }
                if detection.get("trust_source").and_then(Value::as_bool) != Some(true) {
                    return Err(
                        "Hãy xác nhận bạn tin cậy nguồn checkpoint trước khi nạp tệp `.pt`."
                            .to_string(),
                    );
                }
            }
            Ok(())
        }
        "save_extraction_suggestions" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            validate_checkpoint_argument(arguments.get("checkpoint"))?;
            let items = arguments
                .get("items")
                .and_then(Value::as_array)
                .filter(|items| !items.is_empty() && items.len() <= 100_000)
                .ok_or("Danh sách gợi ý trích xuất phải có từ 1 đến 100000 ảnh.")?;
            // The engine checks every box, fingerprint and that each image belongs to the project.
            if items.iter().any(|item| {
                item.get("image_path")
                    .and_then(Value::as_str)
                    .is_none_or(|image| !Path::new(image).is_absolute())
            }) {
                return Err("Đường dẫn ảnh phải là đường dẫn tuyệt đối.".to_string());
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
            if let Some(value) = arguments.get("skipped_large_objects") {
                if value.as_u64().is_none_or(|count| count > 300) {
                    return Err("Số box bị bỏ qua do chạm biên crop không hợp lệ.".to_string());
                }
            }
            if arguments
                .get("replace")
                .is_some_and(|value| !value.is_boolean())
            {
                return Err("Tùy chọn gợi ý lại không hợp lệ.".to_string());
            }
            Ok(())
        }
        "delete_images" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            let images = arguments
                .get("image_paths")
                .and_then(Value::as_array)
                .filter(|images| !images.is_empty() && images.len() <= 10000)
                .ok_or("Hãy chọn từ 1 đến 10000 ảnh để xóa.")?;
            for image in images {
                // The engine also checks that every path belongs to the project.
                let image = image.as_str().ok_or("Đường dẫn ảnh không hợp lệ.")?;
                if !Path::new(image).is_absolute() {
                    return Err("Đường dẫn ảnh phải là đường dẫn tuyệt đối.".to_string());
                }
            }
            Ok(())
        }
        "delete_model" => {
            arguments
                .get("run_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty() && value.len() <= 128)
                .ok_or("Mã model không hợp lệ.")?;
            Ok(())
        }
        "relocate_images" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            if let Some(root) = arguments.get("root").filter(|root| !root.is_null()) {
                let root = root.as_str().ok_or("Thư mục ảnh không hợp lệ.")?;
                if !Path::new(root).is_absolute() || !canonical_existing_path(root)?.is_dir() {
                    return Err("Thư mục ảnh không hợp lệ.".to_string());
                }
            }
            Ok(())
        }
        "review_images" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            let images = arguments
                .get("image_paths")
                .and_then(Value::as_array)
                .filter(|images| !images.is_empty() && images.len() <= 1_000_000)
                .ok_or("Hãy chọn từ 1 đến 1000000 ảnh để duyệt.")?;
            if images.iter().any(|image| {
                image
                    .as_str()
                    .is_none_or(|image| !Path::new(image).is_absolute())
            }) {
                return Err("Đường dẫn ảnh phải là đường dẫn tuyệt đối.".to_string());
            }
            Ok(())
        }
        "import_images" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            let paths = arguments
                .get("paths")
                .and_then(Value::as_array)
                .filter(|paths| !paths.is_empty() && paths.len() <= 10000)
                .ok_or("Hãy chọn từ 1 đến 10000 ảnh hoặc thư mục.")?;
            for value in paths {
                validate_media_path(value.as_str().ok_or("Đường dẫn ảnh không hợp lệ.")?, true)?;
            }
            for flag in ["mark_reviewed", "dry_run"] {
                if arguments.get(flag).is_some_and(|value| !value.is_boolean()) {
                    return Err("Tùy chọn nhập ảnh không hợp lệ.".to_string());
                }
            }
            Ok(())
        }
        "import_labelme_folder" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            let folder = canonical_existing_path(
                arguments
                    .get("folder")
                    .and_then(Value::as_str)
                    .ok_or("Thiếu thư mục LabelMe/X-AnyLabeling.")?,
            )?;
            if !folder.is_dir() {
                return Err("Hãy chọn thư mục chứa ảnh và JSON LabelMe/X-AnyLabeling.".to_string());
            }
            if arguments
                .get("mark_reviewed")
                .is_some_and(|value| !value.is_boolean())
            {
                return Err("Tùy chọn đánh dấu đã review không hợp lệ.".to_string());
            }
            Ok(())
        }
        "set_project_classes" => {
            let path = arguments
                .get("path")
                .and_then(Value::as_str)
                .ok_or("Thiếu đường dẫn dự án.")?;
            validate_existing_project_path(path)?;
            let names = arguments
                .get("class_names")
                .and_then(Value::as_array)
                .filter(|names| !names.is_empty() && names.len() <= 1000)
                .ok_or("Danh sách class phải có từ 1 đến 1000 tên.")?;
            if names.iter().any(|name| {
                name.as_str()
                    .is_none_or(|value| value.trim().is_empty() || value.chars().count() > 200)
            }) {
                return Err("Tên class phải là chuỗi không rỗng, tối đa 200 ký tự.".to_string());
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
            if let Some(value) = arguments.get("negative_ratio") {
                let ratio = value
                    .as_f64()
                    .ok_or("Tỷ lệ ảnh negative phải là số từ 0 đến 100%.")?;
                if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
                    return Err("Tỷ lệ ảnh negative phải từ 0 đến 100%.".to_string());
                }
            }
            if let Some(value) = arguments.get("required_class_ids") {
                let ids = value
                    .as_array()
                    .ok_or("Danh sách class bắt buộc không hợp lệ.")?;
                if ids.len() > 1000 || ids.iter().any(|id| id.as_u64().is_none()) {
                    return Err("Danh sách class bắt buộc không hợp lệ.".to_string());
                }
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

fn engine_binary_path(resource_dir: &Path) -> PathBuf {
    let executable = if cfg!(target_os = "windows") {
        "frameforge-engine.exe"
    } else {
        "frameforge-engine"
    };
    resource_dir
        .join("engine-runtime")
        .join("frameforge-engine")
        .join(executable)
}

fn engine_invocation(app: &AppHandle) -> Result<(PathBuf, Option<PathBuf>), String> {
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|error| format!("Không xác định được thư mục tài nguyên: {error}"))?;
    let bundled_engine = engine_binary_path(&resource_dir);
    if bundled_engine.is_file() {
        return Ok((bundled_engine, None));
    }

    if !cfg!(debug_assertions) {
        return Err("Không tìm thấy engine tích hợp trong app bundle.".to_string());
    }

    let script = project_root().join("engine").join("frameforge_engine.py");
    if !script.is_file() {
        return Err("Không tìm thấy engine Python trong workspace phát triển.".to_string());
    }
    Ok((engine_program(), Some(script)))
}

#[cfg(target_os = "windows")]
fn configure_engine_command(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(0x08000000);
}

#[cfg(not(target_os = "windows"))]
fn configure_engine_command(_: &mut Command) {}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn engine_program() -> PathBuf {
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
        PathBuf::from("python3")
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(EngineJobs::default())
        .manage(EnginePool::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            engine_request,
            cancel_engine_request
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_ENGINE_DIAGNOSTIC_BYTES, engine_binary_path, read_engine_diagnostics, validate_request,
    };
    use serde_json::{Value, json};
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    #[test]
    fn packaged_engine_path_is_inside_app_resources() {
        let path = engine_binary_path(Path::new("/app/Resources"));
        let executable = if cfg!(target_os = "windows") {
            "frameforge-engine.exe"
        } else {
            "frameforge-engine"
        };
        assert_eq!(
            path,
            PathBuf::from("/app/Resources/engine-runtime/frameforge-engine").join(executable)
        );
    }

    #[test]
    fn image_import_takes_existing_absolute_paths() {
        let root = std::env::temp_dir().join(format!("ff-images-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let project = root.join("p.ffproj");
        std::fs::write(&project, b"").unwrap();
        let project = project.to_string_lossy().into_owned();
        let folder = root.to_string_lossy().into_owned();
        let request = |paths: Value, dry_run: Value| {
            validate_request(
                "import_images",
                &json!({"path": project, "paths": paths, "dry_run": dry_run}),
            )
        };
        assert!(request(json!([folder]), json!(true)).is_ok());
        assert!(request(json!([]), json!(true)).is_err());
        assert!(request(json!(["relative/folder"]), json!(false)).is_err());
        assert!(
            request(json!([project]), json!(false)).is_err(),
            "a project is not an image"
        );
        assert!(request(json!([folder]), json!("yes")).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn labelme_import_needs_a_project_and_a_folder() {
        let root = std::env::temp_dir().join(format!("ff-labelme-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let project = root.join("p.ffproj");
        std::fs::write(&project, b"").unwrap();
        let folder = root.to_string_lossy().into_owned();
        let project = project.to_string_lossy().into_owned();
        let request = |folder: &str, reviewed: Value| {
            validate_request(
                "import_labelme_folder",
                &json!({"path": project, "folder": folder, "mark_reviewed": reviewed}),
            )
        };
        assert!(request(&folder, json!(true)).is_ok());
        assert!(
            request(&project, json!(true)).is_err(),
            "a file is not a folder"
        );
        assert!(request(&folder, json!("yes")).is_err());
        assert!(request("relative/folder", json!(false)).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn bulk_review_takes_absolute_image_paths() {
        let root = std::env::temp_dir().join(format!("ff-review-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let project = root.join("p.ffproj");
        std::fs::write(&project, b"").unwrap();
        let project = project.to_string_lossy().into_owned();
        let image = root.join("a.png").to_string_lossy().into_owned();
        let request = |images: Value| {
            validate_request(
                "review_images",
                &json!({"path": project, "image_paths": images}),
            )
        };
        assert!(request(json!([image])).is_ok());
        assert!(request(json!([])).is_err());
        assert!(request(json!(["relative.png"])).is_err());
        assert!(request(json!([1])).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn relocation_takes_an_optional_absolute_image_folder() {
        let root = std::env::temp_dir().join(format!("ff-relocate-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let project = root.join("p.ffproj");
        std::fs::write(&project, b"").unwrap();
        let project = project.to_string_lossy().into_owned();
        let folder = root.to_string_lossy().into_owned();
        assert!(validate_request("relocate_images", &json!({"path": project})).is_ok());
        assert!(
            validate_request("relocate_images", &json!({"path": project, "root": folder})).is_ok()
        );
        assert!(
            validate_request(
                "relocate_images",
                &json!({"path": project, "root": "relative"})
            )
            .is_err()
        );
        assert!(
            validate_request(
                "relocate_images",
                &json!({"path": project, "root": project})
            )
            .is_err()
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn engine_diagnostics_are_bounded_and_the_pipe_is_drained() {
        let contents = vec![b'x'; MAX_ENGINE_DIAGNOSTIC_BYTES + 8192];
        let mut reader = Cursor::new(contents);

        let diagnostics = read_engine_diagnostics(&mut reader);

        assert_eq!(
            reader.position(),
            (MAX_ENGINE_DIAGNOSTIC_BYTES + 8192) as u64
        );
        assert!(diagnostics.starts_with(&"x".repeat(MAX_ENGINE_DIAGNOSTIC_BYTES)));
        assert!(diagnostics.ends_with("[Đã rút gọn log engine vượt 256 KiB.]"));
    }
}
