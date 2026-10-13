use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{AppHandle, Emitter, Manager};

#[derive(Clone, Default)]
struct EngineJobs(Arc<Mutex<HashMap<String, JobHandle>>>);

#[derive(Clone)]
struct JobHandle {
    child: Arc<Mutex<Child>>,
    cancelled: Arc<AtomicBool>,
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
    validate_request(&command, request.get("arguments").unwrap_or(&Value::Null))?;
    let cancellable = command == "extract_frames";
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
    let request_bytes = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
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
    result
}

fn validate_request(command: &str, arguments: &Value) -> Result<(), String> {
    let arguments = arguments
        .as_object()
        .ok_or_else(|| "Arguments phải là một JSON object.".to_string())?;
    match command {
        "health" => Ok(()),
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
                if !(1..=4096).contains(&size) {
                    return Err("Kích thước crop phải từ 1 đến 4096 px.".to_string());
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
                if !(minimum_size..=4096).contains(&size) {
                    return Err(format!(
                        "Kích thước đầu ra phải từ {minimum_size} đến 4096 px."
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
