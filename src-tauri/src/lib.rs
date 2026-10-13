use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{AppHandle, Manager};

#[derive(Debug, Deserialize, Serialize)]
struct EngineHealth {
    status: String,
    engine: String,
    python_version: String,
    capabilities: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EngineError {
    message: String,
}

#[derive(Debug, Deserialize)]
struct EngineReply {
    id: Option<serde_json::Value>,
    ok: bool,
    data: Option<EngineHealth>,
    error: Option<EngineError>,
}

#[derive(Debug, Serialize)]
struct HealthResult {
    ok: bool,
    health: Option<EngineHealth>,
    error: Option<String>,
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

    let development_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("engine")
        .join("frameforge_engine.py");

    if development_path.is_file() {
        return Ok(development_path);
    }

    Err("Không tìm thấy Python engine trong gói ứng dụng.".to_string())
}

#[tauri::command]
fn engine_health(app: AppHandle) -> HealthResult {
    match run_engine_health(&app) {
        Ok(health) => HealthResult {
            ok: true,
            health: Some(health),
            error: None,
        },
        Err(error) => HealthResult {
            ok: false,
            health: None,
            error: Some(error),
        },
    }
}

fn run_engine_health(app: &AppHandle) -> Result<EngineHealth, String> {
    let script = engine_script_path(app)?;
    let python = std::env::var_os("FRAMEFORGE_PYTHON").unwrap_or_else(|| {
        if cfg!(target_os = "windows") {
            "python".into()
        } else {
            "python3".into()
        }
    });

    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "Không chạy được Python. Hãy cài Python 3 hoặc đặt biến FRAMEFORGE_PYTHON. ({error})"
            )
        })?;

    let request = b"{\"id\":\"health\",\"command\":\"health\"}\n";
    child
        .stdin
        .take()
        .ok_or_else(|| "Không mở được kênh gửi lệnh tới Python engine.".to_string())?
        .write_all(request)
        .map_err(|error| format!("Không gửi được yêu cầu tới Python engine: {error}"))?;

    let output = child
        .wait_with_output()
        .map_err(|error| format!("Không đọc được phản hồi từ Python engine: {error}"))?;

    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let detail = detail.trim();
        let detail = if detail.chars().count() > 360 {
            format!("{}…", detail.chars().take(360).collect::<String>())
        } else {
            detail.to_string()
        };
        return Err(if detail.is_empty() {
            format!("Python engine thoát với mã {:?}.", output.status.code())
        } else {
            format!("Python engine gặp lỗi: {detail}")
        });
    }

    let reply: EngineReply = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("Phản hồi từ Python engine không hợp lệ: {error}"))?;
    if reply.id != Some(serde_json::Value::String("health".to_string())) {
        return Err("Phản hồi Python engine không khớp yêu cầu health check.".to_string());
    }
    if !reply.ok {
        return Err(reply
            .error
            .map(|error| error.message)
            .unwrap_or_else(|| "Python engine từ chối yêu cầu health check.".to_string()));
    }

    reply
        .data
        .ok_or_else(|| "Phản hồi health check thiếu dữ liệu engine.".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![engine_health])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
