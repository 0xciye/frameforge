import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

type EngineHealth = {
  status: string;
  engine: string;
  python_version: string;
  capabilities: string[];
};

type HealthResult = {
  ok: boolean;
  health?: EngineHealth;
  error?: string;
};

function App() {
  const [result, setResult] = useState<HealthResult | null>(null);
  const [checking, setChecking] = useState(false);

  async function checkEngine() {
    setChecking(true);
    setResult(null);
    try {
      setResult(await invoke<HealthResult>("engine_health"));
    } catch (error) {
      setResult({ ok: false, error: String(error) });
    } finally {
      setChecking(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="FrameForge trang chủ">
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span>FrameForge</span>
        </a>
        <div className="topbar-meta">
          <span className="build-label">BẢN DỰNG THỬ</span>
          <span className="topbar-divider" />
          <span>v0.1.0</span>
        </div>
      </header>

      <section className="workspace" aria-labelledby="welcome-title">
        <div className="eyebrow"><span className="eyebrow-dot" /> KHÔNG GIAN LÀM VIỆC</div>
        <h1 id="welcome-title">Sẵn sàng cho khung hình đầu tiên?</h1>
        <p className="intro">
          FrameForge giúp bạn chuẩn bị video và nhãn dữ liệu trong một không gian gọn gàng.
          Kết nối engine để bắt đầu.
        </p>

        <section className="engine-card" aria-labelledby="engine-title">
          <div className="card-heading">
            <div className="engine-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M8.5 3.75h7A2.75 2.75 0 0 1 18.25 6.5v11a2.75 2.75 0 0 1-2.75 2.75h-7A2.75 2.75 0 0 1 5.75 17.5v-11A2.75 2.75 0 0 1 8.5 3.75Z" />
                <path d="M9 8.25h6m-6 3.75h6m-6 3.75h3" />
              </svg>
            </div>
            <div>
              <h2 id="engine-title">Python engine</h2>
              <p>Tiến trình xử lý cục bộ của FrameForge</p>
            </div>
            <span className={`connection-state ${result?.ok ? "is-ready" : result ? "is-error" : "is-idle"}`}>
              <span className="state-dot" />
              {result?.ok ? "Đã kết nối" : result ? "Chưa kết nối" : "Chưa kiểm tra"}
            </span>
          </div>

          <div className="card-rule" />

          <div className="engine-body">
            <div className="engine-copy">
              <h3>{result?.ok ? "Engine đang hoạt động" : "Kiểm tra kết nối engine"}</h3>
              <p>
                {result?.ok
                  ? `${result.health?.engine} đã phản hồi qua giao thức JSON-lines.`
                  : "Xác nhận Python đã cài đặt và tiến trình nền có thể phản hồi ứng dụng."}
              </p>
              {result?.ok && result.health && (
                <dl className="health-details">
                  <div><dt>Python</dt><dd>{result.health.python_version}</dd></div>
                  <div><dt>Khả năng</dt><dd>{result.health.capabilities.join(", ")}</dd></div>
                </dl>
              )}
              {result && !result.ok && <p className="error-message">{result.error}</p>}
            </div>
            <button className="primary-button" type="button" onClick={checkEngine} disabled={checking}>
              {checking ? (
                <><span className="spinner" aria-hidden="true" /> Đang kiểm tra</>
              ) : (
                <>
                  Kiểm tra engine
                  <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
                    <path d="M4.25 10h11.5m-5-5 5 5-5 5" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </section>

        <div className="workspace-note">
          <span className="note-icon" aria-hidden="true">i</span>
          <p>Phase 1 chỉ xác nhận kết nối. Công cụ nhập video và gán nhãn sẽ có trong các giai đoạn tiếp theo.</p>
        </div>
      </section>

      <footer className="statusbar">
        <div><span className={`footer-indicator ${result?.ok ? "online" : ""}`} />{result?.ok ? "Engine sẵn sàng" : "Engine chưa kết nối"}</div>
        <span>FrameForge <span className="status-separator">/</span> Workspace</span>
      </footer>
    </main>
  );
}

export default App;
