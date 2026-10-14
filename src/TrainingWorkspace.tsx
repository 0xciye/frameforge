import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, confirm } from "@tauri-apps/plugin-dialog";
import { Activity, AlertCircle, Check, ChevronDown, CircleHelp, Cpu, Database, FolderOpen, LoaderCircle, Play, Plus, RefreshCw, Square, Zap } from "lucide-react";
import "./TrainingWorkspace.css";

type TrainingDevice = "auto" | "cpu" | "mps" | "cuda";
type TrainingSettings = { epochs: number; batch: number; patience: number; workers: number; save_period: number; seed: number; lr0: number | null };
type TrainingCapabilities = {
  available: boolean;
  missing_packages: string[];
  install_command: string;
  python_version: string;
  torch_version: string | null;
  ultralytics_version: string | null;
  devices: { cpu: boolean; mps: boolean; cuda: boolean };
  device_details: { mps_built?: boolean; mps_device_name?: string | null; cuda_device_names?: string[] };
  preferred_device: string;
  hardware: {
    os: string; architecture: string; processor: string; logical_cpu_count: number | null;
    physical_cpu_count: number | null; memory_total_bytes: number | null; memory_available_bytes: number | null; disk_free_bytes: number | null;
  };
  recommendation: TrainingSettings & { device: string; cache: boolean; amp: boolean; reasons: string[] };
  torch_error?: string;
};
type DatasetSummary = {
  valid: boolean; data_yaml: string; dataset_root: string; dataset_id: string | null; dataset_fingerprint: string;
  class_names: string[]; class_instance_counts: Record<string, number>; image_counts: Record<string, number>;
  instance_counts: Record<string, number>; total_images: number; total_instances: number; warnings: string[]; errors: string[];
};
type TrainingRun = {
  run_id: string; status: string; requested_imgsz: number; effective_imgsz?: number; stride?: number;
  source?: string; name?: string; training_imgsz?: number | number[] | null;
  architecture_claimed?: string | null; task_claimed?: string | null; checkpoint_trusted?: boolean;
  checkpoint_path?: string | null; checkpoint_exists?: boolean; checkpoint_checksum_matches?: boolean | null;
  checkpoint_usable?: boolean; device_actual?: string | null; base_model?: string; parent_model_id?: string | null;
  dataset_id?: string | null; class_names?: string[]; completed_at?: string | null; metrics?: Record<string, unknown>;
  best_checkpoint?: string | null; error?: string | null; elapsed_seconds?: number | null;
};
type RunList = { runs: TrainingRun[]; available_count: number };
type QueueResult = { queue_id: string; status: string; completed_count: number; failed_count: number; jobs: TrainingRun[]; queue_root: string };
type TrainProgress = {
  stage: string; queue_id?: string; run_id?: string; job_index?: number; total_jobs?: number;
  requested_imgsz?: number; effective_imgsz?: number; stride?: number; device_selected?: string; device_actual?: string;
  epoch?: number; epochs_total?: number; batch?: number; batches_total?: number; elapsed_seconds?: number;
  estimated_remaining_seconds?: number | null; seconds_per_batch?: number | null; seconds_per_epoch?: number | null;
  images_per_second?: number | null; losses?: Record<string, number | null>; metrics?: Record<string, number | null>;
  runtime?: { cpu_percent?: number | null; process_memory_bytes?: number | null; system_memory_available_bytes?: number | null; gpu_memory_allocated_bytes?: number | null; gpu_utilization_percent?: number | null };
  warnings?: string[]; status?: string; error?: string; best_checkpoint?: string | null; last_checkpoint?: string | null;
};
type MetricPoint = { key: string; epoch: number; loss: number | null; map50: number | null };
type Props = { initialDataYaml?: string | null; onInitialDataYamlConsumed?(): void };

const IMAGE_SIZE_PRESETS = [192, 224, 256, 288, 320, 352, 416, 512, 640];

function requestId() {
  return globalThis.crypto?.randomUUID?.() ?? `train-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function engineRequest<T>(command: string, args: Record<string, unknown> = {}): Promise<T> {
  return invoke<T>("engine_request", { request: { id: requestId(), command, arguments: args } });
}

function formatBytes(value?: number | null) {
  if (!Number.isFinite(value) || !value) return "—";
  const units = ["B", "KB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDuration(value?: number | null) {
  if (!Number.isFinite(value) || value == null || value < 0) return "—";
  const seconds = Math.round(value);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  return hours ? `${hours}g ${minutes}p` : `${minutes}p ${String(rest).padStart(2, "0")}s`;
}

function formatMetric(value: unknown, fractionDigits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(fractionDigits) : "—";
}

function metricValue(metrics: Record<string, unknown> | undefined, ...names: string[]) {
  if (!metrics) return undefined;
  for (const name of names) if (metrics[name] !== undefined) return metrics[name];
  return undefined;
}

function MetricChart({ title, points, field, color }: { title: string; points: MetricPoint[]; field: "loss" | "map50"; color: string }) {
  const values = points.map((point) => point[field]).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const selected = points.filter((point) => typeof point[field] === "number" && Number.isFinite(point[field]));
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const spread = max - min || Math.max(Math.abs(max) * 0.08, 0.01);
  const coordinates = selected.map((point, index) => {
    const x = selected.length < 2 ? 12 : 12 + (index / (selected.length - 1)) * 376;
    const normalized = ((point[field] as number) - min) / spread;
    return `${x},${92 - normalized * 72}`;
  }).join(" ");
  return (
    <div className="train-chart">
      <div className="train-chart-head"><strong>{title}</strong><span>{values.length ? formatMetric(values[values.length - 1]) : "Chưa có dữ liệu"}</span></div>
      <svg viewBox="0 0 400 112" role="img" aria-label={`${title} theo epoch`}>
        <path d="M12 92 H388 M12 56 H388 M12 20 H388" className="train-chart-grid" />
        {coordinates && <polyline points={coordinates} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />}
        {selected.length === 1 && <circle cx="12" cy="92" r="3" fill={color} />}
      </svg>
      <div className="train-chart-foot"><span>Epoch {selected[0]?.epoch ?? "—"}</span><span>Epoch {selected[selected.length - 1]?.epoch ?? "—"}</span></div>
    </div>
  );
}

export default function TrainingWorkspace({ initialDataYaml, onInitialDataYamlConsumed }: Props) {
  const [capabilities, setCapabilities] = useState<TrainingCapabilities | null>(null);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [dataYaml, setDataYaml] = useState("");
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [outputDir, setOutputDir] = useState("");
  const [sizes, setSizes] = useState<number[]>([320]);
  const [customSize, setCustomSize] = useState("");
  const [device, setDevice] = useState<TrainingDevice>("auto");
  const [baseModel, setBaseModel] = useState("yolo11n.pt");
  const [onError, setOnError] = useState<"stop" | "continue">("stop");
  const [settings, setSettings] = useState<TrainingSettings>({ epochs: 100, batch: 8, patience: 30, workers: 0, save_period: 10, seed: 42, lr0: null });
  const [lr0Input, setLr0Input] = useState("");
  const [loading, setLoading] = useState(true);
  const [inspecting, setInspecting] = useState(false);
  const [running, setRunning] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);
  const [activeProgress, setActiveProgress] = useState<TrainProgress | null>(null);
  const [operationError, setOperationError] = useState<string | null>(null);
  const [queueResult, setQueueResult] = useState<QueueResult | null>(null);
  const [history, setHistory] = useState<MetricPoint[]>([]);
  const activeRequestId = useRef<string | null>(null);

  const selectedParent = useMemo(
    () => runs.find((run) => run.checkpoint_path === baseModel) ?? null,
    [runs, baseModel],
  );

  const refreshRuns = useCallback(async () => {
    const response = await engineRequest<RunList>("list_training_runs");
    setRuns(response.runs);
  }, []);

  const inspectDatasetPath = useCallback(async (selected: string) => {
    setDataYaml(selected);
    setDataset(null);
    setOperationError(null);
    setInspecting(true);
    try {
      const report = await engineRequest<DatasetSummary>("inspect_training_dataset", { data_yaml: selected });
      setDataset(report);
      const datasetDir = selected.replace(/[\\/][^\\/]*$/, "");
      setOutputDir((current) => current || datasetDir.replace(/[\\/][^\\/]*$/, ""));
    } catch (reason) {
      setDataset(null);
      setOperationError(String(reason));
    } finally {
      setInspecting(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    void Promise.all([
      engineRequest<TrainingCapabilities>("training_capabilities"),
      engineRequest<RunList>("list_training_runs"),
    ]).then(([status, registry]) => {
      if (!active) return;
      setCapabilities(status);
      setRuns(registry.runs);
      setSettings({
        epochs: status.recommendation?.epochs ?? 100,
        batch: status.recommendation?.batch ?? 8,
        patience: status.recommendation?.patience ?? 30,
        workers: status.recommendation?.workers ?? 0,
        save_period: status.recommendation?.save_period ?? 10,
        seed: 42,
        lr0: null,
      });
      setLoading(false);
    }).catch((reason) => {
      if (active) {
        setOperationError(String(reason));
        setLoading(false);
      }
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!initialDataYaml) return;
    void inspectDatasetPath(initialDataYaml);
    onInitialDataYamlConsumed?.();
  }, [initialDataYaml, inspectDatasetPath, onInitialDataYamlConsumed]);

  useEffect(() => {
    let dispose: (() => void) | undefined;
    let active = true;
    void listen<{ id: string; progress: TrainProgress }>("engine-progress", (event) => {
      if (event.payload.id !== activeRequestId.current) return;
      const next = event.payload.progress;
      setActiveProgress((current) => ({ ...current, ...next }));
      if (next.stage === "epoch_completed" && next.run_id && next.epoch) {
        const loss = next.losses?.box_loss ?? next.losses?.train_box_loss ?? null;
        const map50 = metricValue(next.metrics, "metrics/mAP50(B)", "metrics/mAP50", "mAP50(B)");
        setHistory((current) => {
          const key = `${next.run_id}:${next.epoch}`;
          if (current.some((point) => point.key === key)) return current;
          return [...current, {
            key,
            epoch: next.epoch!,
            loss: typeof loss === "number" ? loss : null,
            map50: typeof map50 === "number" ? map50 : null,
          }].slice(-100);
        });
      }
    }).then((unlisten) => {
      if (active) dispose = unlisten;
      else unlisten();
    }).catch(() => undefined);
    return () => { active = false; dispose?.(); };
  }, []);

  async function chooseDataset() {
    try {
      const selected = await open({
        title: "Chọn data.yaml của dataset YOLO",
        multiple: false,
        directory: false,
        filters: [{ name: "YOLO dataset", extensions: ["yaml", "yml"] }],
      });
      if (typeof selected !== "string") return;
      await inspectDatasetPath(selected);
    } catch (reason) {
      setOperationError(String(reason));
    }
  }

  async function chooseOutputDir() {
    try {
      const selected = await open({ title: "Chọn thư mục lưu training", directory: true, multiple: false });
      if (typeof selected !== "string") return;
      setOutputDir(selected);
      if (capabilities) {
        const updated = await engineRequest<TrainingCapabilities>("training_capabilities", { output_dir: selected });
        setCapabilities(updated);
      }
    } catch (reason) {
      setOperationError(String(reason));
    }
  }

  function toggleSize(value: number) {
    setSizes((current) => current.includes(value) ? current.filter((size) => size !== value) : [...current, value].sort((a, b) => a - b));
  }

  function addCustomSize() {
    const value = Number(customSize);
    if (!Number.isInteger(value) || value < 1 || value > 2048) {
      setOperationError("imgsz phải là số nguyên từ 1 đến 2048 px.");
      return;
    }
    if (!sizes.includes(value)) setSizes((current) => [...current, value].sort((a, b) => a - b));
    setCustomSize("");
    setOperationError(null);
  }

  function applyRecommendation() {
    if (!capabilities?.recommendation) return;
    const suggestion = capabilities.recommendation;
    setDevice(suggestion.device.startsWith("cuda") ? "cuda" : suggestion.device as TrainingDevice);
    setSettings({
      epochs: suggestion.epochs,
      batch: suggestion.batch,
      patience: suggestion.patience,
      workers: suggestion.workers,
      save_period: suggestion.save_period,
      seed: 42,
      lr0: null,
    });
    setLr0Input("");
  }

  async function startQueue() {
    if (!capabilities?.available) {
      setOperationError("PyTorch và Ultralytics chưa sẵn sàng trong môi trường Python của FrameForge.");
      return;
    }
    if (!dataset?.valid || !dataYaml || !outputDir || sizes.length === 0) {
      setOperationError("Chọn dataset hợp lệ, thư mục đầu ra và ít nhất một imgsz.");
      return;
    }
    const stride = selectedParent?.stride || 32;
    const adjustments = sizes.filter((size) => Math.max(stride, Math.ceil(size / stride) * stride) !== size)
      .map((size) => ({ requested: size, effective: Math.max(stride, Math.ceil(size / stride) * stride) }));
    if (adjustments.length) {
      const detail = adjustments.map((item) => `${item.requested} → ${item.effective}`).join(", ");
      const accepted = await confirm(
        `YOLO11n dùng stride ${stride}. Kích thước ${detail} sẽ được làm tròn lên theo stride; config và kết quả sẽ lưu cả kích thước yêu cầu lẫn kích thước thực tế. Tiếp tục?`,
        { title: "Xác nhận kích thước train", kind: "warning" },
      );
      if (!accepted) return;
    }
    setOperationError(null);
    setQueueResult(null);
    setHistory([]);
    setRunning(true);
    setCancelRequested(false);
    setActiveProgress({ stage: "starting", total_jobs: sizes.length });
    const id = requestId();
    activeRequestId.current = id;
    try {
      const result = await invoke<QueueResult>("engine_request", {
        request: {
          id,
          command: "train_yolo_queue",
          arguments: {
            data_yaml: dataYaml,
            output_dir: outputDir,
            sizes,
            device,
            base_model: baseModel,
            on_error: onError,
            settings: {
              ...settings,
              lr0: lr0Input.trim() ? Number(lr0Input) : null,
            },
          },
        },
      });
      setQueueResult(result);
      setActiveProgress((current) => ({ ...current, stage: "queue_completed", status: result.status }));
      await refreshRuns();
    } catch (reason) {
      setOperationError(String(reason));
    } finally {
      activeRequestId.current = null;
      setRunning(false);
      setCancelRequested(false);
    }
  }

  async function requestCancel() {
    const id = activeRequestId.current;
    if (!id || cancelRequested) return;
    try {
      const accepted = await invoke<boolean>("cancel_engine_request", { requestId: id });
      if (accepted) {
        setCancelRequested(true);
        setOperationError(null);
      } else {
        setOperationError("Tác vụ đã kết thúc trước khi nhận yêu cầu hủy.");
      }
    } catch (reason) {
      setOperationError(String(reason));
    }
  }

  const progressPercent = activeProgress?.total_jobs && activeProgress.epochs_total
    ? Math.min(100, Math.round((((Math.max(0, (activeProgress.job_index ?? 1) - 1) + Math.max(0, (activeProgress.epoch ?? 0) - 1) / activeProgress.epochs_total) / activeProgress.total_jobs) * 100)))
    : activeProgress?.stage === "queue_completed" && activeProgress.status === "completed" ? 100 : null;
  const currentMetrics = activeProgress?.metrics;
  const currentRuntime = activeProgress?.runtime;
  const bestRuns = runs.filter((run) => run.checkpoint_usable && run.checkpoint_path);

  return (
    <main className="training-workspace" aria-label="Workspace huấn luyện YOLO11n">
      <header className="train-page-heading">
        <div>
          <span className="train-kicker"><Activity size={13} /> MODEL TRAINING</span>
          <h1>Huấn luyện YOLO11n</h1>
          <p>Train model thật trên dataset YOLO đã export. Mỗi imgsz là một run riêng, nạp lại cùng checkpoint gốc.</p>
        </div>
        <div className={`train-capability-badge ${loading ? "loading" : capabilities?.available ? "ready" : "missing"}`}>
          {loading ? <LoaderCircle className="spin" size={14} /> : capabilities?.available ? <Check size={14} /> : <AlertCircle size={14} />}
          <span>{loading ? "Đang kiểm tra" : capabilities?.available ? `Sẵn sàng · ${capabilities.preferred_device.toUpperCase()}` : "Thiếu dependency"}</span>
        </div>
      </header>

      <div className="train-content-grid">
        <section className="train-column" aria-label="Cấu hình training">
          {!capabilities?.available && !loading && <div className="train-dependency-note" role="status">
            <strong>Chưa thể train trong Python đang dùng.</strong>
            <span>Cài training extra trong môi trường repository, rồi mở lại ứng dụng:</span>
            <code>.venv/bin/python -m pip install -e &quot;engine[train]&quot;</code>
            <small>Windows: <code>.venv\Scripts\python.exe -m pip install -e &quot;engine[train]&quot;</code></small>
            {capabilities?.torch_error && <small>{capabilities.torch_error}</small>}
          </div>}

          <section className="train-card">
            <div className="train-card-heading"><div><Database size={15} /><h2>Dataset</h2></div><span>{dataset?.valid ? "Hợp lệ" : dataset ? "Cần xử lý" : "Chưa chọn"}</span></div>
            <div className="train-path-row">
              <span title={dataYaml || "Chưa chọn data.yaml"}>{dataYaml || "Chọn data.yaml của dataset đã export"}</span>
              <button className="train-icon-button" type="button" onClick={() => void chooseDataset()} disabled={inspecting || running} aria-label="Chọn data.yaml"><FolderOpen size={15} /></button>
            </div>
            {inspecting && <p className="train-inline-status"><LoaderCircle className="spin" size={13} /> Đang đọc ảnh, label và kiểm tra fingerprint…</p>}
            {dataset && <div className="train-dataset-summary">
              <span><strong>{dataset.total_images}</strong> ảnh</span>
              <span><strong>{dataset.total_instances}</strong> box</span>
              <span><strong>{dataset.class_names.length}</strong> class</span>
              <span>train {dataset.image_counts.train ?? 0} · val {dataset.image_counts.val ?? 0}</span>
            </div>}
            {dataset?.class_names.length ? <div className="train-class-list">{dataset.class_names.map((name) => <span key={name}>{name}<small>{dataset.class_instance_counts[name] ?? 0}</small></span>)}</div> : null}
            {dataset?.errors.map((message) => <p className="train-warning train-error" key={message}><AlertCircle size={12} />{message}</p>)}
            {dataset?.warnings.map((message) => <p className="train-warning" key={message}><CircleHelp size={12} />{message}</p>)}
            {dataset && <small className="train-fingerprint" title={dataset.dataset_fingerprint}>Dataset ID: {dataset.dataset_id ?? "external"} · SHA-256 {dataset.dataset_fingerprint.slice(0, 12)}…</small>}
          </section>

          <section className="train-card">
            <div className="train-card-heading"><div><Zap size={15} /><h2>Model và độ phân giải</h2></div><button className="train-text-button" type="button" onClick={() => void refreshRuns()} disabled={running}><RefreshCw size={12} /> Làm mới</button></div>
            <label className="train-field">Checkpoint khởi tạo
              <select value={baseModel} onChange={(event) => setBaseModel(event.target.value)} disabled={running}>
                <option value="yolo11n.pt">YOLO11n pretrained · yolo11n.pt</option>
                {bestRuns.map((run) => <option value={run.checkpoint_path!} key={run.run_id}>YOLO11n · {run.effective_imgsz ?? run.requested_imgsz}px · {run.run_id.slice(0, 8)}{run.checkpoint_checksum_matches === false ? " · checksum mismatch" : ""}</option>)}
              </select>
              <small>{selectedParent ? "Tạo fine-tune mới; checkpoint cha được giữ nguyên." : "Mỗi độ phân giải nạp lại cùng model pretrained, không kế thừa weights của job trước."}</small>
            </label>
            <div className="train-size-heading"><span>Kích thước train · imgsz</span><small>Stride YOLO11n: 32 px</small></div>
            <div className="train-size-grid" role="group" aria-label="Chọn nhiều kích thước train">
              {IMAGE_SIZE_PRESETS.map((size) => <button type="button" key={size} className={sizes.includes(size) ? "selected" : ""} aria-pressed={sizes.includes(size)} onClick={() => toggleSize(size)} disabled={running}>{size}</button>)}
              {sizes.filter((size) => !IMAGE_SIZE_PRESETS.includes(size)).map((size) => <button type="button" key={size} className="selected custom" aria-pressed="true" onClick={() => toggleSize(size)} disabled={running} title="Gỡ kích thước tùy chỉnh">{size} ×</button>)}
            </div>
            <form className="train-custom-size" onSubmit={(event) => { event.preventDefault(); addCustomSize(); }}>
              <input type="number" min="1" max="2048" step="1" value={customSize} onChange={(event) => setCustomSize(event.target.value)} placeholder="Kích thước tùy chỉnh" aria-label="Kích thước imgsz tùy chỉnh" disabled={running} />
              <button className="train-secondary-button" type="submit" disabled={running || !customSize}><Plus size={13} /> Thêm</button>
            </form>
          </section>

          <section className="train-card">
            <div className="train-card-heading"><div><Cpu size={15} /><h2>Thiết bị và cấu hình</h2></div><button className="train-text-button" type="button" onClick={applyRecommendation} disabled={!capabilities?.available || running}>Tự động đề xuất</button></div>
            <label className="train-field">Thiết bị
              <select value={device} onChange={(event) => setDevice(event.target.value as TrainingDevice)} disabled={running}>
                <option value="auto">Tự động · {capabilities?.preferred_device ?? "CPU"}</option>
                <option value="mps" disabled={!capabilities?.devices.mps}>Apple MPS{capabilities?.device_details.mps_device_name ? ` · ${capabilities.device_details.mps_device_name}` : ""}</option>
                <option value="cuda" disabled={!capabilities?.devices.cuda}>NVIDIA CUDA</option>
                <option value="cpu">CPU</option>
              </select>
            </label>
            <div className="train-settings-grid">
              <label className="train-field">Epoch<input type="number" min="1" max="10000" value={settings.epochs} onChange={(event) => setSettings((current) => ({ ...current, epochs: Number(event.target.value) }))} disabled={running} /></label>
              <label className="train-field">Batch<input type="number" min="1" max="512" value={settings.batch} onChange={(event) => setSettings((current) => ({ ...current, batch: Number(event.target.value) }))} disabled={running} /></label>
              <label className="train-field">Patience <small>0 = chạy đủ số epoch đã đặt</small><input type="number" min="0" max="1000" value={settings.patience} onChange={(event) => setSettings((current) => ({ ...current, patience: Number(event.target.value) }))} disabled={running} /></label>
              <label className="train-field">Workers<input type="number" min="0" max="64" value={settings.workers} onChange={(event) => setSettings((current) => ({ ...current, workers: Number(event.target.value) }))} disabled={running} /></label>
              <label className="train-field">Learning rate <small>trống = mặc định framework</small><input type="number" min="0.000001" max="1" step="0.0001" value={lr0Input} onChange={(event) => setLr0Input(event.target.value)} placeholder="Mặc định" disabled={running} /></label>
              <label className="train-field">Khi job lỗi<select value={onError} onChange={(event) => setOnError(event.target.value as "stop" | "continue")} disabled={running}><option value="stop">Dừng queue</option><option value="continue">Tiếp tục job sau</option></select></label>
            </div>
            <div className="train-device-facts">
              <span>{capabilities?.hardware.processor || "CPU chưa rõ"} · {capabilities?.hardware.logical_cpu_count ?? "—"} luồng</span>
              <span>RAM {formatBytes(capabilities?.hardware.memory_available_bytes)} còn trống / {formatBytes(capabilities?.hardware.memory_total_bytes)}</span>
              <span>Dung lượng đĩa {formatBytes(capabilities?.hardware.disk_free_bytes)} còn trống</span>
            </div>
            {capabilities?.recommendation?.reasons.map((reason) => <small className="train-recommendation-reason" key={reason}>{reason}</small>)}
          </section>

          <section className="train-card train-output-card">
            <div className="train-card-heading"><div><FolderOpen size={15} /><h2>Thư mục kết quả</h2></div></div>
            <div className="train-path-row">
              <span title={outputDir || "Chưa chọn thư mục"}>{outputDir || "Chọn thư mục lưu checkpoint và log"}</span>
              <button className="train-icon-button" type="button" onClick={() => void chooseOutputDir()} disabled={running} aria-label="Chọn thư mục kết quả"><FolderOpen size={15} /></button>
            </div>
            <small>Mỗi queue tạo thư mục mới; không ghi đè checkpoint cũ.</small>
          </section>
        </section>

        <section className="train-column train-monitor-column" aria-label="Tiến độ và checkpoint">
          <section className="train-card train-run-card">
            <div className="train-card-heading"><div><Activity size={15} /><h2>Tiến độ thực tế</h2></div><span className={`train-run-status ${running ? "running" : queueResult?.status ?? "idle"}`}>{running ? "Đang chạy" : queueResult?.status ?? "Sẵn sàng"}</span></div>
            <div className="train-run-meta">
              <span><small>Model</small><strong>YOLO11n</strong></span>
              <span><small>Độ phân giải</small><strong>{activeProgress?.effective_imgsz ? `${activeProgress.effective_imgsz} px` : "—"}</strong></span>
              <span><small>Thiết bị thật</small><strong>{activeProgress?.device_actual ?? activeProgress?.device_selected ?? "—"}</strong></span>
              <span><small>Dataset</small><strong title={dataset?.dataset_id ?? ""}>{dataset?.dataset_id?.slice(0, 8) ?? "—"}</strong></span>
            </div>
            <div className="train-progress-label">
              <span>{activeProgress?.stage === "batch_progress" ? `Epoch ${activeProgress.epoch ?? "—"}/${activeProgress.epochs_total ?? "—"} · Batch ${activeProgress.batch ?? "—"}/${activeProgress.batches_total ?? "—"}` : activeProgress?.stage === "epoch_completed" ? `Đã xong epoch ${activeProgress.epoch ?? "—"}/${activeProgress.epochs_total ?? "—"}` : activeProgress?.stage === "queue_completed" ? "Queue hoàn tất" : activeProgress?.stage === "starting" ? "Đang khởi chạy queue…" : "Chưa có tác vụ train"}</span>
              <span>{progressPercent == null ? "—" : `${progressPercent}%`}</span>
            </div>
            <div className="train-progress-track" role="progressbar" aria-label="Tiến độ queue" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progressPercent ?? 0}><span style={{ width: `${progressPercent ?? 0}%` }} /></div>
            <div className="train-runtime-grid">
              <span><small>Đã chạy</small><strong>{formatDuration(activeProgress?.elapsed_seconds)}</strong></span>
              <span><small>ETA ước tính</small><strong>{formatDuration(activeProgress?.estimated_remaining_seconds)}</strong></span>
              <span><small>Ảnh / giây</small><strong>{formatMetric(activeProgress?.images_per_second, 1)}</strong></span>
              <span><small>Giây / batch</small><strong>{formatMetric(activeProgress?.seconds_per_batch, 2)}</strong></span>
              <span><small>CPU process</small><strong>{currentRuntime?.cpu_percent == null ? "—" : `${formatMetric(currentRuntime.cpu_percent, 1)}%`}</strong></span>
              <span><small>RAM process</small><strong>{formatBytes(currentRuntime?.process_memory_bytes)}</strong></span>
              <span><small>MPS/CUDA memory</small><strong>{formatBytes(currentRuntime?.gpu_memory_allocated_bytes)}</strong></span>
              <span><small>GPU utilization</small><strong>{currentRuntime?.gpu_utilization_percent == null ? "Chưa có số đo" : `${formatMetric(currentRuntime.gpu_utilization_percent, 1)}%`}</strong></span>
            </div>
            <div className="train-current-metrics">
              {[["Precision", "metrics/precision(B)", "metrics/precision"], ["Recall", "metrics/recall(B)", "metrics/recall"], ["mAP50", "metrics/mAP50(B)", "metrics/mAP50"], ["mAP50–95", "metrics/mAP50-95(B)", "metrics/mAP50-95"]].map(([label, ...keys]) => <span key={label}><small>{label}</small><strong>{formatMetric(metricValue(currentMetrics, ...keys))}</strong></span>)}
            </div>
            {activeProgress?.stage === "cancel_at_checkpoint" || cancelRequested ? <p className="train-inline-status"><LoaderCircle className="spin" size={13} /> Đang chờ hết epoch và lưu checkpoint để dừng an toàn…</p> : null}
            {activeProgress?.stage === "dataset_warnings" && activeProgress.warnings?.map((warning) => <p className="train-warning" key={warning}><CircleHelp size={12} />{warning}</p>)}
            <div className="train-action-row">
              {!running ? <button className="train-primary-button" type="button" onClick={() => void startQueue()} disabled={loading || inspecting || !dataset?.valid || !outputDir || !sizes.length || !capabilities?.available}><Play size={14} fill="currentColor" /> Bắt đầu queue · {sizes.length} run</button>
                : <button className="train-cancel-button" type="button" onClick={() => void requestCancel()} disabled={cancelRequested}><Square size={12} fill="currentColor" /> {cancelRequested ? "Đang dừng an toàn" : "Dừng sau epoch"}</button>}
              <span>{sizes.length} run tuần tự · độc lập từ {selectedParent ? "checkpoint đã chọn" : "yolo11n.pt"}</span>
            </div>
          </section>

          <section className="train-charts-grid">
            <MetricChart title="Box loss · thấp hơn thường tốt hơn" points={history} field="loss" color="#f0b365" />
            <MetricChart title="mAP50 · validation thực tế" points={history} field="map50" color="#68b996" />
          </section>

          {operationError && <div className="train-operation-error" role="alert"><AlertCircle size={14} /><span>{operationError}</span></div>}
          {queueResult && <section className="train-card">
            <div className="train-card-heading"><div><Check size={15} /><h2>Kết quả queue</h2></div><span>{queueResult.completed_count} hoàn tất · {queueResult.failed_count} lỗi</span></div>
            <small className="train-fingerprint">Queue {queueResult.queue_id} · {queueResult.queue_root}</small>
            <div className="train-result-list">{queueResult.jobs.map((job, index) => <article className="train-result-row" key={job.run_id ?? `${job.requested_imgsz}-${index}`}>
              <div><strong>{job.effective_imgsz ?? job.requested_imgsz}px</strong><span>{job.status}</span></div>
              <small>{job.device_actual ?? "—"} · mAP50 {formatMetric(metricValue(job.metrics, "metrics/mAP50(B)", "metrics/mAP50"))} · {formatDuration(job.elapsed_seconds)}</small>
              {job.best_checkpoint && <code title={job.best_checkpoint}>{job.best_checkpoint}</code>}
              {job.error && <small className="train-error-text">{job.error}</small>}
            </article>)}</div>
          </section>}

          <section className="train-card train-registry-card">
            <div className="train-card-heading"><div><Database size={15} /><h2>Checkpoint đã đăng ký</h2></div><button className="train-text-button" type="button" onClick={() => void refreshRuns()} disabled={running}><RefreshCw size={12} /> Làm mới</button></div>
            <p className="train-registry-note">Có {bestRuns.length} checkpoint nội bộ đủ điều kiện fine-tune. Run quét từ thư mục ngoài được lưu metadata riêng và không nạp checkpoint trước khi xác minh nguồn.</p>
            {runs.length === 0 ? <small className="train-empty-note">Chưa có checkpoint hoặc run lịch sử.</small> : <div className="train-result-list">{runs.slice(0, 8).map((run) => {
              const imageSize = run.training_imgsz ?? (run.requested_imgsz > 0 ? run.requested_imgsz : null);
              const imageSizeLabel = Array.isArray(imageSize) ? `${imageSize.join("×")} px` : imageSize ? `${imageSize} px` : "imgsz chưa ghi";
              return <article className="train-result-row" key={run.run_id}>
              <div><strong>{run.source === "external_discovery" ? run.name ?? "Run bên ngoài" : `YOLO11n · ${run.effective_imgsz ?? run.requested_imgsz}px`}</strong><span className={run.checkpoint_usable ? "available" : "unavailable"}>{run.source === "external_discovery" ? "Metadata · checkpoint chưa tin cậy" : run.checkpoint_usable ? "Sẵn sàng" : "Thiếu hoặc checksum đổi"}</span></div>
              <small>{run.source === "external_discovery" ? `${run.architecture_claimed ?? "Model chưa rõ"} · ${run.task_claimed ?? "task chưa rõ"} · ${imageSizeLabel}` : `${run.status} · ${run.device_actual ?? "thiết bị chưa ghi"}`} · {run.run_id.slice(0, 8)}</small>
              {run.checkpoint_path && <code title={run.checkpoint_path}>{run.checkpoint_path}</code>}
            </article>;
            })}</div>}
          </section>
        </section>
      </div>
      <footer className="train-footer"><span><ChevronDown size={12} /> ETA chỉ xuất hiện sau khi có thời gian epoch đo được.</span><span>PyTorch {capabilities?.torch_version ?? "chưa cài"} · Ultralytics {capabilities?.ultralytics_version ?? "chưa cài"}</span></footer>
    </main>
  );
}
