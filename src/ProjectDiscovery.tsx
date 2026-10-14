import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { confirm, open } from "@tauri-apps/plugin-dialog";
import { Activity, AlertCircle, Database, FolderOpen, ShieldAlert, X } from "lucide-react";

type DiscoveryProgress = {
  stage: string;
  files_seen: number;
  directories_scanned: number;
  datasets_found: number;
  runs_found: number;
};

type DiscoverySplit = {
  name: string;
  status: string;
  image_count: number | null;
  observed_image_count: number;
  labeled_count: number | null;
};

type DiscoveryDataset = {
  dataset_id: string;
  path: string;
  name: string;
  status: string;
  class_names: string[] | null;
  class_count: number | null;
  splits: DiscoverySplit[];
  warnings: string[];
};

type DiscoveryCheckpoint = {
  path: string;
  name: string;
  kind: string;
  size_bytes: number | null;
  compatibility: string;
  trust_required: boolean;
};

type DiscoveryRun = {
  experiment_id: string;
  source_dir: string;
  name: string;
  status: string;
  completion_state: string;
  architecture_claimed: string | null;
  task_claimed: string | null;
  dataset_reference: string | null;
  dataset_relationship: string;
  training_imgsz: number | number[] | null;
  requested_epochs: number | null;
  last_logged_epoch: number | null;
  metrics_status: string;
  metrics_history: Record<string, number | null>[];
  best_metrics: Record<string, number>;
  checkpoints: DiscoveryCheckpoint[];
  warnings: string[];
};

type DiscoveryResult = {
  root: string;
  status: "complete" | "partial" | "cancelled";
  files_seen: number;
  directories_scanned: number;
  symlinks_skipped: number;
  permission_errors: number;
  datasets: DiscoveryDataset[];
  runs: DiscoveryRun[];
  exported_models: { path: string; name: string; format: string; size_bytes: number | null }[];
  frameforge_projects: { path: string; name: string }[];
  warnings: string[];
};

type Props = { onUseDataset(path: string): void };

function requestId() {
  return globalThis.crypto?.randomUUID?.() ?? `discovery-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function bytes(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "Dung lượng chưa rõ";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024;
  let unit = units[0];
  for (let index = 1; size >= 1024 && index < units.length; index += 1) {
    size /= 1024;
    unit = units[index];
  }
  return `${size.toFixed(size >= 100 ? 0 : 1)} ${unit}`;
}

function stageLabel(stage?: string) {
  switch (stage) {
    case "discovery_started": return "Chuẩn bị quét thư mục…";
    case "scanning": return "Đang đọc cây thư mục…";
    case "analyzing_datasets": return "Đang lập chỉ mục dataset…";
    case "analyzing_runs": return "Đang đọc cấu hình và metric…";
    case "discovery_complete": return "Quét hoàn tất";
    case "discovery_partial": return "Quét xong một phần";
    case "discovery_cancelled": return "Đã hủy quét";
    default: return "Đang quét…";
  }
}

function datasetStatus(status: string) {
  if (status === "candidate") return "Có thể dùng";
  if (status === "blocked_external") return "Cần nối lại đường dẫn";
  if (status === "invalid_metadata") return "YAML không hợp lệ";
  return "Cần xem lại";
}

function metric(value?: number) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
}

export default function ProjectDiscovery({ onUseDataset }: Props) {
  const [maxDepth, setMaxDepth] = useState(8);
  const [maxFiles, setMaxFiles] = useState(50_000);
  const [ignoredDirs, setIgnoredDirs] = useState(".git, .venv, node_modules, __pycache__, .pytest_cache, target, dist, build");
  const [scanning, setScanning] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);
  const [progress, setProgress] = useState<DiscoveryProgress | null>(null);
  const [result, setResult] = useState<DiscoveryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [usingDatasetId, setUsingDatasetId] = useState<string | null>(null);
  const [importingCheckpointPath, setImportingCheckpointPath] = useState<string | null>(null);
  const [importedCheckpointPaths, setImportedCheckpointPaths] = useState<Set<string>>(() => new Set());
  const [importMessage, setImportMessage] = useState<string | null>(null);
  const [savingRunId, setSavingRunId] = useState<string | null>(null);
  const [savedRunIds, setSavedRunIds] = useState<Set<string>>(() => new Set());
  const requestRef = useRef<string | null>(null);

  useEffect(() => {
    let active = true;
    let dispose: (() => void) | undefined;
    void listen<{ id: string; progress: DiscoveryProgress }>("engine-progress", (event) => {
      if (event.payload.id === requestRef.current) setProgress(event.payload.progress);
    }).then((unlisten) => {
      if (active) dispose = unlisten;
      else unlisten();
    }).catch(() => undefined);
    return () => { active = false; dispose?.(); };
  }, []);

  useEffect(() => () => {
    const id = requestRef.current;
    if (id) void invoke<boolean>("cancel_engine_request", { requestId: id }).catch(() => undefined);
  }, []);

  async function scanFolder() {
    setError(null);
    if (!Number.isInteger(maxDepth) || maxDepth < 1 || maxDepth > 16) {
      setError("Độ sâu quét phải là số nguyên từ 1 đến 16.");
      return;
    }
    if (!Number.isInteger(maxFiles) || maxFiles < 1 || maxFiles > 200_000) {
      setError("Giới hạn tệp phải là số nguyên từ 1 đến 200000.");
      return;
    }
    try {
      const selected = await open({ title: "Chọn thư mục làm việc cần nhập", directory: true, multiple: false });
      if (typeof selected !== "string") return;
      const id = requestId();
      requestRef.current = id;
      setScanning(true);
      setCancelRequested(false);
      setResult(null);
      setProgress({ stage: "discovery_started", files_seen: 0, directories_scanned: 0, datasets_found: 0, runs_found: 0 });
      const ignored = ignoredDirs.split(",").map((name) => name.trim()).filter(Boolean);
      const next = await invoke<DiscoveryResult>("engine_request", {
        request: {
          id,
          command: "scan_project_folder",
          arguments: { path: selected, max_depth: maxDepth, max_files: maxFiles, ignored_dirs: ignored },
        },
      });
      setResult(next);
    } catch (reason) {
      setError(typeof reason === "string" ? reason : String(reason));
    } finally {
      requestRef.current = null;
      setScanning(false);
      setCancelRequested(false);
    }
  }

  async function cancelScan() {
    const id = requestRef.current;
    if (!id || cancelRequested) return;
    try {
      const accepted = await invoke<boolean>("cancel_engine_request", { requestId: id });
      if (accepted) setCancelRequested(true);
      else setError("Tác vụ quét đã kết thúc trước khi nhận yêu cầu hủy.");
    } catch (reason) {
      setError(typeof reason === "string" ? reason : String(reason));
    }
  }

  async function saveRun(run: DiscoveryRun) {
    if (!result || savingRunId) return;
    setSavingRunId(run.experiment_id);
    setError(null);
    try {
      await invoke("engine_request", {
        request: {
          id: requestId(),
          command: "register_discovered_run",
          arguments: { discovery_root: result.root, run },
        },
      });
      setSavedRunIds((current) => new Set(current).add(run.experiment_id));
    } catch (reason) {
      setError(typeof reason === "string" ? reason : String(reason));
    } finally {
      setSavingRunId(null);
    }
  }

  async function useDataset(dataset: DiscoveryDataset) {
    if (!result || usingDatasetId) return;
    setUsingDatasetId(dataset.dataset_id);
    setError(null);
    setImportMessage(null);
    try {
      const prepared = await invoke<{
        data_yaml: string;
        class_names: string[];
        paths_rebased: boolean;
      }>("engine_request", {
        request: {
          id: requestId(),
          command: "prepare_discovered_dataset",
          arguments: { discovery_root: result.root, data_yaml: dataset.path },
        },
      });
      setImportMessage(prepared.paths_rebased
        ? "Đã nối lại đường dẫn và chuyển " + dataset.name + " sang bước kiểm tra dataset."
        : "Đã chọn " + dataset.name + "; FrameForge sẽ kiểm tra ảnh và nhãn trước khi train.");
      onUseDataset(prepared.data_yaml);
    } catch (reason) {
      setError(typeof reason === "string" ? reason : String(reason));
    } finally {
      setUsingDatasetId(null);
    }
  }

  async function importCheckpoint(checkpoint: DiscoveryCheckpoint) {
    if (importingCheckpointPath || importedCheckpointPaths.has(checkpoint.path)) return;
    setError(null);
    setImportMessage(null);
    let trusted = false;
    try {
      trusted = await confirm(
        "Checkpoint " + checkpoint.name + " là tệp .pt có thể chứa dữ liệu thực thi mã khi được nạp. Chỉ tiếp tục nếu bạn tự tạo tệp hoặc tin tưởng nguồn của thư mục này. FrameForge sẽ kiểm tra task detection, kiến trúc YOLO11n, class và checksum, rồi sao chép checkpoint vào thư mục quản lý riêng.",
        { title: "Xác nhận tin cậy checkpoint", kind: "warning", okLabel: "Tin cậy và nhập", cancelLabel: "Hủy" },
      );
    } catch (reason) {
      setError(typeof reason === "string" ? reason : String(reason));
      return;
    }
    if (!trusted) return;
    setImportingCheckpointPath(checkpoint.path);
    try {
      const imported = await invoke<{
        architecture: string;
        display_name: string;
        class_names: string[];
        requested_imgsz: number;
        imgsz_source: "checkpoint_metadata" | "run_args_yaml" | "ultralytics_default";
      }>("engine_request", {
        request: {
          id: requestId(),
          command: "import_yolo_checkpoint",
          arguments: { path: checkpoint.path, trust_source: true },
        },
      });
      setImportedCheckpointPaths((current) => new Set(current).add(checkpoint.path));
      const sizeMessage = imported.imgsz_source === "checkpoint_metadata"
        ? `imgsz ${imported.requested_imgsz}px từ checkpoint`
        : imported.imgsz_source === "run_args_yaml"
          ? `imgsz ${imported.requested_imgsz}px từ args.yaml của run`
          : `checkpoint và args.yaml không ghi imgsz; dùng mặc định ${imported.requested_imgsz}px`;
      setImportMessage("Đã nhập " + imported.display_name + " (" + imported.architecture + "; " + sizeMessage + "; " + imported.class_names.join(", ") + "). Checkpoint chỉ dùng được khi class khớp project.");
    } catch (reason) {
      setError(typeof reason === "string" ? reason : String(reason));
    } finally {
      setImportingCheckpointPath(null);
    }
  }

  const checkpointCount = result?.runs.reduce((total, run) => total + run.checkpoints.length, 0) ?? 0;

  return (
    <section className="panel dataset-card discovery-card" aria-label="Nhập thư mục làm việc">
      <div className="dataset-card-heading discovery-heading">
        <div>
          <h2>Nhập thư mục làm việc</h2>
          <span>Tìm data.yaml, dataset, run và model trong thư mục đã chuyển từ máy khác.</span>
        </div>
        {!scanning ? <button className="dataset-secondary-button" type="button" onClick={() => void scanFolder()}><FolderOpen size={13} /> Chọn thư mục</button>
          : <button className="dataset-secondary-button" type="button" onClick={() => void cancelScan()} disabled={cancelRequested}><X size={13} /> {cancelRequested ? "Đang hủy…" : "Hủy quét"}</button>}
      </div>

      <div className="discovery-limits">
        <label>Độ sâu tối đa<input type="number" min="1" max="16" value={maxDepth} onChange={(event) => setMaxDepth(Number(event.target.value))} disabled={scanning} /></label>
        <label>Số tệp tối đa<input type="number" min="1" max="200000" step="1000" value={maxFiles} onChange={(event) => setMaxFiles(Number(event.target.value))} disabled={scanning} /></label>
        <label className="discovery-ignored">Bỏ qua thư mục<input value={ignoredDirs} onChange={(event) => setIgnoredDirs(event.target.value)} disabled={scanning} title="Nhập tên thư mục, cách nhau bằng dấu phẩy" /></label>
      </div>
      <p className="discovery-safety"><ShieldAlert size={12} /> Ảnh dataset được dùng tại chỗ; FrameForge tạo data.yaml riêng với đường dẫn đã khôi phục. Checkpoint .pt chỉ được nạp sau xác nhận tin cậy và được sao chép vào kho riêng.</p>

      {scanning && <div className="discovery-progress" role="status" aria-live="polite">
        <div className="discovery-progress-track" role="progressbar" aria-label="Tiến độ quét"><span /></div>
        <div><strong>{stageLabel(progress?.stage)}</strong><span>{progress?.files_seen ?? 0} tệp · {progress?.directories_scanned ?? 0} thư mục · {progress?.datasets_found ?? 0} dataset · {progress?.runs_found ?? 0} run</span></div>
      </div>}
      {error && <div className="dataset-message error" role="alert"><AlertCircle size={14} /><span>{error}</span></div>}

      {result && <div className="discovery-results" aria-live="polite">
        {importMessage && <div className="dataset-message success" role="status"><span>{importMessage}</span></div>}
        <div className={`discovery-summary ${result.status}`}>
          <div><strong>{result.status === "complete" ? "Quét hoàn tất" : result.status === "cancelled" ? "Đã hủy quét" : "Quét một phần"}</strong><span>{result.files_seen} tệp · {result.directories_scanned} thư mục · {result.root}</span></div>
          <div className="discovery-counts"><span><strong>{result.datasets.length}</strong> dataset</span><span><strong>{result.runs.length}</strong> run</span><span><strong>{checkpointCount}</strong> checkpoint</span><span><strong>{result.exported_models.length}</strong> model export</span></div>
        </div>

        {result.datasets.length > 0 && <section className="discovery-group">
          <h3><Database size={13} /> Dataset tìm thấy</h3>
          {result.datasets.map((dataset) => <article className="discovery-row" key={dataset.dataset_id}>
            <div className="discovery-row-main"><strong>{dataset.name}</strong><span className={`discovery-state ${dataset.status}`}>{datasetStatus(dataset.status)}</span></div>
            <code title={dataset.path}>{dataset.path}</code>
            <div className="discovery-row-detail">
              <span>{dataset.class_names?.join(", ") ?? `${dataset.class_count ?? "?"} class`}</span>
              <span>{dataset.splits.map((split) => `${split.name}: ${split.image_count ?? `≥${split.observed_image_count}`} ảnh`).join(" · ") || "Chưa có split"}</span>
              {dataset.status !== "invalid_metadata" && <button className="dataset-text-action" type="button" onClick={() => void useDataset(dataset)} disabled={usingDatasetId !== null}>
                {usingDatasetId === dataset.dataset_id ? "Đang nối đường dẫn…" : dataset.status === "blocked_external" ? "Khôi phục và dùng dataset" : "Kiểm tra để huấn luyện"}
              </button>}
            </div>
            {dataset.warnings.map((warning) => <small className="discovery-warning" key={warning}>{warning}</small>)}
          </article>)}
        </section>}

        {result.runs.length > 0 && <section className="discovery-group">
          <h3><Activity size={13} /> Training run tìm thấy</h3>
          {result.runs.map((run) => <article className="discovery-row" key={run.experiment_id}>
            <div className="discovery-row-main"><strong>{run.name}</strong><span>{run.training_imgsz ? `imgsz ${Array.isArray(run.training_imgsz) ? run.training_imgsz.join("×") : run.training_imgsz}` : "imgsz chưa ghi"}</span></div>
            <code title={run.source_dir}>{run.source_dir}</code>
            <div className="discovery-row-detail">
              <span>{run.architecture_claimed ?? "Model chưa ghi"} · {run.task_claimed ?? "task chưa ghi"}</span>
              <span>{run.last_logged_epoch == null ? "Chưa có epoch log" : `Epoch cuối ${run.last_logged_epoch + 1}${run.requested_epochs ? `/${run.requested_epochs}` : ""}`}</span>
              <span>mAP50 {metric(run.best_metrics.map50)} · mAP50–95 {metric(run.best_metrics.map50_95)}</span>
              <button className="dataset-text-action" type="button" onClick={() => void saveRun(run)} disabled={savingRunId !== null || savedRunIds.has(run.experiment_id)}>
                {savedRunIds.has(run.experiment_id) ? "Đã lưu metadata" : savingRunId === run.experiment_id ? "Đang lưu…" : "Lưu run vào registry"}
              </button>
            </div>
            {run.checkpoints.map((checkpoint) => <div className="discovery-checkpoint" key={checkpoint.path}>
              <span>{checkpoint.kind === "best" ? "best" : checkpoint.kind === "last" ? "last" : "checkpoint"} · {checkpoint.name}</span>
              <code title={checkpoint.path}>{checkpoint.path}</code>
              <small>{bytes(checkpoint.size_bytes)} · Chưa xác minh tương thích · cần xác nhận nguồn tin cậy trước khi nạp</small>
              <button className="dataset-text-action" type="button" onClick={() => void importCheckpoint(checkpoint)} disabled={importingCheckpointPath !== null || importedCheckpointPaths.has(checkpoint.path) || (checkpoint.size_bytes != null && checkpoint.size_bytes > 2 * 1024 * 1024 * 1024)}>
                {importedCheckpointPaths.has(checkpoint.path) ? "Đã nhập checkpoint" : importingCheckpointPath === checkpoint.path ? "Đang xác minh và nhập…" : checkpoint.size_bytes != null && checkpoint.size_bytes > 2 * 1024 * 1024 * 1024 ? "Vượt giới hạn 2 GiB" : "Xác nhận và nhập checkpoint"}
              </button>
            </div>)}
            {run.warnings.map((warning) => <small className="discovery-warning" key={warning}>{warning}</small>)}
          </article>)}
        </section>}

        {result.exported_models.length > 0 && <section className="discovery-group">
          <h3>Model export tìm thấy</h3>
          {result.exported_models.map((model) => <article className="discovery-row" key={model.path}>
            <div className="discovery-row-main"><strong>{model.name}</strong><span>{model.format.toUpperCase()} · {bytes(model.size_bytes)}</span></div>
            <code title={model.path}>{model.path}</code>
          </article>)}
        </section>}

        {result.datasets.length === 0 && result.runs.length === 0 && result.exported_models.length === 0 && result.frameforge_projects.length === 0 && <p className="discovery-empty"><FolderOpen size={15} /> Chưa tìm thấy dataset, training run hoặc model export được hỗ trợ trong phạm vi quét.</p>}
        {result.warnings.length > 0 && <div className="discovery-warning-block"><strong>Ghi chú quét</strong><ul>
          {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul></div>}
        {result.frameforge_projects.length > 0 && <small className="discovery-projects">Project FrameForge: {result.frameforge_projects.map((project) => project.name).join(", ")}</small>}
      </div>}
    </section>
  );
}
