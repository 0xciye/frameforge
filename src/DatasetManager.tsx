import { useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { AlertCircle, CheckCircle2, Database, Download, FolderOpen, LoaderCircle, RefreshCw, ShieldAlert } from "lucide-react";
import ProjectDiscovery from "./ProjectDiscovery";
import "./DatasetManager.css";

type DatasetConfig = {
  splitMode: "source" | "segment";
  train: number;
  val: number;
  test: number;
  seed: number;
  segmentSeconds: number;
  negativeRatio: number;
  negativeRatioCustom: boolean;
  freezeEvaluationSet: boolean;
};

type DatasetReport = {
  valid: boolean;
  project_name: string;
  image_count: number;
  included_image_count: number;
  positive_image_count: number;
  reviewed_negative_count: number;
  negative_ratio_requested: number;
  negative_ratio_actual: number;
  negative_sample_target_count: number;
  negative_samples_available_count: number;
  negative_samples_excluded_count: number;
  unreviewed_annotation_excluded_count: number;
  candidate_negative_excluded_count: number;
  annotation_count: number;
  class_distribution: { class_id: number; name: string; box_count: number }[];
  duplicate_basename_groups: number;
  exact_duplicate_count: number;
  split_counts: Record<"train" | "val" | "test", number>;
  split_group_counts: Record<"train" | "val" | "test", number>;
  split_mode: "source" | "segment";
  ratios: Record<"train" | "val" | "test", number>;
  seed: number;
  segment_seconds: number;
  new_image_count: number;
  replayed_image_count: number;
  fixed_validation_count: number;
  frozen_test_count: number;
  errors: string[];
  warnings: string[];
};

type ExportResult = {
  dataset_id: string;
  version: number;
  parent_dataset_id: string | null;
  replay_strategy: string;
  path: string;
  data_yaml: string;
  image_count: number;
  new_image_count: number;
  replayed_image_count: number;
  fixed_validation_count: number;
  frozen_test_count: number;
  split_counts: Record<"train" | "val" | "test", number>;
  warnings: string[];
};

type Props = {
  projectPath: string | null;
  imageCount: number;
  onUseDiscoveredDataset(path: string): void;
};

const DEFAULT_CONFIG: DatasetConfig = {
  splitMode: "source",
  train: 80,
  val: 10,
  test: 10,
  seed: 42,
  segmentSeconds: 30,
  negativeRatio: 10,
  negativeRatioCustom: false,
  freezeEvaluationSet: false,
};

function requestId() {
  return globalThis.crypto?.randomUUID?.() ?? `dataset-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function errorText(error: unknown) {
  return typeof error === "string" ? error : String(error);
}

export default function DatasetManager({ projectPath, imageCount, onUseDiscoveredDataset }: Props) {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [outputDirectory, setOutputDirectory] = useState("");
  const [parentManifestPath, setParentManifestPath] = useState("");
  const [replayStrategy, setReplayStrategy] = useState<"all" | "balanced" | "new">("all");
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [validatedKey, setValidatedKey] = useState("");
  const [exported, setExported] = useState<ExportResult | null>(null);
  const [busy, setBusy] = useState<"validate" | "export" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const configKey = useMemo(() => JSON.stringify({ projectPath, config, parentManifestPath, replayStrategy }), [projectPath, config, parentManifestPath, replayStrategy]);
  const validationIsCurrent = Boolean(report && validatedKey === configKey);
  const ratioTotal = config.train + config.val + config.test;
  const negativeRatioValid = Number.isFinite(config.negativeRatio) && config.negativeRatio >= 0 && config.negativeRatio <= 100;
  const ratioValid = ratioTotal === 100 && config.train > 0 && negativeRatioValid;

  function updateConfig(patch: Partial<DatasetConfig>) {
    setConfig((current) => ({ ...current, ...patch }));
    setReport(null);
    setValidatedKey("");
    setExported(null);
    setError(null);
  }

  function requestArguments() {
    return {
      path: projectPath,
      split_mode: config.splitMode,
      train_ratio: config.train / 100,
      val_ratio: config.val / 100,
      test_ratio: config.test / 100,
      seed: config.seed,
      segment_seconds: config.segmentSeconds,
      negative_ratio: config.negativeRatio / 100,
      parent_manifest_path: parentManifestPath || null,
      replay_strategy: replayStrategy,
      freeze_evaluation_set: config.freezeEvaluationSet,
    };
  }

  async function validate() {
    if (!projectPath) {
      setError("Hãy lưu dự án trước khi kiểm tra dataset.");
      return;
    }
    if (!ratioValid) {
      setError("Tổng train/val/test phải bằng 100% và train phải lớn hơn 0%.");
      return;
    }
    setBusy("validate");
    setError(null);
    setExported(null);
    try {
      const next = await invoke<DatasetReport>("engine_request", {
        request: { id: requestId(), command: "validate_dataset", arguments: requestArguments() },
      });
      setReport(next);
      setValidatedKey(configKey);
    } catch (reason) {
      setError(errorText(reason));
      setReport(null);
      setValidatedKey("");
    } finally {
      setBusy(null);
    }
  }

  async function chooseOutputDirectory() {
    try {
      const selected = await open({ title: "Chọn thư mục chứa các phiên bản dataset", directory: true, multiple: false });
      if (typeof selected === "string") setOutputDirectory(selected);
    } catch (reason) {
      setError(errorText(reason));
    }
  }

  async function chooseParentManifest() {
    try {
      const selected = await open({
        title: "Chọn manifest.json của dataset có test set cố định",
        multiple: false,
        filters: [{ name: "FrameForge manifest", extensions: ["json"] }],
      });
      if (typeof selected !== "string") return;
      setParentManifestPath(selected);
      setConfig((current) => {
        const basis = Math.max(1, current.train + current.val);
        const train = Math.max(1, Math.min(99, Math.round(100 * current.train / basis)));
        return { ...current, train, val: 100 - train, test: 0, freezeEvaluationSet: false };
      });
      setReport(null);
      setValidatedKey("");
      setExported(null);
      setError(null);
    } catch (reason) {
      setError(errorText(reason));
    }
  }

  function clearParentManifest() {
    setParentManifestPath("");
    setConfig((current) => ({ ...current, train: 80, val: 10, test: 10, freezeEvaluationSet: false }));
    setReport(null);
    setValidatedKey("");
    setExported(null);
    setError(null);
  }

  function updateReplayStrategy(value: "all" | "balanced" | "new") {
    setReplayStrategy(value);
    setReport(null);
    setValidatedKey("");
    setExported(null);
  }

  async function exportDataset() {
    if (!projectPath || !validationIsCurrent || !report?.valid || !ratioValid) return;
    let destination = outputDirectory;
    if (!destination) {
      try {
        const selected = await open({ title: "Chọn thư mục chứa các phiên bản dataset", directory: true, multiple: false });
        if (typeof selected !== "string") return;
        destination = selected;
        setOutputDirectory(selected);
      } catch (reason) {
        setError(errorText(reason));
        return;
      }
    }

    setBusy("export");
    setError(null);
    setExported(null);
    try {
      const result = await invoke<ExportResult>("engine_request", {
        request: {
          id: requestId(),
          command: "export_yolo_dataset",
          arguments: { ...requestArguments(), output_dir: destination },
        },
      });
      setExported(result);
    } catch (reason) {
      setError(errorText(reason));
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="dataset-workspace" aria-label="Quản lý và xuất dataset">
      <div className="dataset-content">
        <header className="dataset-heading">
          <div className="dataset-heading-icon"><Database size={18} /></div>
          <div>
            <h1>Dataset</h1>
          <p>Kiểm tra nhãn, giữ nguyên test set qua các phiên bản, rồi tạo snapshot YOLO có lineage.</p>
          </div>
        </header>

        <ProjectDiscovery onUseDataset={onUseDiscoveredDataset} />

        {!projectPath && <div className="dataset-notice" role="status">
          <ShieldAlert size={15} /><span>Lưu dự án trước khi kiểm tra hoặc xuất dataset.</span>
        </div>}
        {projectPath && imageCount === 0 && <div className="dataset-empty">
          <Database size={23} /><strong>Chưa có ảnh trong dự án</strong>
          <span>Trích xuất ảnh trước. Sau đó gán nhãn và review ảnh trống trước khi xuất.</span>
        </div>}

        <div className="dataset-grid">
          <section className="panel dataset-card dataset-config-card">
            <div className="dataset-card-heading"><div><h2>Chia tập</h2><span>Seed giữ kết quả ổn định với cùng nhóm nguồn.</span></div></div>
            <div className="dataset-split-mode" role="group" aria-label="Cách chia dữ liệu">
              <button type="button" className={config.splitMode === "source" ? "selected" : ""} aria-pressed={config.splitMode === "source"} onClick={() => updateConfig({ splitMode: "source" })}>
                <strong>Theo video nguồn</strong><small>Giữ mọi crop của cùng video trong một split.</small>
              </button>
              <button type="button" className={config.splitMode === "segment" ? "selected" : ""} aria-pressed={config.splitMode === "segment"} onClick={() => updateConfig({ splitMode: "segment" })}>
                <strong>Theo đoạn video</strong><small>Chia video thành các nhóm thời gian liền nhau.</small>
              </button>
            </div>
            <div className="dataset-ratios">
              <label>Train (%)<input type="number" min="1" max="100" step="1" value={config.train} onChange={(event) => updateConfig({ train: Number(event.target.value) })} /></label>
              <label>Val (%)<input type="number" min="0" max="100" step="1" value={config.val} onChange={(event) => updateConfig({ val: Number(event.target.value) })} /></label>
              <label>Test (%)<input type="number" min="0" max="100" step="1" value={config.test} disabled={Boolean(parentManifestPath)} onChange={(event) => updateConfig({ test: Number(event.target.value) })} /></label>
            </div>
            <div className="dataset-ratios-caption" aria-live="polite">Tổng: {ratioTotal}%{!ratioValid && " · cần bằng 100%"}</div>
            <div className="dataset-extra-controls">
              <label>Seed<input type="number" min="0" max="4294967295" step="1" value={config.seed} onChange={(event) => updateConfig({ seed: Number(event.target.value) })} /></label>
              {config.splitMode === "segment" && <label>Độ dài đoạn (giây)<input type="number" min="1" max="3600" step="1" value={config.segmentSeconds} onChange={(event) => updateConfig({ segmentSeconds: Number(event.target.value) })} /></label>}
            </div>
            <div className="dataset-negative-control">
              <label htmlFor="dataset-negative-ratio">Tỷ lệ negative tối đa</label>
              <div>
                <select id="dataset-negative-ratio" value={config.negativeRatioCustom ? "custom" : String(config.negativeRatio)} onChange={(event) => {
                  const value = event.target.value;
                  updateConfig(value === "custom"
                    ? { negativeRatioCustom: true }
                    : { negativeRatio: Number(value), negativeRatioCustom: false });
                }}>
                  <option value="0">0%</option>
                  <option value="5">5%</option>
                  <option value="10">10%</option>
                  <option value="25">25%</option>
                  <option value="custom">Tùy chỉnh</option>
                </select>
                {config.negativeRatioCustom && <input type="number" min="0" max="100" step="1" value={config.negativeRatio} aria-label="Tỷ lệ negative tùy chỉnh từ 0 đến 100 phần trăm" aria-invalid={!negativeRatioValid} onChange={(event) => updateConfig({ negativeRatio: Number(event.target.value) })} />}
              </div>
              <small>Tính trên dataset cuối; mặc định giữ tối đa 10%, lấy mẫu tái lập theo seed và bỏ ảnh gần như trùng (dHash ≤1). Chọn 100% để giữ mọi ảnh negative đủ điều kiện.</small>
              {!negativeRatioValid && <small className="dataset-negative-error" role="alert">Nhập tỷ lệ từ 0 đến 100%.</small>}
            </div>
            {!parentManifestPath && <label className="dataset-freeze-control">
              <input type="checkbox" checked={config.freezeEvaluationSet} onChange={(event) => updateConfig({ freezeEvaluationSet: event.target.checked })} />
              <span><strong>Đóng băng val/test</strong><small>Chỉ bật khi mọi ảnh trong hai split đã được review; các phiên bản sau sẽ dùng lại nguyên các ảnh và nhãn này.</small></span>
            </label>}
            {parentManifestPath && <div className="dataset-lineage-controls">
              <div className="dataset-parent-row"><span title={parentManifestPath}>{parentManifestPath}</span><button type="button" onClick={clearParentManifest}>Bỏ parent</button></div>
              <label>Replay ảnh train cũ<select value={replayStrategy} onChange={(event) => updateReplayStrategy(event.target.value as "all" | "balanced" | "new")}>
                <option value="all">Toàn bộ nhóm train cũ</option>
                <option value="balanced">Tập cũ cân bằng theo class</option>
                <option value="new">Chỉ ảnh mới</option>
              </select></label>
              <small>Validation cố định và test set được giữ nguyên từ parent; Test (%) được khóa ở 0 cho ảnh mới.</small>
            </div>}
            <p className="dataset-footnote">Ảnh trống chưa review sẽ bị loại khỏi dataset; ảnh Reviewed Negative được xuất kèm nhãn `.txt` rỗng.</p>
            <button type="button" className="dataset-secondary-button" disabled={!projectPath || imageCount === 0 || busy !== null || !ratioValid} onClick={() => void validate()}>
              {busy === "validate" ? <LoaderCircle className="spin" size={14} /> : <RefreshCw size={14} />}
              {busy === "validate" ? "Đang kiểm tra…" : "Kiểm tra dataset"}
            </button>
          </section>

          <section className="panel dataset-card dataset-export-card">
            <div className="dataset-card-heading"><div><h2>Xuất YOLO</h2><span>Mỗi lần xuất tạo một thư mục phiên bản mới; ảnh gốc không bị resize.</span></div></div>
            {!parentManifestPath && <button type="button" className="dataset-output-button" onClick={() => void chooseParentManifest()}>
              <FolderOpen size={14} /><span>Chọn parent dataset có val/test cố định</span>
            </button>}
            {parentManifestPath && <div className="dataset-lineage-summary">Phiên bản kế thừa parent đã chọn: val {report?.fixed_validation_count ?? "—"} · test {report?.frozen_test_count ?? "—"}</div>}
            <button type="button" className="dataset-output-button" onClick={() => void chooseOutputDirectory()}>
              <FolderOpen size={14} /><span>{outputDirectory || "Chọn thư mục xuất"}</span>
            </button>
            <div className="dataset-output-layout">
              <code>images/train · images/val · images/test<br />labels/train · labels/val · labels/test</code>
              <span>data.yaml<br />manifest.json</span>
            </div>
            <button type="button" className="dataset-export-button" disabled={!projectPath || !validationIsCurrent || !report?.valid || busy !== null || !ratioValid} onClick={() => void exportDataset()}>
              {busy === "export" ? <LoaderCircle className="spin" size={15} /> : <Download size={15} />}
              {busy === "export" ? "Đang tạo dataset…" : "Xuất dataset YOLO"}
            </button>
            {!validationIsCurrent && <small className="dataset-export-hint">Kiểm tra dataset với cấu hình hiện tại trước khi xuất.</small>}
          </section>
        </div>

        {error && <div className="dataset-message error" role="alert"><AlertCircle size={14} /><span>{error}</span></div>}

        {validationIsCurrent && report && <section className="panel dataset-report">
          <div className="dataset-report-heading">
            <div><h2>{report.valid ? "Kết quả kiểm tra" : "Cần sửa trước khi xuất"}</h2><span>{report.project_name} · {report.included_image_count} ảnh cuối · {report.new_image_count} ảnh mới · {report.replayed_image_count} ảnh replay</span></div>
            <span className={`dataset-validity${report.valid ? " valid" : " invalid"}`}>{report.valid ? <><CheckCircle2 size={14} /> Hợp lệ</> : <><AlertCircle size={14} /> Có lỗi</>}</span>
          </div>
          <div className="dataset-stat-grid">
            <div><strong>{report.included_image_count}</strong><span>Ảnh export</span></div>
            <div><strong>{report.positive_image_count}</strong><span>Ảnh có box</span></div>
            <div><strong>{report.reviewed_negative_count}</strong><span>Negative đã review</span></div>
            <div><strong>{report.annotation_count}</strong><span>Bounding box</span></div>
          </div>
          <p className="dataset-negative-summary" aria-live="polite">Negative: giữ {report.reviewed_negative_count}/{report.negative_samples_available_count} ảnh đã review · tỷ lệ thực tế {(report.negative_ratio_actual * 100).toFixed(1)}% / giới hạn {(report.negative_ratio_requested * 100).toFixed(1)}%{report.negative_samples_excluded_count > 0 ? ` · đã loại ${report.negative_samples_excluded_count} ảnh` : ""}</p>
          <div className="dataset-report-columns">
            <div><h3>Split thực tế</h3>
              {(["train", "val", "test"] as const).map((split) => <div className="dataset-split-row" key={split}>
                <span>{split}</span><strong>{report.split_counts[split]}</strong><small>{report.split_group_counts[split]} nhóm</small>
              </div>)}
            </div>
            <div><h3>Class</h3>
              {report.class_distribution.map((item) => <div className="dataset-class-row" key={item.class_id}><span>{item.class_id}. {item.name}</span><strong>{item.box_count} box</strong></div>)}
            </div>
          </div>
          {report.candidate_negative_excluded_count > 0 && <p className="dataset-candidate-note">Đã loại {report.candidate_negative_excluded_count} ảnh trống chưa review.</p>}
          {report.unreviewed_annotation_excluded_count > 0 && <p className="dataset-candidate-note">Đã loại {report.unreviewed_annotation_excluded_count} ảnh có box thủ công chưa review. Đánh dấu ảnh đã review để đưa nhãn vào dataset.</p>}
          {report.errors.length > 0 && <div className="dataset-findings errors"><h3>Lỗi cần sửa</h3><ul>{report.errors.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul></div>}
          {report.warnings.length > 0 && <div className="dataset-findings warnings"><h3>Cảnh báo</h3><ul>{report.warnings.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul></div>}
        </section>}

        {exported && <div className="dataset-message success" role="status">
          <CheckCircle2 size={15} /><span>Đã tạo v{exported.version} · {exported.image_count} ảnh · mới {exported.new_image_count} · replay {exported.replayed_image_count} · train {exported.split_counts.train} · val {exported.split_counts.val} · test {exported.split_counts.test}<small>{exported.path}</small><small>{exported.data_yaml}</small></span>
        </div>}
      </div>
    </section>
  );
}
