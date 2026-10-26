import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from "react";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save, confirm, message } from "@tauri-apps/plugin-dialog";
import {
  AlertCircle,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  CircleHelp,
  Database,
  Download,
  FileImage,
  FileJson,
  FileVideo2,
  FolderOpen,
  FolderPlus,
  ImagePlus,
  LoaderCircle,
  Pause,
  Play,
  Plus,
  Save as SaveIcon,
  ScanLine,
  Scissors,
  Settings2,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import "./App.css";
import DatasetManager from "./DatasetManager";
import ProjectGallery, { type GalleryImage } from "./ProjectGallery";
import ManualLabeling, { type Annotation, type InferenceAttempt, type NegativeCandidate } from "./ManualLabeling";
import { availableModels, modelMatchesClasses, pickAndImportCheckpoint, type RegisteredModel } from "./modelImport";

type MediaSource = {
  path: string;
  name: string;
  kind: "video" | "image";
  width: number;
  height: number;
  fps: number | null;
  frame_count: number | null;
  duration_seconds: number | null;
  size_bytes: number;
};

type ProjectImage = {
  path: string;
  source_path?: string;
  source_id?: string;
  frame_index?: number;
  timestamp_seconds?: number;
  crop_rect?: CropRect;
  sharpness_score?: number;
  extraction_settings?: ExtractionOptions;
  width: number;
  height: number;
  status: string;
  annotations?: Annotation[];
  negative_candidates?: NegativeCandidate[];
  inference_attempts?: InferenceAttempt[];
};

type ExtractionOptions = {
  interval: number;
  interval_unit: "frames" | "seconds";
  start_seconds: number;
  end_seconds: number | null;
  max_frames: number;
  filter_duplicates: boolean;
  duplicate_threshold: number;
  filter_blurry: boolean;
  strict_sharpness?: boolean;
  pick_sharpest?: boolean;
  verify_duplicates?: boolean;
  blur_threshold: number;
  crop_mode: "center" | "letterbox" | "grid" | "object_centered";
  output_width: number;
  output_height: number;
  overlap_x: number;
  overlap_y: number;
  padding_color: string;
  image_format: "png";
  existing_file_behavior: "skip" | "overwrite" | "rename";
  /** Registered model run that filters crops and pre-labels them while extracting; "" = off. */
  detect_model_id: string;
  detect_enabled?: boolean;
  detect_use_gpu?: boolean;
  detect_confidence: number;
  detect_keep_empty: boolean;
};

type ProjectState = {
  id?: string;
  name: string;
  settings_version: number;
  settings_migrated?: boolean;
  sources: MediaSource[];
  settings: ExtractionOptions;
  images: ProjectImage[];
  class_names: string[];
};

type WorkspaceTab = "extract" | "label" | "dataset";

type ImageImportResult = {
  found_images: number;
  imported_images: number;
  json_images: number;
  labeled_images: number;
  unlabeled_images: number;
  reviewed_images: number;
  box_count: number;
  skipped_existing: number;
  unsupported_shapes: number;
  unknown_labels: number;
  clipped_boxes: number;
  degenerate_boxes: number;
  errors: { path: string; message: string }[];
  error_count: number;
};

type ImageImportPreview = { found_images: number; new_images: number; json_images: number; skipped_existing: number };
type SessionSnapshot = {
  available: boolean;
  project?: ProjectState;
  session?: Record<string, unknown>;
  original_project_exists?: boolean;
  original_project_unchanged?: boolean;
};

type FramePreview = {
  source_path?: string;
  data_url: string;
  width: number;
  height: number;
  source_width?: number;
  source_height?: number;
  frame_index: number;
  sharpness_score?: number;
  predictions?: { class_id: number; confidence: number; bbox_xyxy: number[] }[];
  crop_rects?: CropRect[];
  crop_sharpness?: number[];
};

type CropRect = { x: number; y: number; width: number; height: number };

type EngineProgress = {
  source_index: number;
  source_total: number;
  source_name: string;
  frame_index: number;
  elapsed_seconds: number;
  candidate_frames: number;
  accepted_frames: number;
  candidate_crops: number;
  accepted_crops: number;
  duplicate_crops: number;
  blurry_crops: number;
  exported_images: number;
  skipped_files: number;
  failed_sources: number;
  no_object_crops?: number;
  detected_images?: number;
  source_frames_total?: number;
  scanned_frames?: number;
  sampling_skipped_frames?: number;
  blurry_object_crops?: number;
};

type ExtractionFile = ProjectImage & {
  name: string;
  thumbnail_data_url?: string;
};

type ExtractionDetection = {
  image_fingerprint: string;
  predictions: { class_id: number; confidence: number; bbox_xyxy: number[] }[];
  skipped_large_objects: number;
};

type ExtractedImage = Omit<ProjectImage, "status" | "annotations" | "negative_candidates" | "inference_attempts" | "extraction_settings"> & {
  overwritten: boolean;
  ai_detection?: ExtractionDetection;
};

type PendingExtractionSuggestions = {
  checkpoint: string;
  checkpoint_sha256: string;
  items: ({ image_path: string } & ExtractionDetection)[];
};

type ExtractionResult = EngineProgress & {
  files: ExtractionFile[];
  file_list_truncated: boolean;
  images: ExtractedImage[];
  extraction_settings: ExtractionOptions;
  detection: { source_model_id: string; checkpoint_sha256: string; checkpoint: string } | null;
  errors: { path: string; message: string }[];
};

type PanelLayout = { sourceWidth: number; settingsWidth: number; galleryHeight: number };
type ResizeKind = "source" | "settings" | "gallery";
type ResizeDrag = { kind: ResizeKind; startX: number; startY: number; initialValue: number };

const FHD_WIDTH = 1920;
const FHD_HEIGHT = 1080;
const PROJECT_SETTINGS_VERSION = 1;

const DEFAULT_OPTIONS: ExtractionOptions = {
  interval: 10,
  interval_unit: "frames",
  start_seconds: 0,
  end_seconds: null,
  max_frames: 0,
  filter_duplicates: true,
  duplicate_threshold: 6,
  filter_blurry: true,
  strict_sharpness: true,
  pick_sharpest: true,
  verify_duplicates: true,
  blur_threshold: 60,
  crop_mode: "center",
  output_width: 320,
  output_height: 320,
  overlap_x: 0.2,
  overlap_y: 0.2,
  padding_color: "#000000",
  image_format: "png",
  existing_file_behavior: "skip",
  detect_model_id: "",
  detect_confidence: 35,
  detect_keep_empty: false,
};

const MEDIA_FILTERS = [
  { name: "Video và ảnh", extensions: ["mp4", "mov", "mkv", "avi", "webm", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"] },
];

const DEFAULT_PANEL_LAYOUT: PanelLayout = { sourceWidth: 226, settingsWidth: 320, galleryHeight: 174 };

function readPanelLayout(): PanelLayout {
  try {
    const stored = localStorage.getItem("frameforge.extract-layout.v1");
    if (!stored) return DEFAULT_PANEL_LAYOUT;
    const value = JSON.parse(stored) as Partial<PanelLayout>;
    return {
      sourceWidth: Number.isFinite(value.sourceWidth) ? Math.max(180, Math.min(360, value.sourceWidth!)) : DEFAULT_PANEL_LAYOUT.sourceWidth,
      settingsWidth: Number.isFinite(value.settingsWidth) ? Math.max(260, Math.min(440, value.settingsWidth!)) : DEFAULT_PANEL_LAYOUT.settingsWidth,
      galleryHeight: Number.isFinite(value.galleryHeight) ? Math.max(110, Math.min(300, value.galleryHeight!)) : DEFAULT_PANEL_LAYOUT.galleryHeight,
    };
  } catch {
    return DEFAULT_PANEL_LAYOUT;
  }
}

function makeRequestId() {
  return globalThis.crypto?.randomUUID?.() ?? `ff-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatTime(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "--:--";
  const safe = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const rest = safe % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${minutes}:${String(rest).padStart(2, "0")}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const LAST_PROJECT_KEY = "frameforge.last-project.v1";

function rememberProject(path: string) {
  try { localStorage.setItem(LAST_PROJECT_KEY, path); } catch { /* per-device convenience only */ }
}

function lastProject() {
  try { return localStorage.getItem(LAST_PROJECT_KEY); } catch { return null; }
}

function projectNameFromPath(path: string) {
  return path.split(/[\\/]/).pop()?.replace(/\.ffproj$/i, "") || "Dự án FrameForge";
}

function App() {
  const [project, setProject] = useState<ProjectState>({
    name: "Dự án chưa lưu",
    settings_version: PROJECT_SETTINGS_VERSION,
    settings_migrated: false,
    sources: [],
    settings: DEFAULT_OPTIONS,
    images: [],
    class_names: ["head", "body"],
  });
  const [projectPath, setProjectPath] = useState<string | null>(null);
  const projectPathRef = useRef(projectPath);
  useEffect(() => { projectPathRef.current = projectPath; }, [projectPath]);
  const [dirty, setDirty] = useState(false);
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [preview, setPreview] = useState<FramePreview | null>(null);
  const [previewStageSize, setPreviewStageSize] = useState({ width: 0, height: 0 });
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [detectionPreview, setDetectionPreview] = useState<FramePreview | null>(null);
  const [previewDetecting, setPreviewDetecting] = useState(false);
  const [detectionPreviewError, setDetectionPreviewError] = useState<string | null>(null);
  const [showDetections, setShowDetections] = useState(true);
  const [previewRefresh, setPreviewRefresh] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [outputDirectory, setOutputDirectory] = useState("");
  const [galleryThumbnails, setGalleryThumbnails] = useState<Record<string, string>>({});
  const [progress, setProgress] = useState<EngineProgress | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [engineStatus, setEngineStatus] = useState<"checking" | "ready" | "error">("checking");
  const [sessionReady, setSessionReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dropActive, setDropActive] = useState(false);
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("extract");
  const [importingLabels, setImportingLabels] = useState(false);
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [modelBusy, setModelBusy] = useState(false);
  const [inferenceAvailable, setInferenceAvailable] = useState<boolean | null>(null);
  const [inferenceDevice, setInferenceDevice] = useState<{ preferred_device?: string; device_name?: string } | null>(null);
  const [pendingSuggestions, setPendingSuggestions] = useState<PendingExtractionSuggestions | null>(null);
  const [detectionSummary, setDetectionSummary] = useState<string | null>(null);
  const [panelLayout, setPanelLayout] = useState(readPanelLayout);
  const projectWriteQueue = useRef<Promise<void>>(Promise.resolve());
  const annotationSaveFailed = useRef(false);
  const sessionSaveErrorShown = useRef(false);
  const previewSequence = useRef(0);
  const previewRequestStartedAt = useRef(0);
  const previewDetectionContext = useRef("");
  const activeExtractionId = useRef<string | null>(null);
  const progressSubscription = useRef<Promise<void>>(Promise.resolve());
  const resizeDrag = useRef<ResizeDrag | null>(null);
  const sourcePanelRef = useRef<HTMLElement | null>(null);
  const settingsPanelRef = useRef<HTMLElement | null>(null);
  const galleryPanelRef = useRef<HTMLElement | null>(null);
  const previewStageRef = useRef<HTMLDivElement | null>(null);

  const selectedSource = useMemo(
    () => project.sources.find((source) => source.path === selectedPath) ?? null,
    [project.sources, selectedPath],
  );
  const selectedSources = useMemo(
    () => project.sources.filter((source) => selectedPaths.includes(source.path)),
    [project.sources, selectedPaths],
  );
  const sourceFrameCount = selectedSource?.frame_count ?? (selectedSource?.kind === "image" ? 1 : null);
  const lastFrame = sourceFrameCount ? Math.max(0, sourceFrameCount - 1) : 0;
  const sourceResizeMax = Math.min(360, window.innerWidth * (window.innerWidth <= 1240 ? 0.24 : 0.25));
  const settingsResizeMax = Math.min(440, window.innerWidth * 0.31);
  const currentTime = selectedSource?.kind === "image"
    ? 0
    : selectedSource?.fps
      ? frameIndex / selectedSource.fps
      : null;
  const duration = selectedSource?.duration_seconds ?? null;
  const minimumOutputSize = options.crop_mode === "center" || options.crop_mode === "object_centered" ? 1 : 16;
  const cropSizeValid = Number.isInteger(options.output_width) && Number.isInteger(options.output_height)
    && options.output_width >= minimumOutputSize && options.output_width <= FHD_WIDTH
    && options.output_height >= minimumOutputSize && options.output_height <= FHD_HEIGHT;
  const cropFitsPreview = Boolean(cropSizeValid && preview && options.output_width <= preview.width && options.output_height <= preview.height);
  const cropRect = preview
    ? {
        x: Math.floor((preview.width - options.output_width) / 2),
        y: Math.floor((preview.height - options.output_height) / 2),
        width: options.output_width,
        height: options.output_height,
      }
    : null;
  const previewFrameBox = useMemo(() => {
    if (!preview || !previewStageSize.width || !previewStageSize.height) return null;
    const scale = Math.min(1, previewStageSize.width / preview.width, previewStageSize.height / preview.height);
    return { width: preview.width * scale, height: preview.height * scale };
  }, [preview, previewStageSize]);

  async function callEngine<T>(command: string, args: Record<string, unknown> = {}) {
    return invoke<T>("engine_request", {
      request: { id: makeRequestId(), command, arguments: args },
    });
  }

  function serializeProjectWrite<T>(operation: () => Promise<T>): Promise<T> {
    const queued = projectWriteQueue.current.then(operation, operation);
    projectWriteQueue.current = queued.then(() => undefined, () => undefined);
    return queued;
  }

  const loadLabelPreview = useCallback((path: string) => (
    callEngine<{ data_url: string; width: number; height: number; image_fingerprint?: string }>(
      "preview_frame",
      { path, frame_index: 0, lossless: true },
    )
  ), []);

  const saveLabelImage = useCallback(async (
    imagePath: string,
    annotations: Annotation[],
    status: "ready" | "reviewed",
    fingerprint: string,
  ): Promise<Annotation[]> => {
    const path = projectPath;
    if (!path) throw new Error("Hãy lưu dự án trước khi chỉnh sửa nhãn.");
    setProject((current) => ({
      ...current,
      images: current.images.map((image) => image.path === imagePath
        ? { ...image, annotations: annotations.map((annotation) => ({ ...annotation })), status }
        : image),
    }));
    try {
      const saved = await serializeProjectWrite(() => callEngine<{ annotations: Annotation[]; json_errors?: { path: string; message: string }[] }>("save_annotations", {
        path,
        image_path: imagePath,
        annotations,
        status,
        image_fingerprint: fingerprint,
      }));
      setProject((current) => ({
        ...current,
        images: current.images.map((image) => image.path === imagePath
          ? { ...image, annotations: saved.annotations, status }
          : image),
      }));
      annotationSaveFailed.current = false;
      if (saved.json_errors?.length) {
        setError(`Nhãn đã lưu trong dự án nhưng chưa ghi được file JSON: ${saved.json_errors[0].message} (${saved.json_errors[0].path})`);
      }
      return saved.annotations;
    } catch (reason) {
      annotationSaveFailed.current = true;
      setDirty(true);
      throw reason;
    }
  }, [projectPath]);

  const saveLabelSuggestions = useCallback(async (
    imagePath: string,
    predictions: { class_id: number; confidence: number; bbox_xyxy: number[] }[],
    checkpoint: string,
    fingerprint: string,
    skippedLargeObjects: number,
    replace: boolean,
  ): Promise<{ annotations: Annotation[]; negative_candidate?: NegativeCandidate | null }> => {
    const path = projectPath;
    if (!path) throw new Error("Hãy lưu dự án trước khi tạo gợi ý AI.");
    const saved = await serializeProjectWrite(() => callEngine<{
      annotations: Annotation[];
      negative_candidate?: NegativeCandidate | null;
      status?: string;
      source_model_id?: string;
      checkpoint_sha256?: string;
    }>("save_ai_suggestions", {
      path,
      image_path: imagePath,
      predictions,
      checkpoint,
      image_fingerprint: fingerprint,
      skipped_large_objects: skippedLargeObjects,
      replace,
    }));
    const attempt: InferenceAttempt | null = !saved.annotations.length && !saved.negative_candidate && saved.source_model_id && saved.checkpoint_sha256
      ? { source_model_id: saved.source_model_id, checkpoint_sha256: saved.checkpoint_sha256, image_fingerprint: fingerprint, skipped_large_objects: skippedLargeObjects }
      : null;
    setProject((current) => ({
      ...current,
      images: current.images.map((image) => image.path === imagePath
        ? {
            ...image,
            inference_attempts: attempt
              ? [...(replace ? [] : (image.inference_attempts ?? []).filter((item) => item.checkpoint_sha256 !== attempt.checkpoint_sha256)), attempt]
              : replace ? [] : image.inference_attempts,
            annotations: [...(replace ? [] : image.annotations ?? []), ...saved.annotations],
            negative_candidates: saved.negative_candidate
              ? [...(replace ? [] : image.negative_candidates ?? []), saved.negative_candidate]
              : replace ? [] : image.negative_candidates,
            status: saved.status ?? image.status,
          }
        : image),
    }));
    return saved;
  }, [projectPath]);

  /** Asks, then removes the image from the project and moves the image file and its JSON to the trash. */
  const deleteLabelImage = useCallback(async (imagePath: string, boxCount: number): Promise<boolean> => {
    const path = projectPath;
    if (!path) throw new Error("Hãy lưu dự án trước khi xóa ảnh.");
    const name = imagePath.split(/[\\/]/).pop() ?? imagePath;
    const approved = await confirm(
      `Xóa ảnh "${name}"${boxCount ? ` và ${boxCount} box` : ""}?\n\nẢnh và file .json của nó được chuyển vào Thùng rác (khôi phục được từ đó) và gỡ khỏi dự án.`,
      { title: "Xóa ảnh", kind: "warning", okLabel: "Xóa ảnh", cancelLabel: "Hủy" },
    );
    if (!approved) return false;
    await serializeProjectWrite(() => callEngine("delete_images", { path, image_paths: [imagePath] }));
    setProject((current) => ({ ...current, images: current.images.filter((image) => image.path !== imagePath) }));
    return true;
  }, [projectPath]);

  const reviewLabelNegativeCandidate = useCallback(async (
    candidateId: string,
    action: "confirmed_negative" | "rejected",
    fingerprint?: string,
  ): Promise<void> => {
    const path = projectPath;
    if (!path) throw new Error("Hãy lưu dự án trước khi review candidate.");
    const saved = await serializeProjectWrite(() => callEngine<{
      image_path: string;
      status: string;
      negative_candidate: NegativeCandidate;
    }>("review_ai_negative_candidate", {
      path,
      candidate_id: candidateId,
      action,
      ...(fingerprint ? { image_fingerprint: fingerprint } : {}),
    }));
    setProject((current) => ({
      ...current,
      images: current.images.map((image) => image.path === saved.image_path
        ? {
            ...image,
            status: saved.status,
            negative_candidates: (image.negative_candidates ?? []).map((candidate) => (
              candidate.id === candidateId ? saved.negative_candidate : candidate
            )),
          }
        : image),
    }));
  }, [projectPath]);

  const applyModelClasses = useCallback(async (classNames: string[]) => {
    const labeled = project.images.some((image) => (
      (image.annotations?.length ?? 0) > 0 || (image.negative_candidates?.length ?? 0) > 0
    ));
    if (labeled) throw new Error("Dự án đã có nhãn hoặc candidate AI; không thể đổi class mà không làm sai class ID.");
    const path = projectPath;
    let applied = classNames;
    if (path) {
      const saved = await serializeProjectWrite(() => callEngine<{ class_names: string[] }>("set_project_classes", {
        path,
        class_names: classNames,
      }));
      applied = saved.class_names;
    } else {
      setDirty(true);
    }
    setProject((current) => ({
      ...current,
      class_names: applied,
      images: current.images.map((image) => ({
        ...image,
        status: image.status === "reviewed" ? "ready" : image.status,
        inference_attempts: [],
      })),
    }));
  }, [project.images, projectPath]);

  /** Accepts the pending AI boxes of these images and marks every image that has boxes reviewed. */
  const reviewLabelImages = useCallback(async (imagePaths: string[]): Promise<string | null> => {
    const path = projectPath;
    if (!path) throw new Error("Hãy lưu dự án trước khi duyệt ảnh.");
    const approved = await confirm(
      `Duyệt ${imagePaths.length} ảnh đang hiện trong danh sách?\n\nMọi box AI đang chờ duyệt của các ảnh này thành nhãn thật (ghi vào JSON) và ảnh có box được đánh dấu đã review. Ảnh không có box giữ nguyên, không thành ảnh negative.`,
      { title: "Duyệt toàn bộ ảnh", kind: "warning", okLabel: "Duyệt", cancelLabel: "Hủy" },
    );
    if (!approved) return null;
    try {
      const result = await serializeProjectWrite(() => callEngine<{
        accepted_suggestion_count: number;
        reviewed_image_count: number;
        skipped_empty_count: number;
        skipped_pending_negative_count: number;
      }>("review_images", { path, image_paths: imagePaths }));
      const skipped = [
        result.skipped_empty_count ? `${result.skipped_empty_count} ảnh không có box` : "",
        result.skipped_pending_negative_count ? `${result.skipped_pending_negative_count} ảnh có candidate negative chờ xác nhận` : "",
      ].filter(Boolean).join(", ");
      return `Đã duyệt ${result.accepted_suggestion_count} box AI · ${result.reviewed_image_count} ảnh thành đã review${skipped ? ` · bỏ qua ${skipped}` : ""}.`;
    } finally {
      await reloadLabels(path);
    }
  }, [projectPath]);

  const runProjectInference = useCallback(async (checkpoint: string, requestId: string, confidence: number, replace: boolean) => {
    const path = projectPath;
    if (!path) throw new Error("Hãy lưu dự án trước khi chạy inference hàng loạt.");
    try {
      return await serializeProjectWrite(() => invoke<{
        status: string;
        processed_images: number;
        images_with_detections: number;
        candidate_negative_count: number;
        annotation_count: number;
        skipped_changed: number;
        skipped_large_objects: number;
        skipped_attempted: number;
      }>("engine_request", {
        request: {
          id: requestId,
          command: "auto_label_project",
          arguments: { path, checkpoint, confidence_threshold: confidence, replace },
        },
      }));
    } finally {
      await reloadLabels(path);
    }
  }, [projectPath]);

  /** Reloads labels and review state written by a project-wide engine command. */
  async function reloadLabels(path: string) {
    const loaded = await serializeProjectWrite(() => callEngine<ProjectState>("load_project", { path }));
    if (projectPathRef.current !== path) return;
    const savedImages = new Map(loaded.images.map((image) => [image.path, image]));
    setProject((current) => ({
      ...current,
      images: current.images.map((image) => {
        const saved = savedImages.get(image.path);
        return saved ? {
          ...image,
          annotations: saved.annotations,
          negative_candidates: saved.negative_candidates,
          inference_attempts: saved.inference_attempts,
          status: saved.status,
        } : image;
      }),
    }));
  }

  useEffect(() => {
    let alive = true;
    let interacted = false;
    const noteInteraction = () => { interacted = true; };
    window.addEventListener("pointerdown", noteInteraction, true);
    window.addEventListener("keydown", noteInteraction, true);
    if (!isTauri()) {
      setEngineStatus("error");
      setSessionReady(true);
      return () => {
        alive = false;
        window.removeEventListener("pointerdown", noteInteraction, true);
        window.removeEventListener("keydown", noteInteraction, true);
      };
    }
    void (async () => {
      let engineReady = false;
      try {
        await callEngine<{ status: string }>("health", {});
        if (!alive) return;
        engineReady = true;
        setEngineStatus("ready");
        const snapshot = await callEngine<SessionSnapshot>("load_session", {});
        if (!alive || interacted) return;
        const snapshotHasWork = Boolean(snapshot.available && snapshot.project && snapshot.session && (
          typeof snapshot.session.project_path === "string"
          || snapshot.project.images.length
          || snapshot.project.sources.length
        ));
        if (!snapshotHasWork) {
          const previous = lastProject();
          if (previous) {
            try {
              await openProjectPath(previous);
            } catch {
              // The last project moved or was deleted: start empty.
            }
          }
          return;
        }
        if (!snapshot.project || !snapshot.session) return;
        if (snapshot.session.light === true && typeof snapshot.session.project_path === "string") {
          if (snapshot.original_project_exists) {
            await openProjectPath(snapshot.session.project_path);
            if (typeof snapshot.session.output_directory === "string") setOutputDirectory(snapshot.session.output_directory);
          }
          return;
        }

        const settingsMigrated = snapshot.project.settings_migrated === true;
        const restoredProject = {
          ...snapshot.project,
          settings_version: PROJECT_SETTINGS_VERSION,
          settings_migrated: false,
          settings: { ...DEFAULT_OPTIONS, ...(snapshot.project.settings as Partial<ExtractionOptions>), image_format: "png" as const },
        };
        const sourcePaths = new Set(restoredProject.sources.map((source) => source.path));
        const requestedSelectedPaths = snapshot.session.selected_paths;
        const restoredSelectedPaths = Array.isArray(requestedSelectedPaths)
          ? requestedSelectedPaths.filter((path): path is string => typeof path === "string" && sourcePaths.has(path))
          : restoredProject.sources.map((source) => source.path);
        const requestedSelectedPath = snapshot.session.selected_path;
        const restoredSelectedPath = typeof requestedSelectedPath === "string" && sourcePaths.has(requestedSelectedPath)
          ? requestedSelectedPath
          : restoredProject.sources[0]?.path ?? null;
        const selectedSource = restoredProject.sources.find((source) => source.path === restoredSelectedPath);
        const requestedFrame = snapshot.session.frame_index;
        const restoredFrame = typeof requestedFrame === "number" && Number.isFinite(requestedFrame)
          ? Math.max(0, Math.min(selectedSource?.frame_count ? selectedSource.frame_count - 1 : Number.MAX_SAFE_INTEGER, Math.floor(requestedFrame)))
          : 0;
        const requestedProjectPath = snapshot.session.project_path;
        const originalIsSafeToResume = snapshot.original_project_exists === true
          && snapshot.original_project_unchanged === true
          && typeof requestedProjectPath === "string";

        setProject(restoredProject);
        setOptions(restoredProject.settings);
        setProjectPath(originalIsSafeToResume ? requestedProjectPath : null);
        setDirty(snapshot.session.dirty === true || !originalIsSafeToResume || settingsMigrated);
        setSelectedPaths(restoredSelectedPaths);
        setSelectedPath(restoredSelectedPath);
        setFrameIndex(restoredFrame);
        setOutputDirectory(typeof snapshot.session.output_directory === "string" ? snapshot.session.output_directory : "");
        setActiveTab("extract");
        if (originalIsSafeToResume) rememberProject(requestedProjectPath);
        setGalleryThumbnails({});
        annotationSaveFailed.current = false;
      } catch (reason) {
        if (alive && engineReady && !interacted) {
          setError(`Không khôi phục được phiên trước. Mở dự án thủ công để tiếp tục. ${String(reason)}`);
        }
        if (alive && !engineReady) setEngineStatus("error");
      } finally {
        if (alive) setSessionReady(true);
      }
    })();
    return () => {
      alive = false;
      window.removeEventListener("pointerdown", noteInteraction, true);
      window.removeEventListener("keydown", noteInteraction, true);
    };
  }, []);

  useEffect(() => {
    if (!sessionReady || !isTauri()) return;
    const hasSessionState = Boolean(
      projectPath
      || project.sources.length
      || project.images.length
      || outputDirectory
      || JSON.stringify(options) !== JSON.stringify(DEFAULT_OPTIONS),
    );
    if (!hasSessionState) return;
    const timer = window.setTimeout(() => {
      // A saved project without unsaved changes is its own state: send no images.
      const light = Boolean(projectPath && !dirty);
      void serializeProjectWrite(() => callEngine("save_session", {
        project: light ? { ...project, images: [], sources: [], settings: options } : { ...project, settings: options },
        session: {
          light,
          project_path: projectPath,
          selected_paths: selectedPaths,
          selected_path: selectedPath,
          frame_index: frameIndex,
          output_directory: outputDirectory,
          active_tab: activeTab,
          dirty,
        },
      })).then(() => {
        sessionSaveErrorShown.current = false;
      }).catch((reason) => {
        if (!sessionSaveErrorShown.current) {
          setError(`Không lưu được bản khôi phục phiên: ${String(reason)}`);
          sessionSaveErrorShown.current = true;
        }
      });
    }, 900);
    return () => window.clearTimeout(timer);
  }, [sessionReady, project, projectPath, selectedPaths, selectedPath, frameIndex, outputDirectory, activeTab, dirty, options]);

  useEffect(() => {
    try {
      localStorage.setItem("frameforge.extract-layout.v1", JSON.stringify(panelLayout));
    } catch {
      // Keep the workspace usable when storage is unavailable.
    }
  }, [panelLayout]);

  useEffect(() => {
    const element = previewStageRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setPreviewStageSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const fitPanels = () => {
      const width = window.innerWidth;
      const sourceMax = Math.min(360, width * (width <= 1240 ? 0.24 : 0.25));
      const settingsMax = Math.min(440, width * 0.31);
      setPanelLayout((current) => {
        const sourceWidth = Math.max(180, Math.min(sourceMax, current.sourceWidth));
        const settingsWidth = Math.max(260, Math.min(settingsMax, current.settingsWidth));
        return sourceWidth === current.sourceWidth && settingsWidth === current.settingsWidth
          ? current
          : { ...current, sourceWidth, settingsWidth };
      });
    };
    window.addEventListener("resize", fitPanels);
    return () => window.removeEventListener("resize", fitPanels);
  }, []);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let active = true;
    if (!isTauri()) return () => { active = false; };
    progressSubscription.current = listen<{ id: string; progress: EngineProgress }>("engine-progress", (event) => {
      if (event.payload.id === activeExtractionId.current) setProgress(event.payload.progress);
    }).then((dispose) => {
      if (active) unlisten = dispose;
      else dispose();
    }).catch(() => undefined);
    return () => { active = false; unlisten?.(); };
  }, []);

  useEffect(() => {
    let active = true;
    let unlisten: (() => void) | undefined;
    if (!isTauri()) return () => { active = false; };
    void getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === "enter") setDropActive(true);
      if (event.payload.type === "leave") setDropActive(false);
      if (event.payload.type === "drop") {
        setDropActive(false);
        void importMedia(event.payload.paths);
      }
    }).then((dispose) => {
      if (active) unlisten = dispose;
      else dispose();
    }).catch(() => undefined);
    return () => { active = false; unlisten?.(); };
  }, []);

  const previewModel = availableModels(models).find((item) => item.run_id === options.detect_model_id);
  const previewDetectionKey = JSON.stringify(showDetections && !extracting && !modelBusy && cropSizeValid
    && previewModel?.checkpoint_path && modelMatchesClasses(previewModel, project.class_names) ? {
      crop_mode: options.crop_mode === "object_centered" ? "object_centered" : "letterbox",
      output_width: options.output_width, output_height: options.output_height,
      detection: { checkpoint: previewModel.checkpoint_path, confidence_threshold: options.detect_confidence / 100,
        class_names: project.class_names, device: options.detect_use_gpu === false ? "cpu" : "gpu", trust_source: true },
    } : null);

  useEffect(() => {
    const sequence = ++previewSequence.current;
    const detectionContext = `${selectedSource?.path ?? ""}|${previewDetectionKey}`;
    if (previewDetectionContext.current !== detectionContext) setDetectionPreview(null);
    previewDetectionContext.current = detectionContext;
    setDetectionPreviewError(null);
    setPreviewDetecting(false);
    if (!selectedSource) {
      setPreview(null);
      setPreviewError(null);
      return;
    }
    const path = selectedSource.path;
    const index = selectedSource.kind === "image" ? 0 : frameIndex;
    const detectionOptions = JSON.parse(previewDetectionKey);
    setPreviewDetecting(Boolean(detectionOptions));
    setPreviewError(null);
    const timer = window.setTimeout(() => {
      previewRequestStartedAt.current = performance.now();
      setPreviewDetecting(Boolean(detectionOptions));
      void callEngine<FramePreview>("preview_frame", { path, frame_index: index, ...(detectionOptions ? { options: detectionOptions } : {}) })
        .then((result) => {
          if (sequence === previewSequence.current) {
            setPreview({ ...result, source_path: path });
            if (detectionOptions) setDetectionPreview(result);
          }
        })
        .catch(async (reason) => {
          if (sequence === previewSequence.current) {
            if (detectionOptions) {
              setDetectionPreview(null);
              setDetectionPreviewError(String(reason));
              setPlaying(false);
              try {
                const result = await callEngine<FramePreview>("preview_frame", { path, frame_index: index });
                if (sequence === previewSequence.current) setPreview({ ...result, source_path: path });
              } catch (fallbackReason) {
                if (sequence === previewSequence.current) {
                  setPreview(null);
                  setPreviewError(String(fallbackReason));
                }
              }
            } else {
              setPreview(null);
              setPreviewError(String(reason));
            }
          }
        }).finally(() => {
          if (sequence === previewSequence.current) setPreviewDetecting(false);
        });
    }, playing ? 0 : 90);
    return () => {
      window.clearTimeout(timer);
      previewSequence.current += 1;
    };
  }, [selectedSource?.path, selectedSource?.kind, frameIndex, playing, previewDetectionKey, previewRefresh]);

  useEffect(() => {
    if (!playing || !selectedSource || selectedSource.kind !== "video") return;
    if (previewDetecting) return;
    if (previewError) {
      setPlaying(false);
      return;
    }
    // Wait for decoding: advancing on a separate clock can invalidate every preview response.
    if (!preview || preview.frame_index !== frameIndex || preview.source_path !== selectedSource.path) return;
    if (sourceFrameCount && frameIndex >= sourceFrameCount - 1) {
      setPlaying(false);
      return;
    }
    const frameDuration = 1000 / Math.min(60, selectedSource.fps || 24);
    const interval = Math.max(0, frameDuration - (performance.now() - previewRequestStartedAt.current));
    const timer = window.setTimeout(() => setFrameIndex((index) => index + 1), interval);
    return () => window.clearTimeout(timer);
  }, [playing, selectedSource?.path, selectedSource?.fps, selectedSource?.kind, sourceFrameCount, frameIndex, preview, previewError, previewDetecting]);

  function updateOptions(patch: Partial<ExtractionOptions>) {
    setOptions((current) => ({ ...current, ...patch }));
    setProject((current) => ({ ...current, settings: { ...current.settings, ...patch } }));
    setDirty(true);
  }

  function updateFrame(next: number) {
    setFrameIndex(Math.max(0, sourceFrameCount ? Math.min(sourceFrameCount - 1, next) : next));
  }

  function beginResize(kind: ResizeKind, event: PointerEvent<HTMLDivElement>) {
    event.preventDefault();
    const initialValue = kind === "source"
      ? sourcePanelRef.current?.getBoundingClientRect().width ?? panelLayout.sourceWidth
      : kind === "settings"
        ? settingsPanelRef.current?.getBoundingClientRect().width ?? panelLayout.settingsWidth
        : galleryPanelRef.current?.getBoundingClientRect().height ?? panelLayout.galleryHeight;
    resizeDrag.current = { kind, startX: event.clientX, startY: event.clientY, initialValue };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveResize(event: PointerEvent<HTMLDivElement>) {
    const drag = resizeDrag.current;
    if (!drag) return;
    if (drag.kind === "source") {
      const width = Math.max(180, Math.min(sourceResizeMax, drag.initialValue + event.clientX - drag.startX));
      setPanelLayout((current) => ({ ...current, sourceWidth: width }));
    } else if (drag.kind === "settings") {
      const width = Math.max(260, Math.min(settingsResizeMax, drag.initialValue - (event.clientX - drag.startX)));
      setPanelLayout((current) => ({ ...current, settingsWidth: width }));
    } else {
      const height = Math.max(110, Math.min(300, drag.initialValue + drag.startY - event.clientY));
      setPanelLayout((current) => ({ ...current, galleryHeight: height }));
    }
  }

  function endResize() {
    resizeDrag.current = null;
  }

  function resizeByKeyboard(kind: ResizeKind, event: KeyboardEvent<HTMLDivElement>) {
    const step = event.shiftKey ? 32 : 12;
    if (kind === "gallery" && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
      event.preventDefault();
      setPanelLayout((current) => ({ ...current, galleryHeight: Math.max(110, Math.min(300, current.galleryHeight + (event.key === "ArrowUp" ? step : -step))) }));
    } else if (kind === "source" && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      event.preventDefault();
      setPanelLayout((current) => ({ ...current, sourceWidth: Math.max(180, Math.min(sourceResizeMax, current.sourceWidth + (event.key === "ArrowRight" ? step : -step))) }));
    } else if (kind === "settings" && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      event.preventDefault();
      setPanelLayout((current) => ({ ...current, settingsWidth: Math.max(260, Math.min(settingsResizeMax, current.settingsWidth + (event.key === "ArrowLeft" ? step : -step))) }));
    }
  }

  function splitterValue(kind: ResizeKind) {
    if (kind === "source") return { min: 180, max: sourceResizeMax, value: Math.min(panelLayout.sourceWidth, sourceResizeMax) };
    if (kind === "settings") return { min: 260, max: settingsResizeMax, value: Math.min(panelLayout.settingsWidth, settingsResizeMax) };
    return { min: 110, max: 300, value: panelLayout.galleryHeight };
  }

  function splitterProps(kind: ResizeKind, label: string) {
    const limits = splitterValue(kind);
    return {
      role: "separator" as const,
      "aria-orientation": kind === "gallery" ? "horizontal" as const : "vertical" as const,
      "aria-label": label,
      "aria-valuemin": Math.round(limits.min),
      "aria-valuemax": Math.round(limits.max),
      "aria-valuenow": Math.round(limits.value),
      tabIndex: 0,
      onPointerDown: (event: PointerEvent<HTMLDivElement>) => beginResize(kind, event),
      onPointerMove: moveResize,
      onPointerUp: endResize,
      onPointerCancel: endResize,
      onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => resizeByKeyboard(kind, event),
      title: "Kéo hoặc dùng phím mũi tên để chỉnh panel",
    };
  }

  async function mayReplaceCurrentProject() {
    if (!dirty) return true;
    return confirm("Thay đổi chưa được lưu sẽ mất. Bạn muốn tiếp tục?", {
      title: "Thay dự án hiện tại",
      kind: "warning",
      okLabel: "Tiếp tục",
      cancelLabel: "Ở lại",
    });
  }

  async function importMedia(paths: string[]) {
    setError(null);
    try {
      if (!paths.length) return;
      const projectFile = paths.find((path) => /\.ffproj$/i.test(path));
      if (projectFile) {
        // A project dropped or chosen here is opened, not treated as media.
        if (await mayReplaceCurrentProject()) await openProjectPath(projectFile);
        return;
      }
      const result = await callEngine<{
        sources: MediaSource[];
        unsupported_count: number;
        errors: { path: string; message: string }[];
      }>("inspect_sources", { paths });
      if (!result.sources.length) {
        setError(result.errors[0]?.message || "Không tìm thấy video hoặc ảnh được hỗ trợ trong lựa chọn.");
        return;
      }
      if (result.sources.every((source) => source.kind === "image")) {
        const direct = await confirm(
          `Đã chọn ${result.sources.length} ảnh. Thêm thẳng vào dự án để gán nhãn (ảnh dùng tại chỗ, nhãn lưu JSON cạnh ảnh)?

`
            + "Chọn \"Nguồn cắt frame\" nếu muốn cắt crop từ các ảnh này ở tab Trích xuất.",
          { title: "Thêm ảnh", kind: "info", okLabel: "Thêm để gán nhãn", cancelLabel: "Nguồn cắt frame" },
        );
        if (direct) {
          await importImages(paths);
          return;
        }
      }
      const existing = new Set(project.sources.map((source) => source.path));
      const added = result.sources.filter((source) => !existing.has(source.path));
      setProject((current) => ({ ...current, sources: [...current.sources, ...added] }));
      setSelectedPaths((selected) => [...new Set([...selected, ...result.sources.map((source) => source.path)])]);
      if (!selectedPath && added[0]) setSelectedPath(added[0].path);
      setDirty(true);
      const importNotices = [
        result.unsupported_count
          ? `${result.unsupported_count} tệp không thuộc định dạng media được hỗ trợ đã bị bỏ qua.`
          : "",
        result.errors.length
          ? `${result.errors.length} tệp bị từ chối hoặc không đọc được. ${result.errors[0]?.message ?? ""}`
          : "",
      ].filter(Boolean);
      if (importNotices.length) {
        setError(importNotices.join(" "));
      }
    } catch (reason) {
      setError(String(reason));
    }
  }

  async function chooseMedia(directory: boolean) {
    try {
      const chosen = await open({
        title: directory ? "Chọn thư mục media" : "Thêm video và ảnh",
        multiple: true,
        directory,
        ...(directory ? {} : { filters: MEDIA_FILTERS }),
      });
      const paths = Array.isArray(chosen) ? chosen : chosen ? [chosen] : [];
      await importMedia(paths);
    } catch (reason) {
      setError(String(reason));
    }
  }

  function removeSource(path: string) {
    setProject((current) => ({ ...current, sources: current.sources.filter((source) => source.path !== path) }));
    setSelectedPaths((current) => current.filter((selected) => selected !== path));
    if (selectedPath === path) {
      const next = project.sources.find((source) => source.path !== path);
      setSelectedPath(next?.path ?? null);
      setFrameIndex(0);
    }
    setDirty(true);
  }

  async function createProject() {
    if (!(await mayReplaceCurrentProject())) return;
    try {
      const path = await save({
        title: "Tạo dự án FrameForge",
        defaultPath: "FrameForge.ffproj",
        filters: [{ name: "Dự án FrameForge", extensions: ["ffproj"] }],
      });
      if (!path) return;
      const created = await callEngine<{ id: string; name: string }>("create_project", {
        path,
        name: projectNameFromPath(path),
      });
      const empty: ProjectState = {
        id: created.id,
        name: created.name,
        settings_version: PROJECT_SETTINGS_VERSION,
        settings_migrated: false,
        sources: [],
        settings: DEFAULT_OPTIONS,
        images: [],
        class_names: ["head", "body"],
      };
      setProject(empty);
      setProjectPath(path);
      rememberProject(path);
      setOptions(DEFAULT_OPTIONS);
      setSelectedPaths([]);
      setSelectedPath(null);
      setGalleryThumbnails({});
      setFrameIndex(0);
      setDirty(false);
      setError(null);
    } catch (reason) {
      setError(String(reason));
    }
  }

  async function openProject() {
    if (!(await mayReplaceCurrentProject())) return;
    try {
      const path = await open({
        title: "Mở dự án FrameForge",
        multiple: false,
        filters: [{ name: "Dự án FrameForge", extensions: ["ffproj"] }],
      });
      if (typeof path !== "string") return;
      await openProjectPath(path);
    } catch (reason) {
      setError(String(reason));
    }
  }

  /** Finds images again when the project (with its images) came from another folder or computer. */
  async function relocateProjectImages(path: string) {
    type Relocation = { image_count: number; missing_count: number; relocated_count: number; still_missing_count: number; ambiguous_count: number };
    let result = await callEngine<Relocation>("relocate_images", { path });
    const relocated = result.relocated_count;
    let total = relocated;
    if (result.still_missing_count > 0) {
      const pick = await confirm(
        `${result.still_missing_count}/${result.image_count} ảnh của dự án không còn ở đường dẫn cũ (dự án chép từ máy hoặc thư mục khác?).${relocated ? `

Đã tự tìm lại ${relocated} ảnh cạnh file dự án.` : ""}

Chọn thư mục chứa ảnh để FrameForge tìm lại theo tên file? Nhãn đi theo file JSON cạnh ảnh.`,
        { title: "Không tìm thấy ảnh", kind: "warning", okLabel: "Chọn thư mục ảnh", cancelLabel: "Mở luôn" },
      );
      if (pick) {
        const root = await open({ title: "Chọn thư mục chứa ảnh của dự án", directory: true, multiple: false });
        if (typeof root === "string") {
          result = await callEngine<Relocation>("relocate_images", { path, root });
          total += result.relocated_count;
        }
      }
    }
    if (result.still_missing_count > 0) {
      setError(`${result.still_missing_count} ảnh vẫn không tìm thấy${result.ambiguous_count ? ` (${result.ambiguous_count} ảnh trùng tên ở nhiều nơi)` : ""}. Mở lại dự án để chọn thư mục ảnh khác.`);
    } else if (total > 0) {
      await message(`Đã tìm lại ${total} ảnh ở vị trí mới và cập nhật dự án.`, { title: "Dự án đã chuyển chỗ" });
    }
  }

  async function openProjectPath(path: string) {
    await relocateProjectImages(path);
    const loaded = await callEngine<ProjectState>("load_project", { path });
    const nextOptions = { ...DEFAULT_OPTIONS, ...(loaded.settings as Partial<ExtractionOptions>), image_format: "png" as const };
    const settingsMigrated = loaded.settings_migrated === true;
    setProject({
      ...loaded,
      settings_version: PROJECT_SETTINGS_VERSION,
      settings_migrated: false,
      settings: nextOptions,
    });
    setOptions(nextOptions);
    setProjectPath(path);
    setSelectedPaths(loaded.sources.map((source) => source.path));
    setSelectedPath(loaded.sources[0]?.path ?? null);
    setFrameIndex(0);
    setGalleryThumbnails({});
    setDirty(settingsMigrated);
    setError(null);
    rememberProject(path);
  }

  async function saveProject(): Promise<string | null> {
    if (annotationSaveFailed.current) {
      setError("Một số nhãn chưa được tự lưu. Quay lại Gán nhãn, tải lại ảnh và thử lưu nhãn trước.");
      return null;
    }
    try {
      let path = projectPath;
      if (!path) {
        path = await save({
          title: "Lưu dự án FrameForge",
          defaultPath: `${project.name.replace(/[/\\:*?"<>|]/g, "-") || "FrameForge"}.ffproj`,
          filters: [{ name: "Dự án FrameForge", extensions: ["ffproj"] }],
        });
      }
      if (!path) return null;
      const saved = await serializeProjectWrite(() => callEngine<{ id: string; name: string; modified_at: string }>("save_project", {
        path,
        project: { ...project, settings: options },
      }));
      setProject((current) => ({
        ...current,
        id: saved.id,
        name: saved.name,
        settings_version: PROJECT_SETTINGS_VERSION,
        settings_migrated: false,
        settings: options,
      }));
      setProjectPath(path);
      rememberProject(path);
      setDirty(false);
      setError(null);
      return path;
    } catch (reason) {
      setError(String(reason));
      return null;
    }
  }

  async function importImages(givenPaths?: string[]) {
    setError(null);
    try {
      let path = projectPath;
      if (!path || dirty) {
        path = await saveProject();
        if (!path) return;
      }
      let paths = givenPaths ?? [];
      if (!givenPaths) {
        const chosen = await open({ title: "Chọn thư mục ảnh (ảnh dùng tại chỗ, nhãn lưu JSON cạnh ảnh)", directory: true, multiple: true });
        paths = Array.isArray(chosen) ? chosen : chosen ? [chosen] : [];
      }
      if (!paths.length) return;
      const projectFile = path;
      setImportingLabels(true);
      const preview = await callEngine<ImageImportPreview>("import_images", { path: projectFile, paths, dry_run: true });
      if (!preview.new_images) {
        await message(`${preview.found_images} ảnh đều đã có trong dự án.`, { title: "Nhập ảnh", kind: "info" });
        return;
      }
      // Only JSON from other tools (LabelMe/X-AnyLabeling) lacks FrameForge's review status.
      const markReviewed = preview.json_images > 0 && await confirm(
        `${preview.json_images} ảnh đã có file JSON nhãn. Đánh dấu các ảnh có box là đã review để xuất thẳng vào dataset?\n\n`
          + "Chọn \"Chưa review\" nếu muốn duyệt lại từng ảnh. JSON do FrameForge ghi giữ nguyên trạng thái review của nó.",
        { title: "Nhập ảnh", kind: "info", okLabel: "Đã review", cancelLabel: "Chưa review" },
      );
      const result = await serializeProjectWrite(() => callEngine<ImageImportResult>("import_images", {
        path: projectFile,
        paths,
        mark_reviewed: markReviewed,
      }));
      const loaded = await serializeProjectWrite(() => callEngine<ProjectState>("load_project", { path: projectFile }));
      if (projectPathRef.current === projectFile || projectPath === projectFile) {
        setProject((current) => ({ ...current, images: loaded.images, class_names: loaded.class_names }));
      }
      const lines = [
        `Đã thêm ${result.imported_images}/${result.found_images} ảnh: ${result.labeled_images} ảnh có ${result.box_count} box`
          + `${result.reviewed_images ? ` (${result.reviewed_images} ảnh đã review)` : ""}, ${result.unlabeled_images} ảnh chưa gán.`,
        "Nhãn được lưu vào file .json cạnh mỗi ảnh.",
        result.skipped_existing ? `${result.skipped_existing} ảnh đã có trong dự án nên giữ nguyên.` : "",
        result.unknown_labels ? `${result.unknown_labels} box có nhãn ngoài class dự án được giữ nguyên trong JSON nhưng không hiện trong FrameForge.` : "",
        result.unsupported_shapes ? `${result.unsupported_shapes} shape không phải rectangle/polygon được giữ nguyên trong JSON.` : "",
        result.clipped_boxes ? `${result.clipped_boxes} box vượt biên ảnh đã được cắt vào trong ảnh.` : "",
        result.degenerate_boxes ? `${result.degenerate_boxes} box nhỏ hơn 1 px bị bỏ qua.` : "",
        result.error_count ? `${result.error_count} tệp lỗi, ví dụ: ${result.errors[0]?.message ?? ""}` : "",
      ].filter(Boolean);
      await message(lines.join("\n"), { title: "Nhập ảnh", kind: result.error_count ? "warning" : "info" });
    } catch (reason) {
      setError(String(reason));
    } finally {
      setImportingLabels(false);
    }
  }

  async function openLabelWorkspace() {
    setError(null);
    if (project.images.length > 0 && (!projectPath || dirty)) {
      if (!(await saveProject())) return;
    }
    setActiveTab("label");
  }

  async function openDatasetWorkspace() {
    setError(null);
    if (project.images.length > 0 && (!projectPath || dirty)) {
      if (!(await saveProject())) return;
    }
    setActiveTab("dataset");
  }

  const refreshModels = useCallback(async () => {
    try {
      const [registry, capabilities] = await Promise.all([
        callEngine<{ runs: RegisteredModel[] }>("list_models"),
        callEngine<{ available?: boolean; preferred_device?: string; device_name?: string }>("inference_capabilities"),
      ]);
      setModels(registry.runs ?? []);
      setInferenceAvailable(capabilities.available === true);
      setInferenceDevice(capabilities);
    } catch (reason) {
      setError(String(reason));
    }
  }, []);

  useEffect(() => {
    if (engineStatus === "ready" && activeTab === "extract") void refreshModels();
  }, [engineStatus, activeTab, refreshModels]);

  async function importDetectionModel() {
    setError(null);
    setModelBusy(true);
    try {
      const imported = await pickAndImportCheckpoint(callEngine);
      if (!imported) return;
      await refreshModels();
      if (!modelMatchesClasses(imported, project.class_names)) {
        const labeled = project.images.some((image) => (image.annotations?.length ?? 0) > 0 || (image.negative_candidates?.length ?? 0) > 0);
        if (labeled || !(await confirm(
          `Model dùng class: ${imported.class_names.join(", ")}.\nDự án đang dùng: ${project.class_names.join(", ")}.\n\nDự án chưa có nhãn nào. Đổi class của dự án theo model để dùng model này?`,
          { title: "Dùng class của model", kind: "info", okLabel: "Dùng class của model", cancelLabel: "Giữ class hiện tại" },
        ))) {
          setError(`Đã nhập model (${imported.class_names.join(", ")}) nhưng class khác dự án nên chưa dùng được để lọc ảnh.`);
          return;
        }
        await applyModelClasses(imported.class_names);
      }
      updateOptions({ detect_model_id: imported.run_id, detect_enabled: true });
    } catch (reason) {
      setError(String(reason));
    } finally {
      setModelBusy(false);
    }
  }

  async function deleteDetectionModel() {
    const model = models.find((item) => item.run_id === options.detect_model_id);
    if (!model || extracting || modelBusy) return;
    if (!(await confirm(
      `Xóa model "${model.display_name ?? "YOLO11n"}" khỏi FrameForge?\n\nBox đã gán vẫn được giữ. File .pt gốc không bị xóa.`,
      { title: "Xóa model", kind: "warning", okLabel: "Xóa model", cancelLabel: "Hủy" },
    ))) return;
    setModelBusy(true);
    setError(null);
    try {
      await callEngine("delete_model", { run_id: model.run_id });
      updateOptions({ detect_model_id: "", detect_enabled: false });
      await refreshModels();
    } catch (reason) {
      setError(String(reason));
    } finally {
      setModelBusy(false);
    }
  }

  // Boxes found while extracting become AI suggestions once the new images are saved into the project.
  useEffect(() => {
    if (!pendingSuggestions) return;
    const pending = pendingSuggestions;
    setPendingSuggestions(null);
    void (async () => {
      const path = await saveProject();
      if (!path) {
        setError("Chưa lưu được dự án nên box model tìm thấy chưa được lưu thành gợi ý.");
        return;
      }
      try {
        const saved = await serializeProjectWrite(() => callEngine<{
          images_with_detections: number;
          annotation_count: number;
          candidate_negative_count: number;
          skipped_changed: number;
        }>("save_extraction_suggestions", { path, ...pending }));
        setDetectionSummary(`${saved.annotation_count} box AI trên ${saved.images_with_detections} ảnh chờ duyệt ở tab Gán nhãn${saved.candidate_negative_count ? ` · ${saved.candidate_negative_count} ảnh trống chờ xác nhận` : ""}${saved.skipped_changed ? ` · ${saved.skipped_changed} ảnh đổi nội dung bị bỏ qua` : ""}`);
      } catch (reason) {
        setError(`Ảnh đã vào dự án nhưng chưa lưu được box của model: ${String(reason)}`);
      } finally {
        await reloadLabels(path);
      }
    })();
  }, [pendingSuggestions]);

  async function chooseOutputDirectory() {
    try {
      const chosen = await open({ title: "Chọn thư mục lưu frame", directory: true, multiple: false });
      if (typeof chosen === "string") setOutputDirectory(chosen);
    } catch (reason) {
      setError(String(reason));
    }
  }

  async function startExtraction() {
    setPlaying(false);
    if (options.crop_mode === "object_centered" && !(options.detect_enabled ?? Boolean(options.detect_model_id))) {
      setError("Crop theo vật thể cần bật lọc bằng model và chọn checkpoint YOLO11n.");
      return;
    }
    if (!selectedSources.length) {
      setError("Hãy chọn ít nhất một nguồn ở danh sách bên trái.");
      return;
    }
    if (!options.interval || options.interval <= 0) {
      setError("Khoảng lấy mẫu phải lớn hơn 0.");
      return;
    }
    const minimumOutputSize = options.crop_mode === "center" || options.crop_mode === "object_centered" ? 1 : 16;
    if (!Number.isInteger(options.output_width) || !Number.isInteger(options.output_height)
      || options.output_width < minimumOutputSize || options.output_width > FHD_WIDTH
      || options.output_height < minimumOutputSize || options.output_height > FHD_HEIGHT) {
      setError(`Kích thước tối đa là ${FHD_WIDTH} × ${FHD_HEIGHT} px (FHD).`);
      return;
    }
    let detection: Record<string, unknown> | null = null;
    let extractionProjectPath = projectPath;
    if (options.detect_enabled ?? Boolean(options.detect_model_id)) {
      const model = availableModels(models).find((item) => item.run_id === options.detect_model_id);
      if (!model?.checkpoint_path) {
        setError("Model lọc ảnh không còn khả dụng. Chọn model khác hoặc tắt lọc bằng model.");
        return;
      }
      if (!modelMatchesClasses(model, project.class_names)) {
        setError(`Class của model (${model.class_names?.join(", ")}) khác class dự án (${project.class_names.join(", ")}). Đổi class dự án theo model ở tab Gán nhãn trước.`);
        return;
      }
      if (!Number.isInteger(options.detect_confidence) || options.detect_confidence < 1 || options.detect_confidence > 99) {
        setError("Ngưỡng confidence phải từ 1 đến 99%.");
        return;
      }
      // Boxes are saved into the project as AI suggestions, so it must exist on disk first.
      if (!extractionProjectPath) {
        extractionProjectPath = await saveProject();
        if (!extractionProjectPath) return;
      }
      detection = {
        checkpoint: model.checkpoint_path,
        confidence_threshold: options.detect_confidence / 100,
        class_names: project.class_names,
        keep_empty: options.detect_keep_empty,
        device: options.detect_use_gpu === false ? "cpu" : "gpu",
        trust_source: true,
      };
    }
    let destination = outputDirectory;
    if (!destination && extractionProjectPath) {
      // Default: a frames folder next to the project file, so extraction needs no folder dialog.
      const separator = extractionProjectPath.includes("\\") ? "\\" : "/";
      const parent = extractionProjectPath.slice(0, extractionProjectPath.lastIndexOf(separator));
      destination = `${parent}${separator}${projectNameFromPath(extractionProjectPath)}_frames`;
      setOutputDirectory(destination);
    }
    if (!destination) {
      try {
        const chosen = await open({ title: "Chọn thư mục lưu frame", directory: true, multiple: false });
        if (typeof chosen !== "string") return;
        destination = chosen;
        setOutputDirectory(chosen);
      } catch (reason) {
        setError(String(reason));
        return;
      }
    }
    if (options.existing_file_behavior === "overwrite") {
      const approved = await confirm(`Các frame trùng tên trong thư mục này sẽ bị thay thế:\n${destination}`, {
        title: "Xác nhận ghi đè",
        kind: "warning",
        okLabel: "Ghi đè file trùng",
        cancelLabel: "Hủy",
      });
      if (!approved) return;
    }

    const id = makeRequestId();
    await progressSubscription.current;
    activeExtractionId.current = id;
    setProgress(null);
    setDetectionSummary(null);
    setExtracting(true);
    setError(null);
    try {
      const result = await invoke<ExtractionResult>("engine_request", {
        request: {
          id,
          command: "extract_frames",
          arguments: {
            paths: selectedSources.map((source) => source.path),
            output_dir: destination,
            options: detection ? { ...options, detection } : options,
          },
        },
      });
      setProgress(result);
      setGalleryThumbnails(Object.fromEntries(result.files
        .filter((file) => file.thumbnail_data_url)
        .map((file) => [file.path, file.thumbnail_data_url as string])));
      setProject((current) => {
        const imagesByPath = new Map<string, ProjectImage>(current.images.map((image) => [image.path, image]));
        for (const written of result.images) {
          const { overwritten, ai_detection: _detection, ...image } = written;
          // An overwritten file has new pixels, so labels, AI candidates and attempts made on the old bytes no longer apply.
          const existing = overwritten ? undefined : imagesByPath.get(image.path);
          imagesByPath.set(image.path, {
            ...image,
            extraction_settings: result.extraction_settings,
            status: existing?.status ?? "ready",
            annotations: existing?.annotations ?? [],
            negative_candidates: existing?.negative_candidates ?? [],
            inference_attempts: existing?.inference_attempts ?? [],
          });
        }
        return { ...current, images: [...imagesByPath.values()] };
      });
      setDirty(true);
      if (result.detection) {
        const items = result.images.flatMap((image) => image.ai_detection ? [{ image_path: image.path, ...image.ai_detection }] : []);
        if (items.length) {
          setPendingSuggestions({ checkpoint: result.detection.checkpoint, checkpoint_sha256: result.detection.checkpoint_sha256, items });
        }
      }
      const overwrittenLabeled = result.images.filter((image) => image.overwritten && project.images.some((existing) => (
        existing.path === image.path
        && ((existing.annotations?.length ?? 0) > 0 || (existing.negative_candidates?.length ?? 0) > 0 || existing.status === "reviewed")
      ))).length;
      if (overwrittenLabeled) {
        setError(`${overwrittenLabeled} ảnh bị ghi đè đã được xóa nhãn và gợi ý AI cũ vì nội dung ảnh đã thay đổi.`);
      }
      if (result.errors.length) {
        const firstFailure = result.errors[0];
        setError(`Đã trích ${result.exported_images} ảnh; lỗi ở ${firstFailure.path}: ${firstFailure.message}`);
      }
    } catch (reason) {
      const message = String(reason);
      setError(message.includes("Đã hủy tác vụ") ? "Đã dừng trích xuất. Các ảnh đã ghi vẫn được giữ lại." : message);
    } finally {
      activeExtractionId.current = null;
      setExtracting(false);
    }
  }

  async function cancelExtraction() {
    const requestId = activeExtractionId.current;
    if (!requestId) return;
    try {
      await invoke<boolean>("cancel_engine_request", { requestId });
    } catch (reason) {
      setError(String(reason));
    }
  }

  function toggleSource(path: string) {
    setSelectedPaths((current) => current.includes(path)
      ? current.filter((value) => value !== path)
      : [...current, path]);
  }

  function selectGalleryImage(image: GalleryImage) {
    if (image.source_path && project.sources.some((source) => source.path === image.source_path)) {
      setSelectedPath(image.source_path);
      setFrameIndex(image.frame_index ?? 0);
    }
  }

  const loadGalleryThumbnails = useCallback((paths: string[]) => (
    callEngine<{ thumbnails: Record<string, string | null> }>("image_thumbnails", { paths })
      .then((result) => result.thumbnails)
  ), []);

  const extractionPercent = progress?.source_total === 1 && (progress.source_frames_total ?? 0) > 0
    ? Math.min(100, Math.round(((progress.scanned_frames ?? 0) / progress.source_frames_total!) * 100))
    : progress?.source_total ? Math.min(100, Math.round((progress.source_index / progress.source_total) * 100)) : 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <span className="brand-name">FrameForge</span>
          <span className="brand-version">0.1</span>
        </div>
        <div className="project-name-block">
          <span className="project-name">{project.name}{dirty ? <i aria-label="Chưa lưu"> •</i> : null}</span>
          <span className="project-location">{projectPath ?? "Dự án cục bộ · chưa lưu"}</span>
        </div>
        <div className="toolbar-actions">
          <button className="tool-button" type="button" onClick={() => void createProject()} title="Tạo dự án mới">
            <Plus size={15} /><span>Mới</span>
          </button>
          <button className="tool-button" type="button" onClick={() => void openProject()} title="Mở dự án">
            <FolderOpen size={15} /><span>Mở</span>
          </button>
          <button className="tool-button" type="button" onClick={() => void saveProject()} title="Lưu dự án">
            <SaveIcon size={15} /><span>Lưu</span>
          </button>
          <span className="toolbar-divider" />
          <button className="tool-button" type="button" onClick={() => void importImages()} disabled={engineStatus !== "ready" || importingLabels} title="Thêm thư mục ảnh có sẵn vào dự án; nhãn lưu file JSON cạnh từng ảnh (đọc được JSON LabelMe/X-AnyLabeling)">
            {importingLabels ? <LoaderCircle size={15} className="spin" /> : <FileJson size={15} />}<span>Nhập ảnh</span>
          </button>
          <button className="tool-button" type="button" onClick={() => void openLabelWorkspace()} title="Mở Gán nhãn để chọn checkpoint và chạy gợi ý AI">
            <Sparkles size={15} /><span>Auto label</span>
          </button>
          <button className="tool-button" type="button" onClick={() => void openDatasetWorkspace()} disabled={engineStatus !== "ready"} title="Kiểm tra và xuất dataset YOLO">
            <Download size={15} /><span>Xuất YOLO</span>
          </button>
        </div>
        <div className={`engine-badge engine-${engineStatus}`} title={engineStatus === "error" ? "Không kết nối được Python engine" : "Trạng thái Python engine"}>
          <span className="engine-dot" />
          <span>{engineStatus === "ready" ? "Engine sẵn sàng" : engineStatus === "checking" ? "Đang kết nối" : "Engine lỗi"}</span>
        </div>
      </header>

      <nav className="workflow-tabs" aria-label="Các giai đoạn">
        <button className={activeTab === "extract" ? "workflow-tab active" : "workflow-tab"} onClick={() => setActiveTab("extract")} type="button">
          <Scissors size={15} /><span>Trích xuất</span>
        </button>
        <button className={activeTab === "label" ? "workflow-tab active" : "workflow-tab"} type="button" onClick={() => void openLabelWorkspace()}>
          <Settings2 size={15} /><span>Gán nhãn</span>
        </button>
        <button className={activeTab === "dataset" ? "workflow-tab active" : "workflow-tab"} type="button" onClick={() => void openDatasetWorkspace()}>
          <Database size={15} /><span>Dataset</span>
        </button>
        <div className="tab-spacer" />
        <span className="workflow-hint"><CircleHelp size={13} /> Dữ liệu xử lý trên máy này</span>
      </nav>

      {error ? (
        <div className="error-banner" role="alert">
          <AlertCircle size={15} /><span>{error}</span>
          <button type="button" className="icon-button" onClick={() => setError(null)} aria-label="Đóng thông báo"><X size={14} /></button>
        </div>
      ) : null}

      {activeTab === "label" ? (
        <ManualLabeling
          projectKey={projectPath}
          deleteImage={deleteLabelImage}
          reviewImages={reviewLabelImages}
          images={project.images}
          classNames={project.class_names}
          loadPreview={loadLabelPreview}
          saveImage={saveLabelImage}
          saveSuggestions={saveLabelSuggestions}
          reviewNegativeCandidate={reviewLabelNegativeCandidate}
          runBatchInference={runProjectInference}
          applyModelClasses={applyModelClasses}
        />
      ) : activeTab === "dataset" ? (
        <DatasetManager projectPath={projectPath} imageCount={project.images.length} classNames={project.class_names} />
      ) : (
      <section
        className={`workspace-layout ${dropActive ? "drop-active" : ""}`}
        aria-label="Không gian trích xuất frame"
        style={{ "--source-width": `${panelLayout.sourceWidth}px`, "--settings-width": `${panelLayout.settingsWidth}px`, "--gallery-height": `${panelLayout.galleryHeight}px` } as CSSProperties}
      >
        <aside className="source-panel panel" ref={sourcePanelRef}>
          <div className="panel-heading source-heading">
            <div>
              <h1>Nguồn media</h1>
              <span>{project.sources.length} tệp · {selectedSources.length} đã chọn</span>
            </div>
            <button className="icon-button" type="button" onClick={() => void chooseMedia(false)} aria-label="Thêm media" title="Thêm video hoặc ảnh">
              <Plus size={16} />
            </button>
          </div>
          <div className="source-actions">
            <button type="button" className="subtle-button" onClick={() => void chooseMedia(false)}><ImagePlus size={14} /> Thêm tệp</button>
            <button type="button" className="subtle-button" onClick={() => void chooseMedia(true)}><FolderPlus size={14} /> Thư mục</button>
          </div>
          <div className="source-list" role="list" aria-label="Danh sách nguồn">
            {project.sources.map((source) => {
              const selected = source.path === selectedPath;
              const checked = selectedPaths.includes(source.path);
              const Icon = source.kind === "video" ? FileVideo2 : FileImage;
              return (
                <div className={`source-row ${selected ? "selected" : ""}`} key={source.path} role="listitem">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleSource(source.path)}
                    aria-label={`Chọn ${source.name} để trích xuất`}
                  />
                  <button type="button" className="source-select" onClick={() => { setSelectedPath(source.path); setFrameIndex(0); setPlaying(false); }}>
                    <Icon size={16} />
                    <span className="source-details">
                      <strong title={source.name}>{source.name}</strong>
                      <small>
                        {source.width} × {source.height}
                        {` · ${source.kind === "video" ? formatTime(source.duration_seconds) : "Ảnh"} · ${formatBytes(source.size_bytes)}`}
                      </small>
                    </span>
                  </button>
                  <button type="button" className="source-remove icon-button" onClick={() => removeSource(source.path)} aria-label={`Gỡ ${source.name}`} title="Gỡ khỏi dự án"><X size={13} /></button>
                </div>
              );
            })}
            {project.sources.length === 0 ? (
              <div className="source-empty">
                <FolderOpen size={22} />
                <p>Thêm video hoặc ảnh để bắt đầu.</p>
              </div>
            ) : null}
          </div>
          <div className="source-footnote">
            <span className="key-hint">⌘</span> chọn nguồn cần xử lý
          </div>
        </aside>

        {dropActive && <div className="drop-overlay" aria-live="polite"><FolderPlus size={25} /><strong>Thả video hoặc ảnh để thêm vào dự án</strong><span>FrameForge chỉ đọc tệp trên máy này</span></div>}

        <div className="split-handle split-handle-vertical split-source" {...splitterProps("source", "Điều chỉnh chiều rộng danh sách nguồn")} />

        <section className="preview-panel panel" aria-label="Preview nguồn media">
          <div className="preview-header">
            <div className="preview-title">
              <span className="live-indicator" />
              <strong>{selectedSource?.name ?? "Preview"}</strong>
            </div>
            <div className="preview-tools">
              <button type="button" className="detect-import-button" disabled={!preview || previewDetecting || extracting || modelBusy || !options.detect_model_id || engineStatus !== "ready"} onClick={() => { setShowDetections(true); setPreviewRefresh((value) => value + 1); }} title="Box tự cập nhật theo frame; bấm để chạy lại frame hiện tại">
                {previewDetecting ? <LoaderCircle size={13} className="spin" /> : <ScanLine size={13} />} {previewDetecting ? "Đang detect…" : "Cập nhật box"}
              </button>
              <span className="preview-resolution">
                {preview
                  ? preview.source_width && preview.source_height
                    && (preview.source_width !== preview.width || preview.source_height !== preview.height)
                    ? `${preview.source_width} × ${preview.source_height} → ${preview.width} × ${preview.height}`
                    : `${preview.width} × ${preview.height}`
                  : selectedSource ? `${selectedSource.width} × ${selectedSource.height}` : "Chưa có nguồn"}
              </span>
            </div>
          </div>
          <div className="preview-stage" ref={previewStageRef}>
            {preview && previewFrameBox ? (
              <div className="preview-frame" style={{ width: previewFrameBox.width, height: previewFrameBox.height }}>
                <img
                  className="preview-image"
                  src={preview.data_url}
                  alt={`Frame ${frameIndex + 1} từ ${selectedSource?.name ?? "nguồn"}`}
                />
                {options.crop_mode === "center" && cropRect && <div
                  className={`crop-overlay${cropFitsPreview ? "" : " invalid"}`}
                  style={{
                    left: `${cropRect.x / preview.width * 100}%`,
                    top: `${cropRect.y / preview.height * 100}%`,
                    width: `${cropRect.width / preview.width * 100}%`,
                    height: `${cropRect.height / preview.height * 100}%`,
                  }}
                  aria-hidden="true"
                ><span>Vùng crop · {options.output_width} × {options.output_height} px</span></div>}
                {options.crop_mode === "object_centered" && detectionPreview?.crop_rects?.map((rect, index) => <div key={`crop-${index}`} className="crop-overlay object-crop-overlay" style={{ left: `${rect.x / preview.width * 100}%`, top: `${rect.y / preview.height * 100}%`, width: `${rect.width / preview.width * 100}%`, height: `${rect.height / preview.height * 100}%` }} aria-hidden="true"><span>Crop {index + 1} · {rect.width} × {rect.height}</span></div>)}
                {showDetections && detectionPreview?.predictions?.map((item, index) => <div key={`box-${index}`} className="detection-overlay" style={{ left: `${item.bbox_xyxy[0] / preview.width * 100}%`, top: `${item.bbox_xyxy[1] / preview.height * 100}%`, width: `${(item.bbox_xyxy[2] - item.bbox_xyxy[0]) / preview.width * 100}%`, height: `${(item.bbox_xyxy[3] - item.bbox_xyxy[1]) / preview.height * 100}%` }} aria-hidden="true"><span>{project.class_names[item.class_id]} {(item.confidence * 100).toFixed(0)}%</span></div>)}
              </div>
            ) : previewError ? (
              <div className="preview-empty preview-error"><AlertCircle size={26} /><p>Không tải được frame</p><small>{previewError}</small></div>
            ) : (
              <div className="preview-empty">
                {selectedSource ? <LoaderCircle className="spin" size={25} /> : <FilmGlyph />}
                <p>{selectedSource ? "Đang đọc frame nguồn…" : "Chọn hoặc thêm video / ảnh"}</p>
                <small>{selectedSource?.kind === "video" ? "Chỉ nhận video tối đa FHD (1920 × 1080); video lớn hơn sẽ bị từ chối." : selectedSource ? "Đang tải ảnh nguồn thật, không có dữ liệu minh họa." : "Preview hiển thị nội dung media thật, không có dữ liệu minh họa."}</small>
                {!selectedSource && <button type="button" className="primary-button" onClick={() => void chooseMedia(false)}><Plus size={15} /> Thêm media</button>}
              </div>
            )}
          </div>
          <div className="preview-detection-status">
            <label><input type="checkbox" checked={showDetections} onChange={(event) => setShowDetections(event.target.checked)} /> Hiện detection trên preview</label>
            <span role="status">{previewDetecting ? "Đang detect và vẽ box lên ảnh…" : !showDetections ? "Đã tắt box preview." : detectionPreview ? `${detectionPreview.predictions?.length ?? 0} vật thể${options.crop_mode === "object_centered" ? ` · ${detectionPreview.crop_rects?.length ?? 0} vùng crop` : ""} · tự cập nhật theo frame` : !previewModel || !modelMatchesClasses(previewModel, project.class_names) ? "Chọn model có class khớp dự án trong mục Lọc bằng model để hiện box." : "Box sẽ tự hiện trên ảnh sau khi model xử lý frame."}</span>
          </div>
          {detectionPreviewError && <p className="preview-detection-error" role="alert">{detectionPreviewError}</p>}
          <div className="player-controls">
            <div className="transport-controls">
              <button type="button" className="icon-button" disabled={!selectedSource || selectedSource.kind === "image"} onClick={() => updateFrame(0)} aria-label="Frame đầu" title="Frame đầu"><ChevronsLeft size={16} /></button>
              <button type="button" className="icon-button" disabled={!selectedSource || selectedSource.kind === "image"} onClick={() => updateFrame(frameIndex - 1)} aria-label="Frame trước" title="Frame trước"><ChevronLeft size={16} /></button>
              <button type="button" className="play-button" disabled={!selectedSource || selectedSource.kind === "image" || extracting} onClick={() => {
                if (!playing && sourceFrameCount && frameIndex >= lastFrame) setFrameIndex(0);
                setPlaying((current) => !current);
              }} aria-label={playing ? "Tạm dừng" : "Phát"} title={playing ? "Tạm dừng" : "Phát"}>
                {playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}
              </button>
              <button type="button" className="icon-button" disabled={!selectedSource || selectedSource.kind === "image" || (sourceFrameCount != null && frameIndex >= sourceFrameCount - 1)} onClick={() => updateFrame(frameIndex + 1)} aria-label="Frame tiếp" title="Frame tiếp"><ChevronRight size={16} /></button>
              <button type="button" className="icon-button" disabled={!selectedSource || selectedSource.kind === "image" || sourceFrameCount == null} onClick={() => updateFrame(lastFrame)} aria-label="Frame cuối" title="Frame cuối"><ChevronsRight size={16} /></button>
            </div>
            <input
              className="timeline-slider"
              type="range"
              min={0}
              max={sourceFrameCount ? Math.max(1, sourceFrameCount - 1) : 1000}
              value={sourceFrameCount ? Math.min(frameIndex, Math.max(1, sourceFrameCount - 1)) : Math.min(frameIndex, 1000)}
              onChange={(event) => updateFrame(Number(event.target.value))}
              disabled={!selectedSource || selectedSource.kind === "image" || sourceFrameCount === 1}
              aria-label="Vị trí frame trong video"
            />
            <span className="frame-counter">{selectedSource?.kind === "image" ? "Ảnh tĩnh" : `Frame ${String(frameIndex + 1).padStart(6, "0")}${sourceFrameCount ? ` / ${String(sourceFrameCount).padStart(6, "0")}` : ""}`}</span>
            <span className="time-counter">{formatTime(currentTime)}{duration != null ? ` / ${formatTime(duration)}` : ""}</span>
          </div>
        </section>

        <div className="split-handle split-handle-vertical split-settings" {...splitterProps("settings", "Điều chỉnh chiều rộng cài đặt")} />

        <aside className="settings-panel panel" aria-label="Cài đặt trích xuất" ref={settingsPanelRef}>
          <div className="settings-header"><h2>Cài đặt trích xuất</h2><Settings2 size={16} /></div>
          <div className="settings-scroll">
            <section className="settings-section">
              <div className="section-title-row"><h3>Lấy frame</h3><span className="section-caption">Sampling</span></div>
              <div className="form-row compact-row">
                <label htmlFor="sample-interval">Mỗi</label>
                <input id="sample-interval" type="number" min="1" step="1" value={options.interval} onChange={(event) => updateOptions({ interval: Number(event.target.value) })} />
                <select aria-label="Đơn vị lấy mẫu" value={options.interval_unit} onChange={(event) => updateOptions({ interval_unit: event.target.value as ExtractionOptions["interval_unit"] })}>
                  <option value="frames">frame</option><option value="seconds">giây</option>
                </select>
              </div>
              <div className="form-grid two-col time-range-row">
                <label className="field-label">Bắt đầu (s)<input type="number" min="0" step="0.1" value={options.start_seconds} onChange={(event) => updateOptions({ start_seconds: Number(event.target.value) })} /></label>
                <label className="field-label">Kết thúc (s)<input type="number" min="0" step="0.1" placeholder="Cuối video" value={options.end_seconds ?? ""} onChange={(event) => updateOptions({ end_seconds: event.target.value ? Number(event.target.value) : null })} /></label>
              </div>
              <label className="field-label max-frames-field">Giới hạn tổng frame <span>0 = không giới hạn</span><input type="number" min="0" step="1" value={options.max_frames} onChange={(event) => updateOptions({ max_frames: Number(event.target.value) })} /></label>
              <SettingToggle label="Chọn frame nét nhất mỗi khoảng" checked={options.pick_sharpest !== false} disabled={extracting} onChange={(checked) => updateOptions({ pick_sharpest: checked })} />
              <p className="setting-note">{options.pick_sharpest !== false ? "Đọc cả khoảng để chọn frame nét; tốn thêm thời gian giải mã." : "Lấy một frame cố định mỗi khoảng."} Mỗi 10 frame là lấy khoảng 1/10 video.</p>
              <SettingToggle label="Lọc frame gần giống" checked={options.filter_duplicates} onChange={(checked) => updateOptions({ filter_duplicates: checked })} />
              {options.filter_duplicates && <>
                <ThresholdControl label="Ngưỡng dHash" value={options.duplicate_threshold} min={0} max={24} suffix="bit" onChange={(value) => updateOptions({ duplicate_threshold: value })} />
                <SettingToggle label="Xác nhận trùng bằng nội dung" checked={options.verify_duplicates !== false} disabled={extracting} onChange={(checked) => updateOptions({ verify_duplicates: checked })} />
                <p className="setting-note">So với tối đa 4.096 ảnh đã ghi, lệch ≤ {options.duplicate_threshold} bit dHash. {options.verify_duplicates !== false ? "Kiểm tra thêm màu, thay đổi cục bộ và box model để giữ lại mục tiêu thay đổi." : "Chỉ so dHash: có thể bỏ nhầm ảnh có vật thể nhỏ thay đổi."} Ngưỡng cao hơn lọc mạnh hơn.</p>
              </>}
              <SettingToggle label="Lọc ảnh mờ" checked={options.filter_blurry} onChange={(checked) => updateOptions({ filter_blurry: checked })} />
              {options.filter_blurry && <ThresholdControl label="Độ nét Laplacian" value={options.blur_threshold} min={0} max={250} suffix="" onChange={(value) => updateOptions({ blur_threshold: value })} />}
              <p className="setting-note" role="status">Frame đang xem: {!selectedSource ? "chưa có nguồn" : preview?.source_path === selectedSource.path && preview.frame_index === frameIndex && preview.sharpness_score != null ? `độ nét ${preview.sharpness_score.toFixed(0)} (toàn frame nguồn)` : "đang đọc…"}</p>
              {options.filter_blurry && <SettingToggle label="Lọc mờ chặt cho dữ liệu train" checked={options.strict_sharpness !== false} disabled={extracting} onChange={(checked) => updateOptions({ strict_sharpness: checked })} />}
              <p className="setting-note">Lọc chặt kiểm tra chi tiết viền và độ nét vùng mục tiêu khi dùng model. Video mềm do nén/chuyển động có thể giữ ít ảnh; không làm nét giả.</p>
            </section>

            <section className="settings-section">
              <div className="section-title-row"><h3>Lọc bằng model</h3><span className="section-caption">YOLO11n</span></div>
              <SettingToggle label="Sử dụng model để lọc" checked={options.detect_enabled ?? Boolean(options.detect_model_id)} disabled={extracting || modelBusy} onChange={(checked) => updateOptions({ detect_enabled: checked })} />
              <div className="detect-model-row">
                <select aria-label="Model lọc và gán nhãn khi trích xuất" value={options.detect_model_id} disabled={extracting || modelBusy} onChange={(event) => updateOptions({ detect_model_id: event.target.value, detect_enabled: Boolean(event.target.value) })}>
                  <option value="">Chọn model</option>
                  {availableModels(models).map((model) => {
                    const matches = modelMatchesClasses(model, project.class_names);
                    return <option key={model.run_id} value={model.run_id} disabled={!matches}>{model.display_name ?? "YOLO11n"} · {model.class_names?.join(", ")}{matches ? "" : " (class khác dự án)"}</option>;
                  })}
                </select>
                <button type="button" className="detect-import-button" disabled={extracting || modelBusy || engineStatus !== "ready"} onClick={() => void importDetectionModel()} title="Nhập checkpoint YOLO11n .pt">
                  {modelBusy ? <LoaderCircle size={14} className="spin" /> : <Upload size={14} />} Nhập .pt
                </button>
              </div>
              <button type="button" className="detect-import-button detect-delete-button" disabled={extracting || modelBusy || !options.detect_model_id || engineStatus !== "ready"} onClick={() => void deleteDetectionModel()}>Xóa model đang chọn</button>
              {inferenceAvailable === false && <p className="crop-validation" role="alert">Engine chưa có PyTorch/Ultralytics nên chưa lọc được bằng model.</p>}
              {(options.detect_enabled ?? Boolean(options.detect_model_id)) && <>
                <SettingToggle label="Sử dụng GPU" checked={options.detect_use_gpu !== false} disabled={extracting || modelBusy} onChange={(checked) => updateOptions({ detect_use_gpu: checked })} />
                <p className="setting-note">{options.detect_use_gpu === false ? "Model xử lý bằng CPU." : inferenceDevice?.preferred_device && inferenceDevice.preferred_device !== "cpu" ? `GPU: ${inferenceDevice.device_name ?? inferenceDevice.preferred_device}` : "GPU chưa khả dụng trong engine. Tắt GPU để dùng CPU."}</p>
                <label className="field-label detect-confidence-field">Ngưỡng confidence (%)<input type="number" min="1" max="99" step="1" value={options.detect_confidence} disabled={extracting} onChange={(event) => updateOptions({ detect_confidence: Number(event.target.value) })} /></label>
                <SettingToggle label="Giữ cả crop không có vật thể" checked={options.detect_keep_empty} onChange={(checked) => updateOptions({ detect_keep_empty: checked })} />
                <p className="setting-note">{options.detect_keep_empty
                  ? "Mọi crop đều được ghi; crop không có box thành ảnh trống chờ xác nhận ở tab Gán nhãn."
                  : "Crop không có vật thể (kể cả chỉ có box chạm biên crop) bị bỏ, không ghi ra đĩa."} Box tìm được lưu thành gợi ý AI; tab Gán nhãn chỉ còn duyệt/sửa.</p>
              </>}
            </section>

            <section className="settings-section">
              <div className="section-title-row"><h3>Chế độ crop</h3></div>
              <label className={`radio-option ${options.crop_mode === "center" ? "checked" : ""}`}>
                <input type="radio" name="crop-mode" value="center" checked={options.crop_mode === "center"} onChange={() => updateOptions({ crop_mode: "center" })} />
                <span><strong>Exact Center Crop</strong><small>Cắt đúng pixel tại tâm frame đã xử lý</small></span>
              </label>
              <label className={`radio-option ${options.crop_mode === "letterbox" ? "checked" : ""}`}>
                <input type="radio" name="crop-mode" value="letterbox" checked={options.crop_mode === "letterbox"} onChange={() => updateOptions({ crop_mode: "letterbox" })} />
                <span><strong>Full frame</strong><small>Giữ tỷ lệ, thêm viền</small></span>
              </label>
              <label className={`radio-option ${options.crop_mode === "grid" ? "checked" : ""}`}>
                <input type="radio" name="crop-mode" value="grid" checked={options.crop_mode === "grid"} onChange={() => updateOptions({ crop_mode: "grid" })} />
                <span><strong>Grid / sliding window</strong><small>Crop lưới chồng lấp</small></span>
              </label>
              {options.crop_mode === "grid" && <div className="form-grid two-col overlap-controls">
                <label className="field-label">Chồng ngang<input type="number" min="0" max="0.9" step="0.05" value={options.overlap_x} onChange={(event) => updateOptions({ overlap_x: Number(event.target.value) })} /></label>
                <label className="field-label">Chồng dọc<input type="number" min="0" max="0.9" step="0.05" value={options.overlap_y} onChange={(event) => updateOptions({ overlap_y: Number(event.target.value) })} /></label>
              </div>}
              <label className={`radio-option ${options.crop_mode === "object_centered" ? "checked" : ""}`}>
                <input type="radio" name="crop-mode" value="object_centered" checked={options.crop_mode === "object_centered"} disabled={extracting} onChange={() => updateOptions({ crop_mode: "object_centered", detect_enabled: true })} />
                <span><strong>Crop theo vật thể (YOLO11n)</strong><small>Detect frame nguồn rồi crop, giữ nguyên pixel</small></span>
              </label>
              {options.crop_mode === "object_centered" && <p className="setting-note">Chỉ crop khi model thấy vật thể vừa khung W × H. Vật thể lớn hơn khung: bỏ qua. Không detect lại từng crop; box được chuyển tọa độ từ frame nguồn.</p>}
              <div className="unavailable-crops">
                <span title="Công cụ chọn crop chưa được triển khai">Thủ công <small>Sắp có</small></span>
              </div>
            </section>

            <section className="settings-section">
              <div className="section-title-row"><h3>Kích thước</h3><span className="section-caption">{options.crop_mode === "center" ? "crop · px" : "output · px"}</span></div>
              <div className="preset-row" role="group" aria-label="Preset kích thước crop vuông">
                {[192, 256, 320, 416, 512, 640].map((size) => <button key={size} type="button" className={options.output_width === size && options.output_height === size ? "size-preset active" : "size-preset"} aria-pressed={options.output_width === size && options.output_height === size} onClick={() => updateOptions({ output_width: size, output_height: size })}>{size}</button>)}
              </div>
              <div className="form-grid two-col dimension-row">
                <label className="field-label">Rộng<input type="number" min={minimumOutputSize} max={FHD_WIDTH} step="1" value={options.output_width} onChange={(event) => updateOptions({ output_width: Number(event.target.value) })} /></label>
                <label className="field-label">Cao<input type="number" min={minimumOutputSize} max={FHD_HEIGHT} step="1" value={options.output_height} onChange={(event) => updateOptions({ output_height: Number(event.target.value) })} /></label>
              </div>
              <div className="form-grid two-col output-options">
                <div className="field-label">Định dạng<strong className="output-format">PNG · không mất dữ liệu</strong></div>
                {options.crop_mode !== "center" && <label className="field-label">Viền<input type="color" value={options.padding_color} onChange={(event) => updateOptions({ padding_color: event.target.value })} /></label>}
              </div>
              {options.crop_mode === "center" && !cropSizeValid
                ? <p className="crop-validation" role="alert">Kích thước tối đa là 1920 × 1080 px (FHD).</p>
                : options.crop_mode === "center" && preview && !cropFitsPreview
                  ? <p className="crop-validation" role="alert">Crop {options.output_width} × {options.output_height} vượt khung nguồn {preview.width} × {preview.height}. Hãy giảm kích thước crop.</p>
                  : options.crop_mode === "center" && <p className="crop-helper">Chỉ nhận video tối đa FHD; tâm crop dùng floor((khung − kích thước) / 2) và không resize vùng crop.</p>}
            </section>

            <section className="settings-section output-section">
              <div className="section-title-row"><h3>Đầu ra</h3><span className="section-caption">Output</span></div>
              <button className="output-path-button" type="button" onClick={() => void chooseOutputDirectory()}>
                <FolderOpen size={14} /><span title={outputDirectory || "Chọn thư mục lưu frame"}>{outputDirectory || (projectPath ? `Mặc định: ${projectNameFromPath(projectPath)}_frames cạnh dự án` : "Chọn thư mục lưu frame")}</span><ChevronDown size={14} />
              </button>
              <label className="field-label collision-field">Tệp đã tồn tại<select value={options.existing_file_behavior} onChange={(event) => updateOptions({ existing_file_behavior: event.target.value as ExtractionOptions["existing_file_behavior"] })}>
                <option value="skip">Bỏ qua, giữ tệp cũ</option><option value="rename">Đổi tên tệp mới</option><option value="overwrite">Ghi đè</option>
              </select></label>
            </section>
          </div>
          <div className="extract-actions">
            {extracting ? (
              <button type="button" className="extract-button cancel" onClick={() => void cancelExtraction()}><X size={16} /> Dừng trích xuất</button>
            ) : (
              <button type="button" className="extract-button" onClick={() => void startExtraction()} disabled={engineStatus !== "ready" || !selectedSources.length || previewDetecting}>
                <Scissors size={16} /> Trích xuất frame <span>{selectedSources.length || "—"}</span>
              </button>
            )}
            {extracting && <div className="extract-progress"><div className="progress-track"><span style={{ width: `${extractionPercent}%` }} /></div><small>{progress?.source_name || "Đang chuẩn bị"} · {progress?.candidate_frames ?? 0} frame</small></div>}
          </div>
        </aside>

        <div className="split-handle split-handle-horizontal" {...splitterProps("gallery", "Điều chỉnh chiều cao gallery")} />

        <section className="gallery-panel panel" aria-label="Frame vừa trích" ref={galleryPanelRef}>
          <div className="gallery-heading">
            <div><h2>Ảnh trong dự án</h2><span>{project.images.length} ảnh · cuộn ngang để xem toàn bộ</span></div>
            <div className="gallery-summary">
              {progress && <span><Check size={13} /> {progress.exported_images} ảnh xuất</span>}
              {progress && <span>{progress.candidate_frames} frame đã lấy mẫu / {progress.source_frames_total ?? "?"} frame nguồn</span>}
              {progress && (progress.sampling_skipped_frames ?? 0) > 0 && <span>{progress.sampling_skipped_frames} frame không chọn theo khoảng/mốc thời gian</span>}
              {progress && (progress.source_frames_total ?? 0) > (progress.scanned_frames ?? 0) && <span>{(progress.source_frames_total ?? 0) - (progress.scanned_frames ?? 0)} frame chưa xử lý/ngoài giới hạn</span>}
              {progress && progress.skipped_files > 0 && <span>{progress.skipped_files} tệp cũ giữ nguyên</span>}
              {progress && progress.duplicate_crops > 0 && <span>{progress.duplicate_crops} crop gần giống đã lọc</span>}
              {progress && progress.blurry_crops > 0 && <span>{progress.blurry_crops} crop mờ</span>}
              {progress && (progress.no_object_crops ?? 0) > 0 && <span>{progress.no_object_crops} crop không có vật thể đã bỏ</span>}
              {detectionSummary && <span><Sparkles size={13} /> {detectionSummary}</span>}
              {progress && <span>{progress.elapsed_seconds}s</span>}
            </div>
          </div>
          <ProjectGallery
            images={project.images}
            seedThumbnails={galleryThumbnails}
            loadThumbnails={loadGalleryThumbnails}
            onSelect={selectGalleryImage}
          />
        </section>
      </section>
      )}

      <footer className="statusbar">
        <div className="status-left">
          <span className={`status-light ${engineStatus}`} />
          <span>{extracting ? "Đang trích xuất" : engineStatus === "ready" ? "Sẵn sàng" : engineStatus === "checking" ? "Đang kết nối engine" : "Engine không khả dụng"}</span>
          {selectedSource && <><span className="status-separator">/</span><span>{selectedSource.kind === "video" ? `${selectedSource.fps?.toFixed(2) ?? "—"} FPS` : "Ảnh"}</span></>}
          {dirty && <><span className="status-separator">/</span><span>Chưa lưu</span></>}
        </div>
        <div className="status-right">
          <span>{outputDirectory || (projectPath ? `${projectNameFromPath(projectPath)}_frames cạnh dự án` : "Đầu ra chưa chọn")}</span>
          <span className="status-separator">/</span>
          <span>FrameForge · {activeTab === "label" ? "Gán nhãn" : activeTab === "dataset" ? "Dataset" : "Trích xuất"}</span>
        </div>
      </footer>
    </main>
  );
}

function SettingToggle({ label, checked, onChange, disabled = false }: { label: string; checked: boolean; onChange: (value: boolean) => void; disabled?: boolean }) {
  return (
    <label className="setting-toggle">
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-track"><span /></span>
      <span>{label}</span>
    </label>
  );
}

function ThresholdControl({ label, value, min, max, suffix, onChange }: { label: string; value: number; min: number; max: number; suffix: string; onChange: (value: number) => void }) {
  return (
    <label className="threshold-control">
      <span>{label}</span>
      <input type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} />
      <strong>{value}{suffix ? ` ${suffix}` : ""}</strong>
    </label>
  );
}

function FilmGlyph() {
  return <FileVideo2 size={27} strokeWidth={1.35} />;
}

export default App;
