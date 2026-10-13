import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from "react";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save, confirm } from "@tauri-apps/plugin-dialog";
import {
  AlertCircle,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  CircleHelp,
  FileImage,
  FileVideo2,
  FolderOpen,
  FolderPlus,
  HardDrive,
  ImagePlus,
  LoaderCircle,
  Pause,
  Play,
  Plus,
  Save as SaveIcon,
  Scissors,
  Settings2,
  Sparkles,
  X,
} from "lucide-react";
import "./App.css";
import ManualLabeling, { type Annotation } from "./ManualLabeling";

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
  blur_threshold: number;
  crop_mode: "center" | "letterbox" | "grid";
  output_width: number;
  output_height: number;
  overlap_x: number;
  overlap_y: number;
  padding_color: string;
  image_format: "jpg" | "png";
  jpeg_quality: number;
  existing_file_behavior: "skip" | "overwrite" | "rename";
};

type ProjectState = {
  id?: string;
  name: string;
  sources: MediaSource[];
  settings: ExtractionOptions;
  images: ProjectImage[];
  class_names: string[];
};

type FramePreview = {
  data_url: string;
  width: number;
  height: number;
  frame_index: number;
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
};

type ExtractionFile = ProjectImage & {
  name: string;
  thumbnail_data_url?: string;
};

type ExtractionResult = EngineProgress & {
  files: ExtractionFile[];
  file_list_truncated: boolean;
  errors: { path: string; message: string }[];
};

type ModelInfo = {
  name: string;
  inputs: { name: string; shape: (number | string | null)[]; type: string }[];
  outputs: { name: string; shape: (number | string | null)[]; type: string }[];
  providers: string[];
  class_names: string[];
};

type PanelLayout = { sourceWidth: number; settingsWidth: number; galleryHeight: number };
type ResizeKind = "source" | "settings" | "gallery";
type ResizeDrag = { kind: ResizeKind; startX: number; startY: number; initialValue: number };

const DEFAULT_OPTIONS: ExtractionOptions = {
  interval: 10,
  interval_unit: "frames",
  start_seconds: 0,
  end_seconds: null,
  max_frames: 0,
  filter_duplicates: true,
  duplicate_threshold: 5,
  filter_blurry: true,
  blur_threshold: 60,
  crop_mode: "center",
  output_width: 320,
  output_height: 320,
  overlap_x: 0.2,
  overlap_y: 0.2,
  padding_color: "#000000",
  image_format: "jpg",
  jpeg_quality: 92,
  existing_file_behavior: "skip",
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

function projectNameFromPath(path: string) {
  return path.split(/[\\/]/).pop()?.replace(/\.ffproj$/i, "") || "Dự án FrameForge";
}

function App() {
  const [project, setProject] = useState<ProjectState>({
    name: "Dự án chưa lưu",
    sources: [],
    settings: DEFAULT_OPTIONS,
    images: [],
    class_names: ["head", "body"],
  });
  const [projectPath, setProjectPath] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [preview, setPreview] = useState<FramePreview | null>(null);
  const [previewStageSize, setPreviewStageSize] = useState({ width: 0, height: 0 });
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [outputDirectory, setOutputDirectory] = useState("");
  const [gallery, setGallery] = useState<ExtractionFile[]>([]);
  const [progress, setProgress] = useState<EngineProgress | null>(null);
  const [fileListTruncated, setFileListTruncated] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [engineStatus, setEngineStatus] = useState<"checking" | "ready" | "error">("checking");
  const [error, setError] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [inspectingModel, setInspectingModel] = useState(false);
  const [dropActive, setDropActive] = useState(false);
  const [activeTab, setActiveTab] = useState<"extract" | "label" | "export">("extract");
  const [panelLayout, setPanelLayout] = useState(readPanelLayout);
  const projectWriteQueue = useRef<Promise<void>>(Promise.resolve());
  const annotationSaveFailed = useRef(false);
  const previewSequence = useRef(0);
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
  const minimumOutputSize = options.crop_mode === "center" ? 1 : 16;
  const cropSizeValid = Number.isInteger(options.output_width) && Number.isInteger(options.output_height)
    && options.output_width >= minimumOutputSize && options.output_width <= 4096
    && options.output_height >= minimumOutputSize && options.output_height <= 4096;
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
      { path, frame_index: 0 },
    )
  ), []);

  const saveLabelImage = useCallback(async (
    imagePath: string,
    annotations: Annotation[],
    status: "ready" | "reviewed",
    fingerprint: string,
  ) => {
    const path = projectPath;
    if (!path) throw new Error("Hãy lưu dự án trước khi chỉnh sửa nhãn.");
    setProject((current) => ({
      ...current,
      images: current.images.map((image) => image.path === imagePath
        ? { ...image, annotations: annotations.map((annotation) => ({ ...annotation })), status }
        : image),
    }));
    try {
      await serializeProjectWrite(() => callEngine("save_annotations", {
        path,
        image_path: imagePath,
        annotations,
        status,
        image_fingerprint: fingerprint,
      }));
      annotationSaveFailed.current = false;
    } catch (reason) {
      annotationSaveFailed.current = true;
      setDirty(true);
      throw reason;
    }
  }, [projectPath]);

  useEffect(() => {
    let alive = true;
    if (!isTauri()) {
      setEngineStatus("error");
      return () => { alive = false; };
    }
    void callEngine<{ status: string }>("health", {})
      .then(() => alive && setEngineStatus("ready"))
      .catch(() => alive && setEngineStatus("error"));
    return () => { alive = false; };
  }, []);

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

  useEffect(() => {
    if (!selectedSource) {
      setPreview(null);
      setPreviewError(null);
      return;
    }
    const sequence = ++previewSequence.current;
    const path = selectedSource.path;
    const index = selectedSource.kind === "image" ? 0 : frameIndex;
    setPreviewError(null);
    const timer = window.setTimeout(() => {
      void callEngine<FramePreview>("preview_frame", { path, frame_index: index })
        .then((result) => {
          if (sequence === previewSequence.current) setPreview(result);
        })
        .catch((reason) => {
          if (sequence === previewSequence.current) {
            setPreview(null);
            setPreviewError(String(reason));
          }
        });
    }, 90);
    return () => window.clearTimeout(timer);
  }, [selectedSource?.path, selectedSource?.kind, frameIndex]);

  useEffect(() => {
    if (!playing || !selectedSource || selectedSource.kind !== "video") return;
    if (sourceFrameCount && frameIndex >= sourceFrameCount - 1) {
      setPlaying(false);
      return;
    }
    const interval = Math.max(120, 1000 / (selectedSource.fps || 24));
    const timer = window.setTimeout(() => setFrameIndex((index) => index + 1), interval);
    return () => window.clearTimeout(timer);
  }, [playing, selectedSource?.path, selectedSource?.fps, selectedSource?.kind, sourceFrameCount, frameIndex]);

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
      const result = await callEngine<{
        sources: MediaSource[];
        unsupported_count: number;
        errors: { path: string; message: string }[];
      }>("inspect_sources", { paths });
      if (!result.sources.length) {
        setError(result.errors[0]?.message || "Không tìm thấy video hoặc ảnh được hỗ trợ trong lựa chọn.");
        return;
      }
      const existing = new Set(project.sources.map((source) => source.path));
      const added = result.sources.filter((source) => !existing.has(source.path));
      setProject((current) => ({ ...current, sources: [...current.sources, ...added] }));
      setSelectedPaths((selected) => [...new Set([...selected, ...result.sources.map((source) => source.path)])]);
      if (!selectedPath && added[0]) setSelectedPath(added[0].path);
      setDirty(true);
      if (result.unsupported_count) {
        setError(`${result.unsupported_count} tệp không thuộc định dạng media được hỗ trợ đã bị bỏ qua.`);
      } else if (result.errors.length) {
        setError(`${result.sources.length} nguồn đã thêm; ${result.errors.length} tệp không đọc được.`);
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
        sources: [],
        settings: DEFAULT_OPTIONS,
        images: [],
        class_names: ["head", "body"],
      };
      setProject(empty);
      setProjectPath(path);
      setOptions(DEFAULT_OPTIONS);
      setSelectedPaths([]);
      setSelectedPath(null);
      setGallery([]);
      setFileListTruncated(false);
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
      const loaded = await callEngine<ProjectState>("load_project", { path });
      const nextOptions = { ...DEFAULT_OPTIONS, ...(loaded.settings as Partial<ExtractionOptions>) };
      setProject({ ...loaded, settings: nextOptions });
      setOptions(nextOptions);
      setProjectPath(path);
      setSelectedPaths(loaded.sources.map((source) => source.path));
      setSelectedPath(loaded.sources[0]?.path ?? null);
      setFrameIndex(0);
      setGallery([]);
      setFileListTruncated(false);
      setDirty(false);
      setError(null);
    } catch (reason) {
      setError(String(reason));
    }
  }

  async function saveProject(): Promise<boolean> {
    if (annotationSaveFailed.current) {
      setError("Một số nhãn chưa được tự lưu. Quay lại Gán nhãn, tải lại ảnh và thử lưu nhãn trước.");
      return false;
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
      if (!path) return false;
      const saved = await serializeProjectWrite(() => callEngine<{ id: string; name: string; modified_at: string }>("save_project", {
        path,
        project: { ...project, settings: options },
      }));
      setProject((current) => ({ ...current, id: saved.id, name: saved.name }));
      setProjectPath(path);
      setDirty(false);
      setError(null);
      return true;
    } catch (reason) {
      setError(String(reason));
      return false;
    }
  }

  async function openLabelWorkspace() {
    setError(null);
    if (project.images.length > 0 && (!projectPath || dirty)) {
      if (!(await saveProject())) return;
    }
    setActiveTab("label");
  }

  async function inspectModel() {
    setError(null);
    try {
      const path = await open({
        title: "Chọn model ONNX để kiểm tra",
        multiple: false,
        filters: [{ name: "Model ONNX", extensions: ["onnx"] }],
      });
      if (typeof path !== "string") return;
      setInspectingModel(true);
      const info = await callEngine<ModelInfo>("inspect_model", { path });
      setModelInfo(info);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setInspectingModel(false);
    }
  }

  async function chooseOutputDirectory() {
    try {
      const chosen = await open({ title: "Chọn thư mục lưu frame", directory: true, multiple: false });
      if (typeof chosen === "string") setOutputDirectory(chosen);
    } catch (reason) {
      setError(String(reason));
    }
  }

  async function startExtraction() {
    if (!selectedSources.length) {
      setError("Hãy chọn ít nhất một nguồn ở danh sách bên trái.");
      return;
    }
    if (!options.interval || options.interval <= 0) {
      setError("Khoảng lấy mẫu phải lớn hơn 0.");
      return;
    }
    const minimumOutputSize = options.crop_mode === "center" ? 1 : 16;
    if (!Number.isInteger(options.output_width) || !Number.isInteger(options.output_height)
      || options.output_width < minimumOutputSize || options.output_width > 4096
      || options.output_height < minimumOutputSize || options.output_height > 4096) {
      setError(`Chiều rộng và chiều cao phải là số nguyên từ ${minimumOutputSize} đến 4096 px.`);
      return;
    }
    let destination = outputDirectory;
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
    setFileListTruncated(false);
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
            options,
          },
        },
      });
      setProgress(result);
      setFileListTruncated(result.file_list_truncated);
      setGallery((current) => [...result.files, ...current].slice(0, 500));
      setProject((current) => {
        const imagesByPath = new Map<string, ProjectImage>(current.images.map((image) => [image.path, image]));
        for (const file of result.files) {
          const { thumbnail_data_url: _thumbnail, name: _name, ...image } = file;
          const existing = imagesByPath.get(file.path);
          imagesByPath.set(file.path, {
            ...image,
            status: existing?.status ?? "ready",
            annotations: existing?.annotations ?? [],
          });
        }
        return { ...current, images: [...imagesByPath.values()] };
      });
      setDirty(true);
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

  function selectGalleryImage(file: ExtractionFile) {
    if (file.source_path) {
      setSelectedPath(file.source_path);
      setFrameIndex(file.frame_index ?? 0);
    }
  }

  const extractionPercent = progress?.source_total
    ? Math.min(100, Math.round((progress.source_index / progress.source_total) * 100))
    : 0;

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
          <button className="tool-button" type="button" onClick={() => void inspectModel()} disabled={inspectingModel} title="Nạp và kiểm tra model ONNX trên CPU">
            {inspectingModel ? <LoaderCircle className="spin" size={15} /> : <Sparkles size={15} />}
            <span>{inspectingModel ? "Đang kiểm tra" : "Kiểm tra model"}</span>
          </button>
          <button className="tool-button tool-button-disabled" type="button" disabled title="Suy luận chưa được triển khai">
            <Sparkles size={15} /><span>Auto label</span>
          </button>
          <button className="tool-button tool-button-disabled" type="button" disabled title="Xuất YOLO sẽ có ở giai đoạn sau">
            <Scissors size={15} /><span>Xuất YOLO</span>
          </button>
        </div>
        <div className={`engine-badge engine-${engineStatus}`} title={engineStatus === "error" ? "Không kết nối được Python engine" : "Trạng thái Python engine"}>
          <span className="engine-dot" />
          <span>{engineStatus === "ready" ? "Engine sẵn sàng" : engineStatus === "checking" ? "Đang kết nối" : "Engine lỗi"}</span>
        </div>
      </header>

      {modelInfo ? (
        <div className="model-strip">
          <Check size={14} />
          <span className="model-summary" title={`Input: ${modelInfo.inputs.map((tensor) => `${tensor.name} [${tensor.shape.join(" × ")}] · ${tensor.type}`).join("; ")} | Output: ${modelInfo.outputs.map((tensor) => `${tensor.name} [${tensor.shape.join(" × ")}] · ${tensor.type}`).join("; ")}`}>
            <strong>{modelInfo.name}</strong> · ONNX · In {modelInfo.inputs.map((tensor) => `[${tensor.shape.join("×")}]`).join(", ") || "—"} · Out {modelInfo.outputs.map((tensor) => `[${tensor.shape.join("×")}]`).join(", ") || "—"} · {modelInfo.providers.join(", ")}
          </span>
          {modelInfo.class_names.length > 0 && <span className="model-classes">{modelInfo.class_names.length} lớp</span>}
          <button type="button" className="icon-button strip-close" onClick={() => setModelInfo(null)} aria-label="Đóng thông tin model"><X size={14} /></button>
        </div>
      ) : null}

      <nav className="workflow-tabs" aria-label="Các giai đoạn">
        <button className={activeTab === "extract" ? "workflow-tab active" : "workflow-tab"} onClick={() => setActiveTab("extract")} type="button">
          <Scissors size={15} /><span>Trích xuất</span>
        </button>
        <button className={activeTab === "label" ? "workflow-tab active" : "workflow-tab"} type="button" onClick={() => void openLabelWorkspace()}>
          <Settings2 size={15} /><span>Gán nhãn</span>
        </button>
        <button className="workflow-tab unavailable" type="button" disabled title="Cần hoàn tất trình gán nhãn">
          <HardDrive size={15} /><span>Xuất dữ liệu</span><small>Sắp có</small>
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
          images={project.images}
          classNames={project.class_names}
          loadPreview={loadLabelPreview}
          saveImage={saveLabelImage}
        />
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
                      <small>{source.width} × {source.height} · {source.kind === "video" ? formatTime(source.duration_seconds) : "Ảnh"} · {formatBytes(source.size_bytes)}</small>
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
              <span className="preview-resolution">
                {preview ? `${preview.width} × ${preview.height}` : selectedSource ? `${selectedSource.width} × ${selectedSource.height}` : "Chưa có nguồn"}
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
              </div>
            ) : previewError ? (
              <div className="preview-empty preview-error"><AlertCircle size={26} /><p>Không tải được frame</p><small>{previewError}</small></div>
            ) : (
              <div className="preview-empty">
                {selectedSource ? <LoaderCircle className="spin" size={25} /> : <FilmGlyph />}
                <p>{selectedSource ? "Đang đọc frame nguồn…" : "Chọn hoặc thêm video / ảnh"}</p>
                <small>{selectedSource ? "Frame được đọc trực tiếp từ tệp gốc." : "Preview hiển thị nội dung media thật, không có dữ liệu minh họa."}</small>
                {!selectedSource && <button type="button" className="primary-button" onClick={() => void chooseMedia(false)}><Plus size={15} /> Thêm media</button>}
              </div>
            )}
          </div>
          <div className="player-controls">
            <div className="transport-controls">
              <button type="button" className="icon-button" disabled={!selectedSource || selectedSource.kind === "image"} onClick={() => updateFrame(0)} aria-label="Frame đầu" title="Frame đầu"><ChevronsLeft size={16} /></button>
              <button type="button" className="icon-button" disabled={!selectedSource || selectedSource.kind === "image"} onClick={() => updateFrame(frameIndex - 1)} aria-label="Frame trước" title="Frame trước"><ChevronLeft size={16} /></button>
              <button type="button" className="play-button" disabled={!selectedSource || selectedSource.kind === "image"} onClick={() => setPlaying((current) => !current)} aria-label={playing ? "Tạm dừng" : "Phát"} title={playing ? "Tạm dừng" : "Phát"}>
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
              <SettingToggle label="Lọc frame gần giống" checked={options.filter_duplicates} onChange={(checked) => updateOptions({ filter_duplicates: checked })} />
              {options.filter_duplicates && <ThresholdControl label="Ngưỡng dHash" value={options.duplicate_threshold} min={0} max={24} suffix="bit" onChange={(value) => updateOptions({ duplicate_threshold: value })} />}
              <SettingToggle label="Lọc ảnh mờ" checked={options.filter_blurry} onChange={(checked) => updateOptions({ filter_blurry: checked })} />
              {options.filter_blurry && <ThresholdControl label="Độ nét Laplacian" value={options.blur_threshold} min={0} max={250} suffix="" onChange={(value) => updateOptions({ blur_threshold: value })} />}
              <p className="setting-note">Ngưỡng độ nét phụ thuộc độ phân giải và nội dung video.</p>
            </section>

            <section className="settings-section">
              <div className="section-title-row"><h3>Chế độ crop</h3></div>
              <label className={`radio-option ${options.crop_mode === "center" ? "checked" : ""}`}>
                <input type="radio" name="crop-mode" value="center" checked={options.crop_mode === "center"} onChange={() => updateOptions({ crop_mode: "center" })} />
                <span><strong>Exact Center Crop</strong><small>Cắt đúng pixel tại tâm, không resize</small></span>
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
              <div className="unavailable-crops">
                <span title="Cần bộ phát hiện object hoạt động">Object-centered <small>Cần inference</small></span>
                <span title="Công cụ chọn crop chưa được triển khai">Thủ công <small>Sắp có</small></span>
              </div>
            </section>

            <section className="settings-section">
              <div className="section-title-row"><h3>Kích thước</h3><span className="section-caption">{options.crop_mode === "center" ? "crop · px" : "output · px"}</span></div>
              <div className="preset-row" role="group" aria-label="Preset kích thước crop vuông">
                {[192, 256, 320, 416, 512, 640].map((size) => <button key={size} type="button" className={options.output_width === size && options.output_height === size ? "size-preset active" : "size-preset"} aria-pressed={options.output_width === size && options.output_height === size} onClick={() => updateOptions({ output_width: size, output_height: size })}>{size}</button>)}
              </div>
              <div className="form-grid two-col dimension-row">
                <label className="field-label">Rộng<input type="number" min={minimumOutputSize} max="4096" step="1" value={options.output_width} onChange={(event) => updateOptions({ output_width: Number(event.target.value) })} /></label>
                <label className="field-label">Cao<input type="number" min={minimumOutputSize} max="4096" step="1" value={options.output_height} onChange={(event) => updateOptions({ output_height: Number(event.target.value) })} /></label>
              </div>
              <div className="form-grid two-col output-options">
                <label className="field-label">Định dạng<select value={options.image_format} onChange={(event) => updateOptions({ image_format: event.target.value as ExtractionOptions["image_format"] })}><option value="jpg">JPG</option><option value="png">PNG</option></select></label>
                {options.crop_mode !== "center" && <label className="field-label">Viền<input type="color" value={options.padding_color} onChange={(event) => updateOptions({ padding_color: event.target.value })} /></label>}
              </div>
              {options.crop_mode === "center" && !cropSizeValid
                ? <p className="crop-validation" role="alert">Nhập chiều rộng và chiều cao là số nguyên từ 1 đến 4096 px.</p>
                : options.crop_mode === "center" && preview && !cropFitsPreview
                  ? <p className="crop-validation" role="alert">Crop {options.output_width} × {options.output_height} vượt khung nguồn {preview.width} × {preview.height}. Hãy giảm kích thước crop.</p>
                  : options.crop_mode === "center" && <p className="crop-helper">Tâm crop dùng floor((khung − kích thước) / 2). Ảnh giữ nguyên pixel nguồn.</p>}
              {options.image_format === "jpg" && <label className="quality-row">JPEG quality <input type="range" min="50" max="100" value={options.jpeg_quality} onChange={(event) => updateOptions({ jpeg_quality: Number(event.target.value) })} /><strong>{options.jpeg_quality}</strong></label>}
            </section>

            <section className="settings-section output-section">
              <div className="section-title-row"><h3>Đầu ra</h3><span className="section-caption">Output</span></div>
              <button className="output-path-button" type="button" onClick={() => void chooseOutputDirectory()}>
                <FolderOpen size={14} /><span title={outputDirectory || "Chọn thư mục lưu frame"}>{outputDirectory || "Chọn thư mục lưu frame"}</span><ChevronDown size={14} />
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
              <button type="button" className="extract-button" onClick={() => void startExtraction()} disabled={engineStatus !== "ready" || !selectedSources.length}>
                <Scissors size={16} /> Trích xuất frame <span>{selectedSources.length || "—"}</span>
              </button>
            )}
            {extracting && <div className="extract-progress"><div className="progress-track"><span style={{ width: `${extractionPercent}%` }} /></div><small>{progress?.source_name || "Đang chuẩn bị"} · {progress?.candidate_frames ?? 0} frame</small></div>}
          </div>
        </aside>

        <div className="split-handle split-handle-horizontal" {...splitterProps("gallery", "Điều chỉnh chiều cao gallery")} />

        <section className="gallery-panel panel" aria-label="Frame vừa trích" ref={galleryPanelRef}>
          <div className="gallery-heading">
            <div><h2>Frame vừa trích</h2><span>{project.images.length} trong dự án{fileListTruncated ? " · đang hiển thị tối đa 500 ảnh gần nhất" : ""}</span></div>
            <div className="gallery-summary">
              {progress && <span><Check size={13} /> {progress.exported_images} ảnh xuất</span>}
              {progress && progress.duplicate_crops > 0 && <span>{progress.duplicate_crops} crop trùng</span>}
              {progress && progress.blurry_crops > 0 && <span>{progress.blurry_crops} crop mờ</span>}
              {progress && <span>{progress.elapsed_seconds}s</span>}
            </div>
          </div>
          <div className="gallery-strip">
            {gallery.map((file, index) => (
              <button type="button" className="gallery-item" key={`${file.path}-${index}`} onClick={() => selectGalleryImage(file)} title={`${file.name} · ${file.width} × ${file.height}`}>
                {file.thumbnail_data_url ? <img src={file.thumbnail_data_url} alt="" loading="lazy" /> : <div className="gallery-placeholder"><FileImage size={20} /></div>}
                <span className="gallery-file-name">{file.name}</span>
                <small>{file.width} × {file.height}</small>
              </button>
            ))}
            {gallery.length === 0 && project.images.length === 0 && (
              <div className="gallery-empty"><ImagePlus size={18} /><span>Frame xuất hiện tại đây sau lần trích đầu tiên.</span></div>
            )}
            {gallery.length === 0 && project.images.length > 0 && (
              <div className="gallery-empty"><FileImage size={18} /><span>{project.images.length} ảnh đã lưu trong dự án. Trích thêm để tạo preview thumbnail.</span></div>
            )}
          </div>
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
          <span>{outputDirectory || "Đầu ra chưa chọn"}</span>
          <span className="status-separator">/</span>
          <span>FrameForge · {activeTab === "label" ? "Gán nhãn" : "Trích xuất"}</span>
        </div>
      </footer>
    </main>
  );
}

function SettingToggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="setting-toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
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
