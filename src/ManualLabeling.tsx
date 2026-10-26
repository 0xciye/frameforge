import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { confirm, open } from "@tauri-apps/plugin-dialog";
import {
  ArrowLeft,
  ArrowRight,
  Ban,
  Check,
  CheckCheck,
  CheckCircle2,
  FileImage,
  Hand,
  LoaderCircle,
  Maximize2,
  MousePointer2,
  RefreshCw,
  Redo2,
  Sparkles,
  Trash2,
  Upload,
  Undo2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import "./ManualLabeling.css";

export type Annotation = {
  id: string;
  class_id: number;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  confidence: number | null;
  source: "manual" | "ai";
  review_status: "unreviewed" | "needs_review" | "reviewed" | "rejected";
  source_model_id?: string | null;
  checkpoint_sha256?: string | null;
};

type LabelImage = {
  path: string;
  width: number;
  height: number;
  status: string;
  annotations?: Annotation[];
  negative_candidates?: NegativeCandidate[];
  inference_attempts?: InferenceAttempt[];
};

export type InferenceAttempt = {
  source_model_id: string;
  checkpoint_sha256: string;
  image_fingerprint: string;
  skipped_large_objects: number;
  created_at?: string;
};

export type NegativeCandidate = {
  id: string;
  source_model_id: string;
  checkpoint_sha256: string;
  image_fingerprint: string;
  review_status: "needs_review" | "confirmed_negative" | "rejected";
  created_at: string;
  reviewed_at?: string | null;
};

type LabelPreview = {
  data_url: string;
  width: number;
  height: number;
  image_fingerprint?: string;
};

type SaveImage = (path: string, annotations: Annotation[], status: "ready" | "reviewed", fingerprint: string) => Promise<Annotation[]>;
type LoadPreview = (path: string) => Promise<LabelPreview>;
type RegisteredModel = {
  run_id: string;
  source?: string;
  display_name?: string;
  checkpoint_path?: string | null;
  checkpoint_usable?: boolean;
  checkpoint_sha256?: string | null;
  class_names?: string[];
  effective_imgsz?: number;
  requested_imgsz?: number;
};
type Prediction = { class_id: number; confidence: number; bbox_xyxy: number[] };
type InferenceProgress = { stage: string; completed_images?: number; total_images?: number; eligible_images?: number; saved_images?: number };
type InferenceResult = { status: string; results: { predictions: Prediction[]; skipped_large_objects?: number }[] };
type BatchInferenceResult = {
  status: string;
  processed_images: number;
  images_with_detections: number;
  candidate_negative_count: number;
  annotation_count: number;
  skipped_changed: number;
  skipped_large_objects: number;
  skipped_attempted?: number;
};
type SaveSuggestionsResult = { annotations: Annotation[]; negative_candidate?: NegativeCandidate | null };
type SaveSuggestions = (path: string, predictions: Prediction[], checkpoint: string, fingerprint: string, skippedLargeObjects: number, replace: boolean) => Promise<SaveSuggestionsResult>;
type ReviewNegativeCandidate = (candidateId: string, action: "confirmed_negative" | "rejected", fingerprint?: string) => Promise<void>;
type RunBatchInference = (checkpoint: string, requestId: string, confidence: number, replace: boolean) => Promise<BatchInferenceResult>;
type ApplyModelClasses = (classNames: string[]) => Promise<void>;
type Point = { x: number; y: number };
type ResizeHandle = "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | "nw";
type Tool = { kind: "draw"; classId: number } | { kind: "select" } | { kind: "pan" };
type Gesture =
  | { kind: "draw"; pointerId: number; start: Point; current: Point; classId: number; clickBoxId?: string; startScreen?: Point }
  | { kind: "move"; pointerId: number; start: Point; current: Point; box: Annotation; startScreen?: Point; cycleId?: string }
  | { kind: "resize"; pointerId: number; start: Point; current: Point; box: Annotation; handle: ResizeHandle }
  | { kind: "pan"; pointerId: number; start: Point; pan: Point };
type Snapshot = { annotations: Annotation[]; status: "ready" | "reviewed" };
type History = { past: Snapshot[]; future: Snapshot[] };
type SaveState = { status: "saving" | "saved" | "error"; message?: string };

type Props = {
  images: LabelImage[];
  classNames: string[];
  loadPreview: LoadPreview;
  saveImage: SaveImage;
  saveSuggestions: SaveSuggestions;
  reviewNegativeCandidate: ReviewNegativeCandidate;
  runBatchInference: RunBatchInference;
  applyModelClasses: ApplyModelClasses;
  /** Saved project path; the last image opened is remembered per project. */
  projectKey?: string | null;
  /** Asks and deletes an image with its JSON; resolves false when the user cancels. */
  deleteImage?: (path: string, boxCount: number) => Promise<boolean>;
  reviewImages?: (paths: string[]) => Promise<string | null>;
};

const DEFAULT_CONFIDENCE = 0.35;

function requestId() {
  return globalThis.crypto?.randomUUID?.() ?? `label-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function engineRequest<T>(command: string, args: Record<string, unknown> = {}): Promise<T> {
  return invoke<T>("engine_request", { request: { id: requestId(), command, arguments: args } });
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function imageName(path: string) {
  return path.split(/[\\/]/).pop() || path;
}

function classNameFor(classNames: string[], classId: number) {
  const name = classNames[classId] ?? `Lớp ${classId}`;
  return `${name.charAt(0).toLocaleUpperCase("vi-VN")}${name.slice(1)} (${classId})`;
}

function copyAnnotations(annotations: Annotation[]): Annotation[] {
  return annotations.map((annotation) => ({ ...annotation }));
}

function pointFromPointer(event: PointerEvent<SVGSVGElement>, width: number, height: number): Point {
  const bounds = event.currentTarget.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return { x: 0, y: 0 };
  return {
    x: clamp((event.clientX - bounds.left) / bounds.width * width, 0, width),
    y: clamp((event.clientY - bounds.top) / bounds.height * height, 0, height),
  };
}

/** Boxes under a point, smallest first, so a head inside a body box or a box under another stays reachable. */
function boxesAt(annotations: Annotation[], point: Point, tolerance: number): Annotation[] {
  return annotations
    .filter((item) => point.x >= item.x_min - tolerance && point.x <= item.x_max + tolerance
      && point.y >= item.y_min - tolerance && point.y <= item.y_max + tolerance)
    .sort((left, right) => (left.x_max - left.x_min) * (left.y_max - left.y_min) - (right.x_max - right.x_min) * (right.y_max - right.y_min));
}

function moveBox(box: Annotation, start: Point, current: Point, width: number, height: number): Annotation {
  const boxWidth = box.x_max - box.x_min;
  const boxHeight = box.y_max - box.y_min;
  const x_min = clamp(box.x_min + current.x - start.x, 0, width - boxWidth);
  const y_min = clamp(box.y_min + current.y - start.y, 0, height - boxHeight);
  return { ...box, x_min, y_min, x_max: x_min + boxWidth, y_max: y_min + boxHeight };
}

function resizeBox(box: Annotation, handle: ResizeHandle, point: Point, width: number, height: number): Annotation {
  const next = { ...box };
  if (handle.includes("w")) next.x_min = clamp(point.x, 0, box.x_max - 1);
  if (handle.includes("e")) next.x_max = clamp(point.x, box.x_min + 1, width);
  if (handle.includes("n")) next.y_min = clamp(point.y, 0, box.y_max - 1);
  if (handle.includes("s")) next.y_max = clamp(point.y, box.y_min + 1, height);
  return next;
}

const NAVIGATION_KEYS = new Set(["a", "d", "n"]);
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 32;
const ZOOM_STEP = 1.25;
// Image list rows have a fixed height so only the rows in view are rendered (projects reach 100 000+ images).
const IMAGE_ROW_HEIGHT = 49;
const IMAGE_ROW_OVERSCAN = 12;

type ImageFilter = "all" | "todo" | "reviewed" | "pending" | `class:${number}`;

function liveAnnotations(image: LabelImage) {
  return (image.annotations ?? []).filter((annotation) => annotation.review_status !== "rejected");
}

function hasPendingAi(image: LabelImage) {
  return (image.annotations ?? []).some((annotation) => annotation.source === "ai" && annotation.review_status === "needs_review")
    || (image.negative_candidates ?? []).some((candidate) => candidate.review_status === "needs_review");
}

function matchesFilter(image: LabelImage, filter: ImageFilter) {
  if (filter === "all") return true;
  if (filter === "todo") return image.status !== "reviewed";
  if (filter === "reviewed") return image.status === "reviewed";
  if (filter === "pending") return hasPendingAi(image);
  const classId = Number(filter.slice("class:".length));
  return liveAnnotations(image).some((annotation) => annotation.class_id === classId);
}

/** Class keys: 1–9 in class ID order, plus a class name's first letter when no other class starts with it. */
function classShortcuts(classNames: string[]) {
  const keys = new Map<string, number>();
  classNames.slice(0, 9).forEach((_, classId) => keys.set(String(classId + 1), classId));
  const byLetter = new Map<string, number[]>();
  classNames.forEach((name, classId) => {
    const letter = name.trim().charAt(0).toLocaleLowerCase("vi-VN");
    if (/^\p{L}$/u.test(letter)) byLetter.set(letter, [...(byLetter.get(letter) ?? []), classId]);
  });
  for (const [letter, ids] of byLetter) if (ids.length === 1 && !NAVIGATION_KEYS.has(letter)) keys.set(letter, ids[0]);
  return keys;
}

function lastImageKey(projectKey?: string | null) {
  return projectKey ? `frameforge.last-image.v1:${projectKey}` : null;
}

function readLastImage(projectKey?: string | null) {
  const key = lastImageKey(projectKey);
  try { return key ? localStorage.getItem(key) : null; } catch { return null; }
}

export default function ManualLabeling({ projectKey, deleteImage, reviewImages, images, classNames, loadPreview, saveImage, saveSuggestions, reviewNegativeCandidate, runBatchInference, applyModelClasses }: Props) {
  const classes = useMemo(() => classNames.length ? classNames : ["head", "body"], [classNames]);
  const shortcuts = useMemo(() => classShortcuts(classes), [classes]);
  const shortcutLabel = (classId: number) => [...shortcuts].filter(([, id]) => id === classId).map(([key]) => key.toLocaleUpperCase("vi-VN")).join(" / ");
  const [activePath, setActivePath] = useState(() => {
    const saved = readLastImage(projectKey);
    return saved && images.some((item) => item.path === saved) ? saved : images[0]?.path ?? "";
  });
  const [imageFilter, setImageFilter] = useState<ImageFilter>("all");
  const [resuggestLabeled, setResuggestLabeled] = useState(false);
  const [bulkReviewing, setBulkReviewing] = useState(false);
  const [bulkReviewMessage, setBulkReviewMessage] = useState<string | null>(null);
  const imageListRef = useRef<HTMLDivElement | null>(null);
  const [listScroll, setListScroll] = useState({ top: 0, height: 600 });
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>({ kind: "select" });
  const [preview, setPreview] = useState<LabelPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [spaceHeld, setSpaceHeld] = useState(false);
  const view = useRef({ zoom: 1, pan: { x: 0, y: 0 } });
  view.current = { zoom, pan };
  const [gesture, setGesture] = useState<Gesture | null>(null);
  const [saveStates, setSaveStates] = useState<Record<string, SaveState>>({});
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelImportBusy, setModelImportBusy] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [confidencePercent, setConfidencePercent] = useState(Math.round(DEFAULT_CONFIDENCE * 100));
  const [applyingClasses, setApplyingClasses] = useState(false);
  const [inferenceBusy, setInferenceBusy] = useState(false);
  const [inferenceCancelling, setInferenceCancelling] = useState(false);
  const [inferenceProgress, setInferenceProgress] = useState<InferenceProgress | null>(null);
  const [inferenceMessage, setInferenceMessage] = useState<string | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [reviewingCandidateId, setReviewingCandidateId] = useState<string | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const gestureRef = useRef<Gesture | null>(null);
  const previewSequence = useRef(0);
  const activeInferenceId = useRef<string | null>(null);
  const histories = useRef(new Map<string, History>());
  const saveRevisions = useRef(new Map<string, number>());
  const availableModels = useMemo(() => models.filter((model) => (
    model.checkpoint_usable && typeof model.checkpoint_path === "string" && (model.class_names?.length ?? 0) > 0
  )), [models]);
  const usableModels = useMemo(() => availableModels.filter((model) => (
    model.class_names?.length === classes.length
    && model.class_names.every((name, index) => name === classes[index])
  )), [availableModels, classes]);
  const selectedEntry = availableModels.find((model) => model.run_id === selectedModelId) ?? null;
  const selectedModel = usableModels.find((model) => model.run_id === selectedModelId) ?? null;
  const projectHasLabels = images.some((item) => (item.annotations?.length ?? 0) > 0 || (item.negative_candidates?.length ?? 0) > 0);
  const confidenceValid = Number.isFinite(confidencePercent) && confidencePercent >= 1 && confidencePercent <= 99;
  const confidence = confidenceValid ? confidencePercent / 100 : DEFAULT_CONFIDENCE;
  const image = images.find((item) => item.path === activePath) ?? images[0] ?? null;
  const visibleImages = useMemo(() => images.filter((item) => matchesFilter(item, imageFilter)), [images, imageFilter]);
  const reviewedCount = useMemo(() => images.filter((item) => item.status === "reviewed").length, [images]);
  const imagePosition = useMemo(() => new Map(images.map((item, index) => [item.path, index])), [images]);
  const firstRow = Math.max(0, Math.floor(listScroll.top / IMAGE_ROW_HEIGHT) - IMAGE_ROW_OVERSCAN);
  const lastRow = Math.min(visibleImages.length, Math.ceil((listScroll.top + listScroll.height) / IMAGE_ROW_HEIGHT) + IMAGE_ROW_OVERSCAN);
  const annotations = image?.annotations ?? [];
  const activeAnnotations = useMemo(() => annotations.filter((item) => item.review_status !== "rejected"), [annotations]);
  const pendingSuggestions = annotations.filter((item) => item.source === "ai" && item.review_status === "needs_review");
  const negativeCandidates = image?.negative_candidates ?? [];
  const pendingNegativeCandidates = negativeCandidates.filter((item) => item.review_status === "needs_review");
  const inferenceQueue = images.filter((item) => (
    item.status !== "reviewed"
    && !(item.annotations ?? []).some((annotation) => annotation.source === "manual" || annotation.review_status !== "rejected")
    && !(item.negative_candidates ?? []).some((candidate) => candidate.review_status !== "rejected")
    && !(selectedModel && (item.inference_attempts ?? []).some((attempt) => attempt.checkpoint_sha256 === selectedModel.checkpoint_sha256))
  ));
  const batchCount = resuggestLabeled ? images.length : inferenceQueue.length;
  const imageHasSuggestionState = Boolean(image && (activeAnnotations.length > 0
    || negativeCandidates.some((item) => item.review_status !== "rejected") || (image.inference_attempts ?? []).length > 0));
  const imageIndex = image ? images.findIndex((item) => item.path === image.path) : -1;
  const selectedAnnotation = activeAnnotations.find((item) => item.id === selectedAnnotationId) ?? null;
  const saveState = image ? saveStates[image.path] ?? { status: "saved" as const } : { status: "saved" as const };
  const sizeMismatch = Boolean(preview && image && (preview.width !== image.width || preview.height !== image.height));
  const canEdit = Boolean(image && preview && preview.image_fingerprint && !sizeMismatch && !inferenceBusy);
  const fitSize = useMemo(() => {
    if (!preview || !stageSize.width || !stageSize.height) return null;
    const scale = Math.min(stageSize.width / preview.width, stageSize.height / preview.height);
    return { width: preview.width * scale, height: preview.height * scale, scale: scale * zoom };
  }, [preview, stageSize, zoom]);
  const handleSize = fitSize ? 12 / fitSize.scale : 12;
  const strokeWidth = fitSize ? 2 / fitSize.scale : 2;

  const refreshModels = useCallback(async () => {
    setModelsLoading(true);
    setModelError(null);
    try {
      const result = await engineRequest<{ runs: RegisteredModel[] }>("list_models");
      setModels(result.runs ?? []);
    } catch (reason) {
      setModelError(String(reason));
    } finally {
      setModelsLoading(false);
    }
  }, []);

  async function deleteSelectedModel() {
    const model = selectedEntry;
    if (!model || modelBusy) return;
    const approved = await confirm(
      `Xóa model "${model.display_name ?? "YOLO11n"} · ${model.run_id.slice(0, 8)}" khỏi FrameForge?\n\nBox đã gán bằng model này vẫn giữ nguyên. File .pt gốc của bạn không bị xóa.`,
      { title: "Xóa model", kind: "warning", okLabel: "Xóa model", cancelLabel: "Hủy" },
    );
    if (!approved) return;
    setModelError(null);
    try {
      await engineRequest("delete_model", { run_id: model.run_id });
      setSelectedModelId("");
      await refreshModels();
    } catch (reason) {
      setModelError(String(reason));
    }
  }

  async function importModelCheckpoint() {
    setModelError(null);
    setModelImportBusy(true);
    try {
      const selected = await open({
        title: "Chọn checkpoint YOLO11n",
        multiple: false,
        filters: [{ name: "YOLO checkpoint", extensions: ["pt"] }],
      });
      if (typeof selected !== "string") return;
      const trusted = await confirm(
        "Checkpoint .pt có thể chứa dữ liệu thực thi mã khi được nạp. Chỉ tiếp tục nếu bạn tự tạo tệp hoặc tin tưởng nguồn của nó. FrameForge sẽ kiểm tra task, kiến trúc YOLO11n và class, rồi sao chép checkpoint vào thư mục quản lý riêng.",
        { title: "Xác nhận tin cậy checkpoint", kind: "warning", okLabel: "Tin cậy và nạp", cancelLabel: "Hủy" },
      );
      if (!trusted) return;
      setInferenceError(null);
      setInferenceMessage(null);
      const imported = await engineRequest<{
        run_id: string;
        checkpoint_path: string;
        class_names: string[];
        architecture: string;
        display_name: string;
      }>("import_yolo_checkpoint", { path: selected, trust_source: true });
      await refreshModels();
      setSelectedModelId(imported.run_id);
      const matchesProject = imported.class_names.length === classes.length
        && imported.class_names.every((name, index) => name === classes[index]);
      if (matchesProject) {
        setInferenceMessage(`Đã nhập ${imported.architecture} với class ${imported.class_names.join(", ")}. Model sẵn sàng để gợi ý box.`);
      } else if (!projectHasLabels && await confirm(
        `Model dùng class: ${imported.class_names.join(", ")}.\nDự án đang dùng: ${classes.join(", ")}.\n\nDự án chưa có nhãn nào. Đổi class của dự án theo model để dùng model này?`,
        { title: "Dùng class của model", kind: "info", okLabel: "Dùng class của model", cancelLabel: "Giữ class hiện tại" },
      )) {
        await applyModelClasses(imported.class_names);
        setInferenceMessage(`Đã nhập ${imported.architecture}; dự án dùng class ${imported.class_names.join(", ")}. Model sẵn sàng để gợi ý box.`);
      } else {
        setInferenceMessage(`Đã nhập ${imported.architecture} (${imported.class_names.join(", ")}), nhưng class chưa khớp dự án nên chưa thể dùng inference.`);
      }
    } catch (reason) {
      setModelError(String(reason));
    } finally {
      setModelImportBusy(false);
    }
  }

  useEffect(() => { void refreshModels(); }, [refreshModels]);

  useEffect(() => {
    if (availableModels.some((model) => model.run_id === selectedModelId)) return;
    setSelectedModelId(usableModels[0]?.run_id ?? availableModels[0]?.run_id ?? "");
  }, [availableModels, usableModels, selectedModelId]);

  async function adoptModelClasses(model: RegisteredModel) {
    const names = model.class_names ?? [];
    if (!names.length || projectHasLabels || applyingClasses) return;
    setApplyingClasses(true);
    setModelError(null);
    try {
      await applyModelClasses(names);
      setSelectedModelId(model.run_id);
      setInferenceMessage(`Dự án dùng class của model: ${names.join(", ")}. Model sẵn sàng để gợi ý box.`);
    } catch (reason) {
      setModelError(String(reason));
    } finally {
      setApplyingClasses(false);
    }
  }

  useEffect(() => {
    if (images.some((item) => item.path === activePath)) return;
    // Images arrived later or the project changed: go back to the last image worked on, else the first.
    const saved = readLastImage(projectKey);
    setActivePath(saved && images.some((item) => item.path === saved) ? saved : images[0]?.path ?? "");
  }, [activePath, images, projectKey]);

  useEffect(() => {
    const key = lastImageKey(projectKey);
    if (!key || !activePath) return;
    try { localStorage.setItem(key, activePath); } catch { /* per-device convenience only */ }
  }, [activePath, projectKey]);

  useEffect(() => {
    const node = stageRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      setStageSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!image) {
      setPreview(null);
      setPreviewLoading(false);
      return;
    }
    const sequence = ++previewSequence.current;
    setPreview(null);
    setPreviewError(null);
    setPreviewLoading(true);
    loadPreview(image.path)
      .then((value) => {
        if (previewSequence.current === sequence) setPreview(value);
      })
      .catch((reason) => {
        if (previewSequence.current === sequence) setPreviewError(String(reason));
      })
      .finally(() => {
        if (previewSequence.current === sequence) setPreviewLoading(false);
      });
    return () => {
      if (previewSequence.current === sequence) previewSequence.current += 1;
    };
  }, [image?.path, loadPreview]);

  useEffect(() => () => {
    const id = activeInferenceId.current;
    if (id) {
      activeInferenceId.current = null;
      void invoke<boolean>("cancel_engine_request", { requestId: id }).catch(() => undefined);
    }
  }, []);

  function setCurrentGesture(next: Gesture | null) {
    gestureRef.current = next;
    setGesture(next);
  }

  function persist(path: string, nextAnnotations: Annotation[], status: "ready" | "reviewed", fingerprint: string | undefined) {
    if (!fingerprint) {
      setSaveStates((current) => ({ ...current, [path]: { status: "error", message: "Không có fingerprint của ảnh preview nên chưa thể lưu nhãn." } }));
      return;
    }
    const revision = (saveRevisions.current.get(path) ?? 0) + 1;
    saveRevisions.current.set(path, revision);
    setSaveStates((current) => ({ ...current, [path]: { status: "saving" } }));
    void saveImage(path, nextAnnotations, status, fingerprint).then(() => {
      if (saveRevisions.current.get(path) !== revision) return;
      setSaveStates((current) => ({ ...current, [path]: { status: "saved" } }));
    }).catch((reason) => {
      if (saveRevisions.current.get(path) !== revision) return;
      setSaveStates((current) => ({ ...current, [path]: { status: "error", message: String(reason) } }));
    });
  }

  async function startInference() {
    if (!image || !preview?.image_fingerprint || !selectedModel?.checkpoint_path || inferenceBusy) return;
    // Suggesting again on an image that already has boxes or a candidate replaces them all.
    const replace = activeAnnotations.length > 0 || negativeCandidates.some((item) => item.review_status !== "rejected")
      || (image.inference_attempts ?? []).length > 0;
    if (replace && activeAnnotations.length > 0 && !(await confirm(
      `Gợi ý lại sẽ xóa ${activeAnnotations.length} box hiện có của ảnh này.`,
      { title: "Gợi ý lại", kind: "warning", okLabel: "Gợi ý lại", cancelLabel: "Hủy" },
    ))) return;
    setInferenceBusy(true);
    setInferenceCancelling(false);
    setInferenceProgress(null);
    setInferenceMessage(null);
    setInferenceError(null);
    const id = requestId();
    activeInferenceId.current = id;
    let dispose: (() => void) | undefined;
    try {
      dispose = await listen<{ id: string; progress: InferenceProgress }>("engine-progress", (event) => {
        if (event.payload.id === id) setInferenceProgress(event.payload.progress);
      });
      const result = await invoke<InferenceResult>("engine_request", {
        request: {
          id,
          command: "predict_crops",
          arguments: { checkpoint: selectedModel.checkpoint_path, images: [image.path], confidence_threshold: confidence },
        },
      });
      if (activeInferenceId.current !== id) return;
      if (result.status === "cancelled") {
        setInferenceMessage("Đã dừng inference; không lưu gợi ý.");
        return;
      }
      const predictions = result.results[0]?.predictions ?? [];
      const skippedLargeObjects = result.results[0]?.skipped_large_objects ?? 0;
      const saved = await saveSuggestions(image.path, predictions, selectedModel.checkpoint_path, preview.image_fingerprint, skippedLargeObjects, replace);
      if (replace) histories.current.delete(image.path);
      if (skippedLargeObjects > 0 && predictions.length === 0) {
        setInferenceMessage(`Bỏ qua ${skippedLargeObjects} box chạm biên crop.`);
        return;
      }
      const resultMessage = saved.negative_candidate ? "Không phát hiện box." : `${saved.annotations.length} gợi ý.`;
      setInferenceMessage(skippedLargeObjects > 0 ? `${resultMessage} Bỏ qua ${skippedLargeObjects} box chạm biên crop.` : resultMessage);
    } catch (reason) {
      setInferenceError(String(reason));
    } finally {
      dispose?.();
      activeInferenceId.current = null;
      setInferenceBusy(false);
      setInferenceCancelling(false);
    }
  }

  async function startBatchInference() {
    if (!selectedModel?.checkpoint_path || inferenceBusy || batchCount === 0) return;
    if (resuggestLabeled && !(await confirm(
      `Gợi ý lại ${images.length} ảnh sẽ xóa mọi box hiện có (kể cả box đã duyệt hoặc tự vẽ) của các ảnh đó.`,
      { title: "Gợi ý lại tất cả", kind: "warning", okLabel: "Gợi ý lại", cancelLabel: "Hủy" },
    ))) return;
    if (resuggestLabeled) histories.current.clear();
    setInferenceBusy(true);
    setInferenceCancelling(false);
    setInferenceProgress(null);
    setInferenceMessage(null);
    setInferenceError(null);
    const id = requestId();
    activeInferenceId.current = id;
    let dispose: (() => void) | undefined;
    try {
      dispose = await listen<{ id: string; progress: InferenceProgress }>("engine-progress", (event) => {
        if (event.payload.id === id) setInferenceProgress(event.payload.progress);
      });
      const result = await runBatchInference(selectedModel.checkpoint_path, id, confidence, resuggestLabeled);
      if (activeInferenceId.current !== id) return;
      setInferenceMessage(`${result.status === "cancelled" ? "Đã dừng" : "Xong"}: ${result.processed_images} ảnh · ${result.images_with_detections} có box · ${result.candidate_negative_count} không phát hiện${result.skipped_changed ? ` · ${result.skipped_changed} ảnh lỗi` : ""}.`);
    } catch (reason) {
      setInferenceError(String(reason));
    } finally {
      dispose?.();
      activeInferenceId.current = null;
      setInferenceBusy(false);
      setInferenceCancelling(false);
    }
  }

  async function reviewCandidate(candidate: NegativeCandidate, action: "confirmed_negative" | "rejected") {
    if (!preview?.image_fingerprint || reviewingCandidateId) return;
    if (action === "confirmed_negative" && (
      preview.image_fingerprint !== candidate.image_fingerprint
      || activeAnnotations.length > 0
      || pendingNegativeCandidates.length > 1
    )) return;
    setReviewingCandidateId(candidate.id);
    setInferenceError(null);
    try {
      await reviewNegativeCandidate(
        candidate.id,
        action,
        action === "confirmed_negative" ? preview.image_fingerprint : undefined,
      );
      setInferenceMessage(action === "confirmed_negative"
        ? "Đã xác nhận ảnh âm tính. Ảnh trống này có thể được đưa vào dataset."
        : "Đã loại candidate âm tính. Ảnh cần được gán nhãn trước khi xuất dataset.");
    } catch (reason) {
      setInferenceError(String(reason));
    } finally {
      setReviewingCandidateId(null);
    }
  }

  async function cancelInference() {
    const id = activeInferenceId.current;
    if (!id || inferenceCancelling) return;
    setInferenceCancelling(true);
    try {
      await invoke<boolean>("cancel_engine_request", { requestId: id });
    } catch (reason) {
      setInferenceError(String(reason));
      setInferenceCancelling(false);
    }
  }

  /**
   * Saves the image's boxes. Boxes a person draws or edits need no review (user rule): they are stored
   * as reviewed. Only AI suggestions wait for review, so without an explicit ``requestedStatus`` an image
   * is reviewed once it has boxes and no AI suggestion or empty-image candidate is pending.
   */
  function commit(nextAnnotations: Annotation[], requestedStatus?: "ready" | "reviewed") {
    if (!image || !canEdit || !preview) return;
    const history = histories.current.get(image.path) ?? { past: [], future: [] };
    history.past.push({ annotations: copyAnnotations(annotations), status: image.status === "reviewed" ? "reviewed" : "ready" });
    if (history.past.length > 100) history.past.shift();
    history.future = [];
    histories.current.set(image.path, history);
    const previousById = new Map(annotations.map((annotation) => [annotation.id, annotation]));
    const next = nextAnnotations.map((annotation) => {
      const previous = previousById.get(annotation.id);
      if (!previous) {
        return {
          ...annotation,
          source: "manual" as const,
          confidence: null,
          review_status: "reviewed" as const,
          source_model_id: null,
          checkpoint_sha256: null,
        };
      }
      const changed = previous.class_id !== annotation.class_id
        || previous.x_min !== annotation.x_min || previous.y_min !== annotation.y_min
        || previous.x_max !== annotation.x_max || previous.y_max !== annotation.y_max;
      let reviewStatus = annotation.review_status;
      if (previous.source === "manual") reviewStatus = "reviewed";
      else if (changed && previous.review_status === "needs_review") reviewStatus = "reviewed";
      return {
        ...annotation,
        source: previous.source,
        confidence: previous.confidence,
        source_model_id: previous.source_model_id,
        checkpoint_sha256: previous.checkpoint_sha256,
        review_status: reviewStatus,
      };
    });
    const pendingAi = next.some((annotation) => annotation.source === "ai" && annotation.review_status === "needs_review")
      || pendingNegativeCandidates.length > 0;
    const hasBoxes = next.some((annotation) => annotation.review_status !== "rejected");
    const status = pendingAi ? "ready" : requestedStatus ?? (hasBoxes ? "reviewed" : "ready");
    persist(image.path, next, status, preview.image_fingerprint);
  }

  function undo() {
    if (!image || !canEdit || !preview) return;
    const history = histories.current.get(image.path);
    const snapshot = history?.past.pop();
    if (!history || !snapshot) return;
    history.future.push({ annotations: copyAnnotations(annotations), status: image.status === "reviewed" ? "reviewed" : "ready" });
    histories.current.set(image.path, history);
    persist(image.path, snapshot.annotations, snapshot.status, preview.image_fingerprint);
  }

  function redo() {
    if (!image || !canEdit || !preview) return;
    const history = histories.current.get(image.path);
    const snapshot = history?.future.pop();
    if (!history || !snapshot) return;
    history.past.push({ annotations: copyAnnotations(annotations), status: image.status === "reviewed" ? "reviewed" : "ready" });
    histories.current.set(image.path, history);
    persist(image.path, snapshot.annotations, snapshot.status, preview.image_fingerprint);
  }

  function deleteSelected() {
    if (!selectedAnnotation) return;
    commit(selectedAnnotation.source === "ai"
      ? annotations.map((annotation) => annotation.id === selectedAnnotation.id ? { ...annotation, review_status: "rejected" } : annotation)
      : annotations.filter((annotation) => annotation.id !== selectedAnnotation.id));
    setSelectedAnnotationId(null);
  }

  useEffect(() => {
    const list = imageListRef.current;
    if (!list) return;
    const update = () => setListScroll({ top: list.scrollTop, height: list.clientHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(list);
    list.addEventListener("scroll", update, { passive: true });
    return () => {
      observer.disconnect();
      list.removeEventListener("scroll", update);
    };
  }, []);

  useEffect(() => {
    // Keep the current image's row in view (it may not be rendered yet).
    const list = imageListRef.current;
    const index = visibleImages.findIndex((item) => item.path === activePath);
    if (!list || index < 0) return;
    const top = index * IMAGE_ROW_HEIGHT;
    if (top < list.scrollTop) list.scrollTop = top;
    else if (top + IMAGE_ROW_HEIGHT > list.scrollTop + list.clientHeight) list.scrollTop = top + IMAGE_ROW_HEIGHT - list.clientHeight;
  }, [activePath, imageFilter, visibleImages]);

  /** Moves through the filtered list; ``onlyTodo`` skips images already reviewed. */
  function stepImage(direction: 1 | -1, onlyTodo = false) {
    const list = visibleImages;
    if (!list.length) return;
    const start = list.findIndex((item) => item.path === image?.path);
    for (let offset = 1; offset <= list.length; offset += 1) {
      const index = start < 0
        ? (direction > 0 ? offset - 1 : list.length - offset)
        : (start + direction * offset + list.length) % list.length;
      const candidate = list[index];
      if (!onlyTodo || candidate.status !== "reviewed") {
        if (candidate.path !== image?.path) selectImage(candidate.path);
        return;
      }
    }
  }

  async function reviewVisibleImages() {
    if (!reviewImages || bulkReviewing || inferenceBusy || visibleImages.length === 0) return;
    setBulkReviewing(true);
    setBulkReviewMessage(null);
    try {
      const summary = await reviewImages(visibleImages.map((item) => item.path));
      if (summary) setBulkReviewMessage(summary);
    } catch (reason) {
      setBulkReviewMessage(String(reason));
    } finally {
      setBulkReviewing(false);
    }
  }

  async function deleteCurrentImage() {
    if (!image || !deleteImage || inferenceBusy) return;
    // Continue with the next image in the current list (or the previous one at the end).
    const list = visibleImages.length ? visibleImages : images;
    const index = list.findIndex((item) => item.path === image.path);
    const next = list[index + 1] ?? list[index - 1] ?? null;
    try {
      if (await deleteImage(image.path, activeAnnotations.length) && next) selectImage(next.path);
    } catch (reason) {
      setPreviewError(String(reason));
    }
  }

  function selectImage(path: string) {
    if (inferenceBusy) return;
    setActivePath(path);
    setSelectedAnnotationId(null);
    fitView();
    setPreviewError(null);
  }

  function refreshPreview() {
    if (!image) return;
    const sequence = ++previewSequence.current;
    const path = image.path;
    setPreview(null);
    setPreviewError(null);
    setPreviewLoading(true);
    void loadPreview(path)
      .then((value) => {
        if (previewSequence.current === sequence) setPreview(value);
      })
      .catch((reason) => {
        if (previewSequence.current === sequence) setPreviewError(String(reason));
      })
      .finally(() => {
        if (previewSequence.current === sequence) setPreviewLoading(false);
      });
  }

  function zoomAt(nextZoom: number, clientX?: number, clientY?: number) {
    const target = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom));
    const { zoom: current, pan: currentPan } = view.current;
    if (target === current) return;
    const bounds = stageRef.current?.getBoundingClientRect();
    const mx = bounds && clientX !== undefined ? clientX - (bounds.left + bounds.width / 2) : 0;
    const my = bounds && clientY !== undefined ? clientY - (bounds.top + bounds.height / 2) : 0;
    const ratio = target / current;
    const next = { x: mx - (mx - currentPan.x) * ratio, y: my - (my - currentPan.y) * ratio };
    view.current = { zoom: target, pan: next };
    setZoom(target);
    setPan(next);
  }

  function fitView() {
    view.current = { zoom: 1, pan: { x: 0, y: 0 } };
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const steps = Math.max(-4, Math.min(4, -event.deltaY / (event.deltaMode === 1 ? 3 : 100)));
      zoomAt(view.current.zoom * Math.pow(ZOOM_STEP, steps), event.clientX, event.clientY);
    };
    stage.addEventListener("wheel", onWheel, { passive: false });
    return () => stage.removeEventListener("wheel", onWheel);
  });

  useEffect(() => {
    // Hold Space to drag the image with the left button.
    const typing = (event: globalThis.KeyboardEvent) => (event.target as HTMLElement | null)?.closest?.("input, textarea, select, [contenteditable='true']");
    const down = (event: globalThis.KeyboardEvent) => {
      if (event.code === "Space" && !typing(event)) {
        event.preventDefault();
        setSpaceHeld(true);
      }
    };
    const up = (event: globalThis.KeyboardEvent) => { if (event.code === "Space") setSpaceHeld(false); };
    const blur = () => setSpaceHeld(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);

  function retrySave() {
    if (!image || !preview) return;
    persist(image.path, annotations, image.status === "reviewed" ? "reviewed" : "ready", preview.image_fingerprint);
  }

  function onPointerDown(event: PointerEvent<SVGSVGElement>) {
    if (!image || !preview) return;
    if (event.button === 2 || event.button === 1 || (event.button === 0 && spaceHeld)) {
      // Right or middle button, or Space + left drag, pans at any zoom without touching boxes.
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      setCurrentGesture({ kind: "pan", pointerId: event.pointerId, start: { x: event.clientX, y: event.clientY }, pan });
      return;
    }
    if (!canEdit || event.button !== 0) return;
    const point = pointFromPointer(event, preview.width, preview.height);
    const target = event.target as Element;
    const handleElement = target.closest("[data-handle]") as SVGElement | null;
    const handleBoxId = (target.closest("[data-annotation-id]") as SVGElement | null)?.dataset.annotationId;
    const handleBox = handleElement ? annotations.find((annotation) => annotation.id === handleBoxId) : undefined;
    // Hit-test geometrically instead of taking the SVG element on top: overlapping boxes are all candidates,
    // the smallest wins, and clicking again on the selected one cycles to the next box under the pointer.
    const bounds = event.currentTarget.getBoundingClientRect();
    const tolerance = 2 * preview.width / Math.max(1, bounds.width);
    const candidates = boxesAt(activeAnnotations, point, tolerance);
    const selectedHit = candidates.find((annotation) => annotation.id === selectedAnnotationId);
    const box = selectedHit ?? candidates[0];
    const cycleId = selectedHit && candidates.length > 1
      ? candidates[(candidates.indexOf(selectedHit) + 1) % candidates.length].id
      : undefined;
    const startScreen = { x: event.clientX, y: event.clientY };
    event.currentTarget.focus();
    event.currentTarget.setPointerCapture(event.pointerId);

    if (handleBox && handleElement?.dataset.handle) {
      setSelectedAnnotationId(handleBox.id);
      setCurrentGesture({ kind: "resize", pointerId: event.pointerId, start: point, current: point, box: handleBox, handle: handleElement.dataset.handle as ResizeHandle });
    } else if (selectedHit) {
      // Dragging inside the selected box moves the whole box, in any tool; a plain click selects the next
      // box under the pointer.
      setCurrentGesture({ kind: "move", pointerId: event.pointerId, start: point, current: point, box: selectedHit, startScreen, cycleId });
    } else if (tool.kind === "draw") {
      // A drag draws a new box, even inside another box (a head inside a body box); a plain click on a
      // box selects it so its handles can resize it right away.
      setSelectedAnnotationId(null);
      setCurrentGesture({
        kind: "draw", pointerId: event.pointerId, start: point, current: point, classId: tool.classId,
        clickBoxId: box?.id, startScreen,
      });
    } else if (tool.kind === "pan") {
      setCurrentGesture({ kind: "pan", pointerId: event.pointerId, start: { x: event.clientX, y: event.clientY }, pan });
    } else if (box) {
      setSelectedAnnotationId(box.id);
      setCurrentGesture({ kind: "move", pointerId: event.pointerId, start: point, current: point, box, startScreen });
    } else {
      setSelectedAnnotationId(null);
      setCurrentGesture(null);
    }
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const current = gestureRef.current;
    if (!current || current.pointerId !== event.pointerId) return;
    if (current.kind === "pan") {
      const next = {
        x: current.pan.x + event.clientX - current.start.x,
        y: current.pan.y + event.clientY - current.start.y,
      };
      view.current = { zoom: view.current.zoom, pan: next };
      setPan(next);
      return;
    }
    const point = pointFromPointer(event, preview?.width ?? 0, preview?.height ?? 0);
    setCurrentGesture({ ...current, current: point });
  }

  function onPointerUp(event: PointerEvent<SVGSVGElement>) {
    const current = gestureRef.current;
    if (!current || current.pointerId !== event.pointerId || !image || !preview) return;
    const clicked = (current.kind === "draw" || current.kind === "move") && current.startScreen
      && Math.hypot(event.clientX - current.startScreen.x, event.clientY - current.startScreen.y) < 4;
    if (current.kind === "move" && clicked) {
      if (current.cycleId) setSelectedAnnotationId(current.cycleId);
    } else if (current.kind === "draw" && clicked && current.clickBoxId) {
      setSelectedAnnotationId(current.clickBoxId);
    } else if (current.kind === "draw" && !clicked) {
      const point = pointFromPointer(event, preview.width, preview.height);
      const box: Annotation = {
        id: globalThis.crypto?.randomUUID?.() ?? `box-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        class_id: current.classId,
        x_min: Math.min(current.start.x, point.x),
        y_min: Math.min(current.start.y, point.y),
        x_max: Math.max(current.start.x, point.x),
        y_max: Math.max(current.start.y, point.y),
        confidence: null,
        source: "manual",
        review_status: "unreviewed",
      };
      if (box.x_max - box.x_min >= 1 && box.y_max - box.y_min >= 1) {
        commit([...annotations, box]);
        setSelectedAnnotationId(box.id);
      }
    } else if (current.kind === "move" || current.kind === "resize") {
      const point = pointFromPointer(event, preview.width, preview.height);
      const updated = current.kind === "move"
        ? moveBox(current.box, current.start, point, preview.width, preview.height)
        : resizeBox(current.box, current.handle, point, preview.width, preview.height);
      if (updated.x_min !== current.box.x_min || updated.y_min !== current.box.y_min
        || updated.x_max !== current.box.x_max || updated.y_max !== current.box.y_max) {
        commit(annotations.map((annotation) => annotation.id === updated.id ? updated : annotation));
      }
    }
    setCurrentGesture(null);
  }

  const keyHandler = useRef<(event: globalThis.KeyboardEvent) => void>(() => undefined);
  useEffect(() => {
    const listener = (event: globalThis.KeyboardEvent) => keyHandler.current(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  function onKeyDown(event: globalThis.KeyboardEvent) {
    if (event.defaultPrevented) return;  // Arrow keys already moved the selected box on the canvas.
    const target = (event.target instanceof Element ? event.target : document.body) as HTMLElement;
    if (target.closest("input, textarea, select, [contenteditable='true'], [role='dialog']")) return;
    const command = event.metaKey || event.ctrlKey;
    const key = event.key.toLocaleLowerCase("vi-VN");
    if (!command && !event.altKey && !gesture && (key === "d" || event.key === "PageDown")) {
      event.preventDefault();
      stepImage(1);
      return;
    }
    if (!command && !event.altKey && !gesture && (key === "a" || event.key === "PageUp")) {
      event.preventDefault();
      stepImage(-1);
      return;
    }
    if (!command && !event.altKey && (event.key === "+" || event.key === "=")) {
      event.preventDefault();
      zoomAt(view.current.zoom * ZOOM_STEP);
      return;
    }
    if (!command && !event.altKey && (event.key === "-" || event.key === "_")) {
      event.preventDefault();
      zoomAt(view.current.zoom / ZOOM_STEP);
      return;
    }
    if (!command && !event.altKey && event.key === "0") {
      event.preventDefault();
      fitView();
      return;
    }
    if (!command && !event.altKey && !gesture && key === "n") {
      event.preventDefault();
      stepImage(1, true);
      return;
    }
    if (!command && !event.altKey && !gesture && event.key === "Enter") {
      // Accept this image (and its pending AI suggestions) and move on to the next one that still needs work.
      event.preventDefault();
      if (image && image.status !== "reviewed") {
        if (!canEdit || pendingNegativeCandidates.length > 0) return;
        commit(annotations.map((item) => (
          item.source === "ai" && item.review_status === "needs_review" ? { ...item, review_status: "reviewed" as const } : item
        )), "reviewed");
      }
      stepImage(1, true);
      return;
    }
    if (command && event.key.toLowerCase() === "z") {
      event.preventDefault();
      if (event.shiftKey) redo(); else undo();
    } else if (command && event.key.toLowerCase() === "y") {
      event.preventDefault();
      redo();
    } else if (event.key === "Delete" && event.shiftKey) {
      event.preventDefault();
      void deleteCurrentImage();
    } else if (event.key === "Delete") {
      event.preventDefault();
      deleteSelected();
    } else if (event.key === "Escape") {
      setCurrentGesture(null);
      setTool({ kind: "select" });
    } else if (!command && !event.altKey && shortcuts.has(event.key.toLocaleLowerCase("vi-VN"))) {
      // Class key: re-class the selected box, otherwise draw the next box with that class.
      const classId = shortcuts.get(event.key.toLocaleLowerCase("vi-VN"))!;
      if (!canEdit || event.repeat) return;
      event.preventDefault();
      if (selectedAnnotation) {
        if (selectedAnnotation.class_id !== classId) changeClass(selectedAnnotation, classId);
      } else {
        setCurrentGesture(null);
        setTool({ kind: "draw", classId });
      }
    }
  }

  keyHandler.current = onKeyDown;

  function onCanvasKeyDown(event: KeyboardEvent<SVGSVGElement>) {
    if (!selectedAnnotation || !preview || !canEdit || !event.key.startsWith("Arrow")) return;
    const delta = event.shiftKey ? 10 : 1;
    const dx = event.key === "ArrowLeft" ? -delta : event.key === "ArrowRight" ? delta : 0;
    const dy = event.key === "ArrowUp" ? -delta : event.key === "ArrowDown" ? delta : 0;
    if (!dx && !dy) return;
    event.preventDefault();
    const updated = event.altKey
      ? resizeBox(selectedAnnotation, dx < 0 ? "w" : dx > 0 ? "e" : dy < 0 ? "n" : "s", {
        x: dx < 0 ? selectedAnnotation.x_min + dx : selectedAnnotation.x_max + dx,
        y: dy < 0 ? selectedAnnotation.y_min + dy : selectedAnnotation.y_max + dy,
      }, preview.width, preview.height)
      : moveBox(selectedAnnotation, { x: 0, y: 0 }, { x: dx, y: dy }, preview.width, preview.height);
    commit(annotations.map((annotation) => annotation.id === updated.id ? updated : annotation));
  }

  function markReviewed() {
    if (!image || !canEdit || (image.status !== "reviewed" && (pendingSuggestions.length > 0 || pendingNegativeCandidates.length > 0))) return;
    const status = image.status === "reviewed" ? "ready" : "reviewed";
    commit(annotations, status);
  }

  function changeClass(annotation: Annotation, classId: number) {
    commit(annotations.map((item) => item.id === annotation.id ? { ...item, class_id: classId } : item));
  }

  function reviewSuggestion(annotation: Annotation, review_status: "reviewed" | "rejected") {
    if (annotation.source !== "ai" || annotation.review_status !== "needs_review") return;
    commit(annotations.map((item) => item.id === annotation.id ? { ...item, review_status } : item));
  }

  function reviewAllSuggestions(review_status: "reviewed" | "rejected") {
    if (!pendingSuggestions.length) return;
    commit(annotations.map((item) => (
      item.source === "ai" && item.review_status === "needs_review" ? { ...item, review_status } : item
    )));
  }

  const history = image ? histories.current.get(image.path) : undefined;
  const visibleAnnotations = useMemo(() => {
    if (!gesture || gesture.kind === "draw" || gesture.kind === "pan") return activeAnnotations;
    const updated = gesture.kind === "move"
      ? moveBox(gesture.box, gesture.start, gesture.current, preview?.width ?? image?.width ?? 1, preview?.height ?? image?.height ?? 1)
      : resizeBox(gesture.box, gesture.handle, gesture.current, preview?.width ?? image?.width ?? 1, preview?.height ?? image?.height ?? 1);
    return activeAnnotations.map((annotation) => annotation.id === updated.id ? updated : annotation);
  }, [activeAnnotations, gesture, image?.height, image?.width, preview?.height, preview?.width]);
  const draft = gesture?.kind === "draw" ? {
    x_min: Math.min(gesture.start.x, gesture.current.x),
    y_min: Math.min(gesture.start.y, gesture.current.y),
    x_max: Math.max(gesture.start.x, gesture.current.x),
    y_max: Math.max(gesture.start.y, gesture.current.y),
    class_id: gesture.classId,
  } : null;

  const modelBusy = modelsLoading || inferenceBusy || modelImportBusy || applyingClasses;
  const selectedMismatch = selectedEntry && !selectedModel ? selectedEntry : null;
  const modelControls = (
    <>
      <button type="button" className="subtle-button label-ai-run-button" disabled={modelBusy} onClick={() => void importModelCheckpoint()}>
        <Upload size={13} /> Nhập model .pt
      </button>
      <label className="label-ai-model">
        <span>Checkpoint YOLO11n</span>
        <div className="label-ai-model-row">
          <select aria-label="Checkpoint inference" value={selectedModelId} disabled={modelBusy || availableModels.length === 0} onChange={(event) => setSelectedModelId(event.target.value)}>
            <option value="">{modelsLoading ? "Đang tải checkpoint…" : "Chọn checkpoint"}</option>
            {availableModels.map((model) => <option key={model.run_id} value={model.run_id}>
              {model.display_name ?? "YOLO11n"} · {model.effective_imgsz ?? model.requested_imgsz ?? "—"} px · {model.run_id.slice(0, 8)}{usableModels.includes(model) ? "" : " · class khác dự án"}
            </option>)}
          </select>
          <button type="button" className="label-icon-button" aria-label="Xóa model đang chọn" title="Xóa model đang chọn" disabled={modelBusy || !selectedEntry} onClick={() => void deleteSelectedModel()}><Trash2 size={12} /></button>
        </div>
      </label>
      <label className="label-ai-model">
        <span>Ngưỡng confidence (%)</span>
        <input type="number" min="1" max="99" step="1" value={confidencePercent} disabled={inferenceBusy} aria-invalid={!confidenceValid} onChange={(event) => setConfidencePercent(Number(event.target.value))} />
      </label>
      {!confidenceValid && <small className="label-ai-error" role="alert">Nhập ngưỡng từ 1 đến 99%.</small>}
      {selectedMismatch && <div className="label-ai-class-mismatch">
        <small>Model: {(selectedMismatch.class_names ?? []).join(", ")}<br />Dự án: {classes.join(", ")}</small>
        {projectHasLabels
          ? <small className="label-ai-note">Dự án đã có nhãn nên không thể đổi class. Tạo dự án mới để dùng model này.</small>
          : <button type="button" className="subtle-button label-ai-run-button" disabled={modelBusy} onClick={() => void adoptModelClasses(selectedMismatch)}>Dùng class của model cho dự án</button>}
      </div>}
      {modelImportBusy && <small className="label-ai-note">Đang xác minh checkpoint…</small>}
      {modelError && <small className="label-ai-error" role="alert">Không thể xử lý checkpoint: {modelError}</small>}
    </>
  );

  if (!image) {
    return (
      <section className="label-workspace" aria-label="Không gian gán nhãn">
        <div className="panel label-empty-workspace with-model-panel">
          <FileImage size={28} aria-hidden="true" />
          <h1>Chưa có ảnh để gán nhãn</h1>
        </div>
        <aside className="panel label-inspector-panel">
          <div className="settings-header"><h2>Model</h2><span>{availableModels.length} checkpoint</span></div>
          <div className="label-inspector-scroll">
            <section className="label-tool-section label-ai-section" aria-label="Nhập và chọn model">
              {modelControls}
              {inferenceMessage && <small className="label-ai-note" aria-live="polite">{inferenceMessage}</small>}
            </section>
          </div>
        </aside>
      </section>
    );
  }

  return (
    <section className="label-workspace" aria-label="Không gian gán nhãn">
      <aside className="panel label-images-panel">
        <div className="panel-heading">
          <div><h1>Ảnh trong dự án</h1><span>{images.length} ảnh · {reviewedCount} đã review · {images.length - reviewedCount} chưa xong</span></div>
        </div>
        <div className="label-image-filter">
          <select aria-label="Lọc danh sách ảnh" value={imageFilter} onChange={(event) => setImageFilter(event.target.value as ImageFilter)}>
            <option value="all">Tất cả ảnh</option>
            <option value="todo">Chưa xong (chưa review)</option>
            <option value="reviewed">Đã review</option>
            <option value="pending">Có gợi ý AI chờ duyệt</option>
            {classes.map((name, classId) => <option key={`${classId}-${name}`} value={`class:${classId}`}>Có box {name}</option>)}
          </select>
          <small>{visibleImages.length} ảnh</small>
        </div>
        {reviewImages && <div className="label-bulk-review">
          <button type="button" disabled={bulkReviewing || visibleImages.length === 0} onClick={() => void reviewVisibleImages()} title="Duyệt mọi box AI chờ duyệt và đánh dấu đã review cho các ảnh có box trong danh sách đang lọc">
            {bulkReviewing ? <LoaderCircle className="spin" size={12} /> : <CheckCheck size={12} />}
            {bulkReviewing ? "Đang duyệt…" : `Duyệt toàn bộ ảnh (${visibleImages.length})`}
          </button>
          {bulkReviewMessage && <small role="status">{bulkReviewMessage}</small>}
        </div>}
        <div className="label-image-list" role="list" aria-label="Danh sách ảnh cần gán nhãn" ref={imageListRef}>
          {visibleImages.length === 0 ? <p className="label-image-empty">Không có ảnh nào khớp bộ lọc.</p> : null}
          <div style={{ height: firstRow * IMAGE_ROW_HEIGHT }} aria-hidden="true" />
          {visibleImages.slice(firstRow, lastRow).map((item) => {
            const index = imagePosition.get(item.path) ?? 0;
            const count = item.annotations?.filter((annotation) => annotation.review_status !== "rejected").length ?? 0;
            const pendingCount = (item.annotations?.filter((annotation) => annotation.source === "ai" && annotation.review_status === "needs_review").length ?? 0)
              + (item.negative_candidates?.filter((candidate) => candidate.review_status === "needs_review").length ?? 0);
            const current = item.path === image.path;
            return (
              <div role="listitem" key={item.path} data-path={item.path}>
                <button
                  className={`label-image-row${current ? " selected" : ""}`}
                  type="button"
                  aria-current={current ? "true" : undefined}
                  disabled={inferenceBusy}
                  onClick={() => selectImage(item.path)}
                >
                  <span className={`label-review-dot${pendingCount ? " needs-review" : item.status === "reviewed" ? " reviewed" : ""}`} aria-label={pendingCount ? `${pendingCount} mục AI cần duyệt` : item.status === "reviewed" ? "Đã review" : "Chưa review"} />
                  <span className="label-image-copy"><strong>{imageName(item.path)}</strong><small>{item.width} × {item.height} · {count} box{pendingCount ? ` · ${pendingCount} AI chờ duyệt` : ""}</small></span>
                  <small className="label-image-index">{index + 1}</small>
                </button>
              </div>
            );
          })}
          <div style={{ height: Math.max(0, visibleImages.length - lastRow) * IMAGE_ROW_HEIGHT }} aria-hidden="true" />
        </div>
      </aside>

      <section className="panel label-canvas-panel" aria-label={`Ảnh đang gán nhãn ${imageName(image.path)}`}>
        <div className="label-canvas-toolbar">
          <div className="label-image-navigation">
            <button type="button" className="label-icon-button" aria-label="Ảnh trước" title="Ảnh trước" disabled={inferenceBusy || imageIndex <= 0} onClick={() => selectImage(images[imageIndex - 1].path)}><ArrowLeft size={14} /></button>
            <span>{imageIndex + 1} / {images.length}</span>
            <button type="button" className="label-icon-button" aria-label="Ảnh tiếp theo" title="Ảnh tiếp theo" disabled={inferenceBusy || imageIndex < 0 || imageIndex >= images.length - 1} onClick={() => selectImage(images[imageIndex + 1].path)}><ArrowRight size={14} /></button>
          </div>
          {deleteImage && <button type="button" className="label-icon-button label-delete-image" aria-label="Xóa ảnh" title="Xóa ảnh này và file .json của nó (vào Thùng rác) · Shift+Delete" disabled={inferenceBusy} onClick={() => void deleteCurrentImage()}><Trash2 size={14} /></button>}
          <div className="label-canvas-zoom" role="group" aria-label="Tỷ lệ ảnh">
            <button type="button" className="label-icon-button" aria-label="Thu nhỏ" title="Thu nhỏ (-, lăn chuột)" disabled={zoom <= MIN_ZOOM} onClick={() => zoomAt(zoom / ZOOM_STEP)}><ZoomOut size={14} /></button>
            <span title="Lăn chuột để zoom quanh con trỏ · Chuột phải, chuột giữa hoặc Space + kéo để di chuyển ảnh">{Math.round(zoom * 100)}%</span>
            <button type="button" className="label-icon-button" aria-label="Phóng to" title="Phóng to (+, lăn chuột)" disabled={zoom >= MAX_ZOOM} onClick={() => zoomAt(zoom * ZOOM_STEP)}><ZoomIn size={14} /></button>
            <button type="button" className="label-icon-button" aria-label="Vừa khung" title="Vừa khung (0)" onClick={fitView}><Maximize2 size={14} /></button>
          </div>
        </div>

        <div onContextMenu={(event) => event.preventDefault()} className={`label-canvas-stage${spaceHeld || tool.kind === "pan" || gesture?.kind === "pan" ? " panning" : tool.kind === "draw" ? " drawing" : ""}`} ref={stageRef}>
          {previewLoading && <div className="label-canvas-state"><LoaderCircle className="spin" size={20} /><span>Đang tải ảnh thật…</span></div>}
          {!previewLoading && previewError && <div className="label-canvas-state label-canvas-error" role="alert"><FileImage size={22} /><strong>Không đọc được ảnh này</strong><span>{previewError}</span><button type="button" className="subtle-button" onClick={refreshPreview}>Thử tải lại</button></div>}
          {!previewLoading && !previewError && preview && fitSize && (
            <div className="label-image-transform" style={{ width: fitSize.width * zoom, height: fitSize.height * zoom, transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px)` } as CSSProperties}>
              <img src={preview.data_url} style={{ imageRendering: fitSize.scale >= 1 ? "pixelated" : "auto" }} alt={`${imageName(image.path)}, ${preview.width} × ${preview.height} px`} draggable={false} />
              {!sizeMismatch && <svg
                className="label-annotation-canvas"
                viewBox={`0 0 ${preview.width} ${preview.height}`}
                preserveAspectRatio="none"
                role="application"
                aria-label="Canvas gán nhãn. Dùng công cụ hiển thị để vẽ, chọn, di chuyển và sửa box. Mũi tên di chuyển box, Alt cùng mũi tên đổi kích thước."
                tabIndex={0}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={() => setCurrentGesture(null)}
                onKeyDown={onCanvasKeyDown}
              >
                {visibleAnnotations.map((annotation, index) => {
                  const color = annotation.class_id === 0 ? "#68a9ff" : annotation.class_id === 1 ? "#54c9a8" : "#d8b46a";
                  const selected = annotation.id === selectedAnnotationId;
                  const labelHeight = 22 / (fitSize.scale || 1);
                  const labelWidth = Math.max(54, classNameFor(classes, annotation.class_id).length * 8) / (fitSize.scale || 1);
                  const x = annotation.x_min;
                  const y = annotation.y_min;
                  const w = annotation.x_max - annotation.x_min;
                  const h = annotation.y_max - annotation.y_min;
                  const aiStroke = annotation.source === "ai" ? `${8 / (fitSize.scale || 1)} ${5 / (fitSize.scale || 1)}` : undefined;
                  return (
                    <g key={annotation.id} className={`label-box-group${annotation.source === "ai" ? " ai-suggestion" : ""}`} data-class-id={annotation.class_id}>
                      <rect data-annotation-id={annotation.id} x={x} y={y} width={w} height={h} fill={color} fillOpacity={selected ? 0.16 : 0.05} stroke={color} strokeWidth={strokeWidth} strokeDasharray={aiStroke} />
                      <rect className="label-box-title-bg" x={x} y={Math.max(0, y - labelHeight)} width={labelWidth} height={labelHeight} fill={color} pointerEvents="none" />
                      <text className="label-box-title" x={x + 6 / (fitSize.scale || 1)} y={Math.max(labelHeight * 0.72, y - labelHeight * 0.28)} fontSize={11 / (fitSize.scale || 1)} pointerEvents="none">{classNameFor(classes, annotation.class_id)}</text>
                      <title>{`${index + 1}. ${classNameFor(classes, annotation.class_id)}${annotation.source === "ai" ? ` · Gợi ý AI · ${annotation.review_status === "needs_review" ? "cần duyệt" : annotation.review_status === "rejected" ? "đã loại" : "đã duyệt"} · ${annotation.confidence == null ? "confidence không rõ" : `${Math.round(annotation.confidence * 100)}%`}` : ""} · ${Math.round(w)} × ${Math.round(h)} px`}</title>
                    </g>
                  );
                })}
                {visibleAnnotations.filter((annotation) => annotation.id === selectedAnnotationId).map((annotation) => {
                  const color = annotation.class_id === 0 ? "#68a9ff" : annotation.class_id === 1 ? "#54c9a8" : "#d8b46a";
                  const x = annotation.x_min;
                  const y = annotation.y_min;
                  const w = annotation.x_max - annotation.x_min;
                  const h = annotation.y_max - annotation.y_min;
                  const handles: { name: ResizeHandle; x: number; y: number }[] = [
                    { name: "nw", x, y }, { name: "n", x: x + w / 2, y }, { name: "ne", x: x + w, y },
                    { name: "e", x: x + w, y: y + h / 2 }, { name: "se", x: x + w, y: y + h },
                    { name: "s", x: x + w / 2, y: y + h }, { name: "sw", x, y: y + h }, { name: "w", x, y: y + h / 2 },
                  ];
                  return <g key={`handles-${annotation.id}`} className="label-box-handles">{handles.map((handle) => <rect
                    key={handle.name}
                    data-handle={handle.name}
                    data-annotation-id={annotation.id}
                    x={handle.x - handleSize / 2}
                    y={handle.y - handleSize / 2}
                    width={handleSize}
                    height={handleSize}
                    rx={2 / (fitSize.scale || 1)}
                    fill="#f7f9fc"
                    stroke={color}
                    strokeWidth={strokeWidth}
                  />)}</g>;
                })}
                {draft && <rect x={draft.x_min} y={draft.y_min} width={draft.x_max - draft.x_min} height={draft.y_max - draft.y_min} fill="#68a9ff" fillOpacity="0.12" stroke="#68a9ff" strokeWidth={strokeWidth} strokeDasharray={`${8 / (fitSize.scale || 1)} ${5 / (fitSize.scale || 1)}`} pointerEvents="none" />}
              </svg>}
            </div>
          )}
          {sizeMismatch && <div className="label-canvas-warning" role="alert">Kích thước ảnh trên đĩa đã đổi từ {image.width} × {image.height} sang {preview?.width} × {preview?.height}. Box cũ được giữ nguyên nhưng tạm khóa để tránh lệch tọa độ. Hãy dùng ảnh đã trích xuất ban đầu.</div>}
        </div>

        <div className="label-canvas-footer">
          <span>{preview ? `${preview.width} × ${preview.height} px` : imageName(image.path)}</span>
          <span>{activeAnnotations.length} box{pendingSuggestions.length ? ` · ${pendingSuggestions.length} AI chờ duyệt` : ""}{pendingNegativeCandidates.length ? ` · ${pendingNegativeCandidates.length} candidate trống` : ""}</span>
          <span className={`label-save-status ${saveState.status}`} aria-live="polite">
            {saveState.status === "saving" ? <><LoaderCircle className="spin" size={12} /> Đang lưu…</> : saveState.status === "error" ? "Chưa lưu" : <><Check size={12} /> Đã tự lưu</>}
          </span>
        </div>
        {saveState.status === "error" && <div className="label-save-error" role="alert">
          <span>{saveState.message}</span>
          <button type="button" className="subtle-button" onClick={refreshPreview}>Tải lại ảnh</button>
          {preview?.image_fingerprint && !sizeMismatch && <button type="button" className="subtle-button" onClick={retrySave}>Thử lưu lại</button>}
        </div>}
      </section>

      <aside className="panel label-inspector-panel">
        <div className="settings-header"><h2>Công cụ gán nhãn</h2><span>{activeAnnotations.length} box</span></div>
        <div className="label-inspector-scroll">
          <section className="label-tool-section label-ai-section" aria-label="Gợi ý nhãn bằng AI">
            <div className="label-ai-heading">
              <h3><Sparkles size={13} /> Gợi ý AI</h3>
              <div className="label-ai-heading-actions">
                <button type="button" className="label-icon-button" aria-label="Làm mới checkpoint" title="Làm mới checkpoint" disabled={modelsLoading || inferenceBusy || modelImportBusy} onClick={() => void refreshModels()}><RefreshCw size={12} /></button>
              </div>
            </div>
            {modelControls}
            {inferenceBusy ? (
              <button type="button" className="subtle-button label-ai-run-button" disabled={inferenceCancelling} onClick={() => void cancelInference()}>
                <X size={13} /> {inferenceCancelling ? "Đang dừng…" : "Dừng inference"}
              </button>
            ) : (
              <>
                <button type="button" className="label-ai-run-button" disabled={!canEdit || !selectedModel || !confidenceValid} onClick={() => void startInference()}>
                  <Sparkles size={13} /> {imageHasSuggestionState ? "Gợi ý lại ảnh này" : "Gợi ý ảnh này"}
                </button>
                <button type="button" className="label-ai-run-button label-ai-batch-button" disabled={!selectedModel || batchCount === 0 || !confidenceValid} onClick={() => void startBatchInference()}>
                  <Sparkles size={13} /> {resuggestLabeled ? `Gợi ý lại ${batchCount} ảnh` : `Gợi ý ${batchCount} ảnh chưa xử lý`}
                </button>
                <label className="label-ai-replace"><input type="checkbox" checked={resuggestLabeled} onChange={(event) => setResuggestLabeled(event.target.checked)} /> Gợi ý lại cả ảnh đã có box (xóa box cũ)</label>
              </>
            )}
            {inferenceBusy && <small className="label-ai-progress" aria-live="polite"><LoaderCircle className="spin" size={12} /> {inferenceCancelling
              ? "Đang dừng an toàn và lưu phần đã xử lý…"
              : inferenceProgress?.stage?.startsWith("project_")
                ? `Đang xử lý project: ${inferenceProgress.completed_images ?? 0}/${inferenceProgress.total_images ?? inferenceProgress.eligible_images ?? 0} ảnh${inferenceProgress.saved_images ? ` · đã lưu ${inferenceProgress.saved_images}` : ""}`
                : inferenceProgress?.stage === "inference_started" ? "Đang chạy YOLO trên ảnh hiện tại…" : "Đang khởi tạo inference…"}</small>}
            {inferenceMessage && <small className="label-ai-note" aria-live="polite">{inferenceMessage}</small>}
            {inferenceError && <small className="label-ai-error" role="alert">Inference chưa hoàn tất: {inferenceError}</small>}
          </section>

          <section className="label-tool-section" aria-label="Chọn class để vẽ box">
            <h3>Lớp cần vẽ</h3>
            <div className="label-class-tools">
              {classes.map((name, classId) => <button
                key={`${classId}-${name}`}
                type="button"
                className={`label-class-tool${tool.kind === "draw" && tool.classId === classId ? " active" : ""}`}
                aria-pressed={tool.kind === "draw" && tool.classId === classId}
                disabled={!canEdit}
                onClick={() => { setSelectedAnnotationId(null); setTool({ kind: "draw", classId }); }}
              >
                <span className={`label-class-swatch class-${classId}`} />
                <span><strong>{classNameFor(classes, classId)}</strong>{shortcutLabel(classId) && <small>Phím {shortcutLabel(classId)}</small>}</span>
              </button>)}
            </div>
          </section>

          <section className="label-tool-section">
            <h3>Chọn và sửa</h3>
            <div className="label-edit-tools">
              <button type="button" className={`subtle-button${tool.kind === "select" ? " active" : ""}`} aria-pressed={tool.kind === "select"} onClick={() => setTool({ kind: "select" })}><MousePointer2 size={13} /> Chọn / sửa</button>
              <button type="button" className={`subtle-button${tool.kind === "pan" ? " active" : ""}`} aria-pressed={tool.kind === "pan"} onClick={() => setTool({ kind: "pan" })}><Hand size={13} /> Di chuyển ảnh</button>
              <button type="button" className="subtle-button danger" disabled={!selectedAnnotation || !canEdit} onClick={deleteSelected}><Trash2 size={13} /> Xóa box</button>
            </div>
            <div className="label-history-tools">
              <button type="button" className="subtle-button" disabled={!history?.past.length || !canEdit} onClick={undo}><Undo2 size={13} /> Hoàn tác</button>
              <button type="button" className="subtle-button" disabled={!history?.future.length || !canEdit} onClick={redo}><Redo2 size={13} /> Làm lại</button>
            </div>
          </section>

          <section className="label-tool-section">
            <div className="label-review-heading"><h3>Danh sách box</h3><button type="button" className={`label-review-button${image.status === "reviewed" ? " reviewed" : ""}`} disabled={!canEdit || (image.status !== "reviewed" && (pendingSuggestions.length > 0 || pendingNegativeCandidates.length > 0))} title={(pendingSuggestions.length || pendingNegativeCandidates.length) && image.status !== "reviewed" ? "Duyệt hoặc loại toàn bộ gợi ý và candidate AI trước khi đánh dấu ảnh đã review." : undefined} aria-pressed={image.status === "reviewed"} onClick={markReviewed}>{image.status === "reviewed" ? <><CheckCircle2 size={13} /> Đã review</> : <><Check size={13} /> Đánh dấu đã review</>}</button></div>
            {pendingSuggestions.length > 0 && <div className="label-ai-review-actions label-ai-bulk-actions">
              <button type="button" className="label-ai-accept" disabled={!canEdit} onClick={() => reviewAllSuggestions("reviewed")}><Check size={11} /> Duyệt tất cả ({pendingSuggestions.length})</button>
              <button type="button" className="label-ai-reject" disabled={!canEdit} onClick={() => reviewAllSuggestions("rejected")}><Ban size={11} /> Loại tất cả</button>
            </div>}
            <div className="label-box-list" role="list" aria-label="Box trên ảnh">
              {activeAnnotations.map((annotation, index) => (
                <div className={`label-box-row${annotation.id === selectedAnnotationId ? " selected" : ""}`} role="listitem" key={annotation.id}>
                  <button type="button" className="label-box-select" aria-pressed={annotation.id === selectedAnnotationId} onClick={() => { setSelectedAnnotationId(annotation.id); setTool({ kind: "select" }); }}>
                    <span className={`label-class-swatch class-${annotation.class_id}`} />
                    <span><strong>{index + 1}. {classNameFor(classes, annotation.class_id)}</strong><small>{Math.round(annotation.x_max - annotation.x_min)} × {Math.round(annotation.y_max - annotation.y_min)} px</small></span>
                  </button>
                  <select aria-label={`Đổi class cho box ${index + 1}`} value={annotation.class_id} disabled={!canEdit} onChange={(event) => changeClass(annotation, Number(event.target.value))}>
                    {classes.map((name, classId) => <option key={`${classId}-${name}`} value={classId}>{classNameFor(classes, classId)}</option>)}
                  </select>
                  <button type="button" className="label-icon-button danger" aria-label={`Xóa box ${index + 1}`} title="Xóa box" disabled={!canEdit} onClick={() => {
                    if (annotation.id === selectedAnnotationId) setSelectedAnnotationId(null);
                    commit(annotation.source === "ai"
                      ? annotations.map((item) => item.id === annotation.id ? { ...item, review_status: "rejected" } : item)
                      : annotations.filter((item) => item.id !== annotation.id));
                  }}><Trash2 size={13} /></button>
                  {annotation.source === "ai" && <div className="label-box-ai-meta">
                    <span className={`label-ai-badge${annotation.review_status === "needs_review" ? " pending" : " accepted"}`}>
                      AI · {annotation.review_status === "needs_review" ? "Cần duyệt" : "Đã duyệt"} · {annotation.confidence == null ? "—" : `${Math.round(annotation.confidence * 100)}%`}
                    </span>
                    {annotation.review_status === "needs_review" && <div className="label-ai-review-actions">
                      <button type="button" className="label-ai-accept" disabled={!canEdit} onClick={() => reviewSuggestion(annotation, "reviewed")}><Check size={11} /> Duyệt</button>
                      <button type="button" className="label-ai-reject" disabled={!canEdit} onClick={() => reviewSuggestion(annotation, "rejected")}><Ban size={11} /> Loại</button>
                    </div>}
                  </div>}
                </div>
              ))}
              {activeAnnotations.length === 0 && <p className="label-box-empty">Chưa có box. Chọn lớp để vẽ hoặc chạy gợi ý AI cho ảnh này.</p>}
            </div>
          </section>

          {pendingNegativeCandidates.length > 0 && <section className="label-tool-section" aria-label="Review candidate âm tính">
            <h3>Candidate âm tính cần review</h3>
            {pendingNegativeCandidates.map((candidate) => {
              const stale = preview?.image_fingerprint !== candidate.image_fingerprint;
              const canConfirmNegative = canEdit && !stale && activeAnnotations.length === 0 && pendingNegativeCandidates.length === 1;
              return <div className="label-negative-candidate" key={candidate.id}>
                <small>Model {candidate.source_model_id.slice(0, 8)} không phát hiện box. Ảnh chưa vào dataset.</small>
                {stale && <small className="label-negative-stale">Ảnh đã đổi kể từ inference; hãy loại candidate này rồi chạy lại.</small>}
                {pendingNegativeCandidates.length > 1 && <small className="label-negative-stale">Review các candidate khác của ảnh trước khi xác nhận âm tính.</small>}
                {activeAnnotations.length > 0 && <small className="label-negative-stale">Ảnh đang có box; không thể xác nhận là ảnh âm tính.</small>}
                <div className="label-ai-review-actions">
                  <button type="button" className="label-ai-accept" disabled={!canConfirmNegative || reviewingCandidateId !== null} onClick={() => void reviewCandidate(candidate, "confirmed_negative")}>
                    <Check size={11} /> Xác nhận ảnh trống
                  </button>
                  <button type="button" className="label-ai-reject" disabled={!canEdit || reviewingCandidateId !== null} onClick={() => void reviewCandidate(candidate, "rejected")}>
                    <Ban size={11} /> {stale ? "Loại candidate cũ" : "Loại candidate"}
                  </button>
                </div>
              </div>;
            })}
          </section>}

          {selectedAnnotation && <p className="label-coordinate-help">Box #{annotations.findIndex((item) => item.id === selectedAnnotation.id) + 1}: x {Math.round(selectedAnnotation.x_min)}, y {Math.round(selectedAnnotation.y_min)} → {Math.round(selectedAnnotation.x_max)}, {Math.round(selectedAnnotation.y_max)} px</p>}
        </div>
      </aside>
    </section>
  );
}
