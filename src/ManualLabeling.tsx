import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  ArrowLeft,
  ArrowRight,
  Ban,
  Check,
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
  checkpoint_path?: string | null;
  checkpoint_usable?: boolean;
  checkpoint_sha256?: string | null;
  class_names?: string[];
  effective_imgsz?: number;
  requested_imgsz?: number;
};
type Prediction = { class_id: number; confidence: number; bbox_xyxy: number[] };
type InferenceProgress = { stage: string; completed_images?: number; total_images?: number; eligible_images?: number; saved_images?: number };
type InferenceResult = { status: string; results: { predictions: Prediction[] }[] };
type BatchInferenceResult = {
  status: string;
  processed_images: number;
  images_with_detections: number;
  candidate_negative_count: number;
  annotation_count: number;
  skipped_changed: number;
};
type SaveSuggestionsResult = { annotations: Annotation[]; negative_candidate?: NegativeCandidate | null };
type SaveSuggestions = (path: string, predictions: Prediction[], checkpoint: string, fingerprint: string) => Promise<SaveSuggestionsResult>;
type ReviewNegativeCandidate = (candidateId: string, action: "confirmed_negative" | "rejected", fingerprint?: string) => Promise<void>;
type RunBatchInference = (checkpoint: string, requestId: string) => Promise<BatchInferenceResult>;
type Point = { x: number; y: number };
type ResizeHandle = "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | "nw";
type Tool = { kind: "draw"; classId: number } | { kind: "select" } | { kind: "pan" };
type Gesture =
  | { kind: "draw"; pointerId: number; start: Point; current: Point; classId: number }
  | { kind: "move"; pointerId: number; start: Point; current: Point; box: Annotation }
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
};

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

export default function ManualLabeling({ images, classNames, loadPreview, saveImage, saveSuggestions, reviewNegativeCandidate, runBatchInference }: Props) {
  const classes = useMemo(() => classNames.length ? classNames : ["head", "body"], [classNames]);
  const [activePath, setActivePath] = useState(images[0]?.path ?? "");
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>({ kind: "select" });
  const [preview, setPreview] = useState<LabelPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [gesture, setGesture] = useState<Gesture | null>(null);
  const [saveStates, setSaveStates] = useState<Record<string, SaveState>>({});
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState("");
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
  const usableModels = useMemo(() => models.filter((model) => (
    model.checkpoint_usable
    && typeof model.checkpoint_path === "string"
    && model.class_names?.length === classes.length
    && model.class_names.every((name, index) => name === classes[index])
  )), [models, classes]);
  const selectedModel = usableModels.find((model) => model.run_id === selectedModelId) ?? null;
  const image = images.find((item) => item.path === activePath) ?? images[0] ?? null;
  const annotations = image?.annotations ?? [];
  const activeAnnotations = useMemo(() => annotations.filter((item) => item.review_status !== "rejected"), [annotations]);
  const pendingSuggestions = annotations.filter((item) => item.source === "ai" && item.review_status === "needs_review");
  const negativeCandidates = image?.negative_candidates ?? [];
  const pendingNegativeCandidates = negativeCandidates.filter((item) => item.review_status === "needs_review");
  const inferenceQueue = images.filter((item) => (
    item.status !== "reviewed"
    && !(item.annotations ?? []).some((annotation) => annotation.source === "manual" || annotation.review_status !== "rejected")
    && !(item.negative_candidates ?? []).some((candidate) => candidate.review_status !== "rejected")
  ));
  const imageIndex = image ? images.findIndex((item) => item.path === image.path) : -1;
  const selectedAnnotation = activeAnnotations.find((item) => item.id === selectedAnnotationId) ?? null;
  const alreadySuggestedByModel = Boolean(image && selectedModel && annotations.some((item) => (
    item.source === "ai"
    && item.review_status !== "rejected"
    && item.source_model_id === selectedModel.run_id
    && item.checkpoint_sha256 === selectedModel.checkpoint_sha256
  )) || Boolean(image && selectedModel && negativeCandidates.some((item) => (
    item.review_status !== "rejected"
    && item.source_model_id === selectedModel.run_id
    && item.checkpoint_sha256 === selectedModel.checkpoint_sha256
  ))));
  const saveState = image ? saveStates[image.path] ?? { status: "saved" as const } : { status: "saved" as const };
  const sizeMismatch = Boolean(preview && image && (preview.width !== image.width || preview.height !== image.height));
  const canEdit = Boolean(image && preview && preview.image_fingerprint && !sizeMismatch && !inferenceBusy);
  const fitSize = useMemo(() => {
    if (!preview || !stageSize.width || !stageSize.height) return null;
    const scale = Math.min(stageSize.width / preview.width, stageSize.height / preview.height);
    return { width: preview.width * scale, height: preview.height * scale, scale: scale * zoom };
  }, [preview, stageSize, zoom]);
  const handleSize = fitSize ? 10 / fitSize.scale : 12;
  const strokeWidth = fitSize ? 2 / fitSize.scale : 2;

  const refreshModels = useCallback(async () => {
    setModelsLoading(true);
    setModelError(null);
    try {
      const result = await engineRequest<{ runs: RegisteredModel[] }>("list_training_runs");
      setModels(result.runs ?? []);
    } catch (reason) {
      setModelError(String(reason));
    } finally {
      setModelsLoading(false);
    }
  }, []);

  useEffect(() => { void refreshModels(); }, [refreshModels]);

  useEffect(() => {
    if (usableModels.some((model) => model.run_id === selectedModelId)) return;
    setSelectedModelId(usableModels[0]?.run_id ?? "");
  }, [usableModels, selectedModelId]);

  useEffect(() => {
    if (images.some((item) => item.path === activePath)) return;
    setActivePath(images[0]?.path ?? "");
  }, [activePath, images]);

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
          arguments: { checkpoint: selectedModel.checkpoint_path, images: [image.path], confidence_threshold: 0.25 },
        },
      });
      if (activeInferenceId.current !== id) return;
      if (result.status === "cancelled") {
        setInferenceMessage("Đã dừng inference; không lưu gợi ý.");
        return;
      }
      const predictions = result.results[0]?.predictions ?? [];
      const saved = await saveSuggestions(image.path, predictions, selectedModel.checkpoint_path, preview.image_fingerprint);
      setInferenceMessage(saved.negative_candidate
        ? "Model không phát hiện box. Candidate âm tính đang chờ bạn xác nhận; ảnh chưa được đưa vào dataset."
        : `Đã lưu ${saved.annotations.length} gợi ý. Duyệt từng box trước khi dùng trong dataset.`);
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
    if (!selectedModel?.checkpoint_path || inferenceBusy || inferenceQueue.length === 0) return;
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
      const result = await runBatchInference(selectedModel.checkpoint_path, id);
      if (activeInferenceId.current !== id) return;
      setInferenceMessage(`${result.status === "cancelled" ? "Đã dừng" : "Hoàn tất"}: ${result.processed_images} ảnh đã lưu, ${result.images_with_detections} ảnh có box, ${result.candidate_negative_count} candidate âm tính cần review${result.skipped_changed ? `, ${result.skipped_changed} ảnh đã đổi hoặc không còn đọc được` : ""}.`);
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

  function commit(nextAnnotations: Annotation[], status: "ready" | "reviewed" = "ready") {
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
          review_status: status === "reviewed" ? "reviewed" as const : "unreviewed" as const,
          source_model_id: null,
          checkpoint_sha256: null,
        };
      }
      const changed = previous.class_id !== annotation.class_id
        || previous.x_min !== annotation.x_min || previous.y_min !== annotation.y_min
        || previous.x_max !== annotation.x_max || previous.y_max !== annotation.y_max;
      let reviewStatus = annotation.review_status;
      if (status === "reviewed" && previous.source === "manual") reviewStatus = "reviewed";
      else if (changed && previous.source === "ai" && previous.review_status === "needs_review") reviewStatus = "reviewed";
      else if (changed && previous.source === "manual") reviewStatus = "unreviewed";
      else if (reviewStatus === previous.review_status) reviewStatus = previous.review_status;
      return {
        ...annotation,
        source: previous.source,
        confidence: previous.confidence,
        source_model_id: previous.source_model_id,
        checkpoint_sha256: previous.checkpoint_sha256,
        review_status: reviewStatus,
      };
    });
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

  function selectImage(path: string) {
    if (inferenceBusy) return;
    setActivePath(path);
    setSelectedAnnotationId(null);
    setZoom(1);
    setPan({ x: 0, y: 0 });
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

  function retrySave() {
    if (!image || !preview) return;
    persist(image.path, annotations, image.status === "reviewed" ? "reviewed" : "ready", preview.image_fingerprint);
  }

  function onPointerDown(event: PointerEvent<SVGSVGElement>) {
    if (!image || !preview || !canEdit || event.button !== 0) return;
    const point = pointFromPointer(event, preview.width, preview.height);
    const target = event.target as Element;
    const handleElement = target.closest("[data-handle]") as SVGElement | null;
    const boxElement = target.closest("[data-annotation-id]") as SVGElement | null;
    const boxId = boxElement?.dataset.annotationId;
    const box = annotations.find((annotation) => annotation.id === boxId);
    event.currentTarget.focus();
    event.currentTarget.setPointerCapture(event.pointerId);

    if (tool.kind === "draw") {
      setSelectedAnnotationId(null);
      setCurrentGesture({ kind: "draw", pointerId: event.pointerId, start: point, current: point, classId: tool.classId });
    } else if (tool.kind === "pan") {
      setCurrentGesture({ kind: "pan", pointerId: event.pointerId, start: { x: event.clientX, y: event.clientY }, pan });
    } else if (box && handleElement?.dataset.handle) {
      setSelectedAnnotationId(box.id);
      setCurrentGesture({ kind: "resize", pointerId: event.pointerId, start: point, current: point, box, handle: handleElement.dataset.handle as ResizeHandle });
    } else if (box) {
      setSelectedAnnotationId(box.id);
      setCurrentGesture({ kind: "move", pointerId: event.pointerId, start: point, current: point, box });
    } else {
      setSelectedAnnotationId(null);
      setCurrentGesture(null);
    }
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const current = gestureRef.current;
    if (!current || current.pointerId !== event.pointerId) return;
    if (current.kind === "pan") {
      setPan({
        x: current.pan.x + event.clientX - current.start.x,
        y: current.pan.y + event.clientY - current.start.y,
      });
      return;
    }
    const point = pointFromPointer(event, preview?.width ?? 0, preview?.height ?? 0);
    setCurrentGesture({ ...current, current: point });
  }

  function onPointerUp(event: PointerEvent<SVGSVGElement>) {
    const current = gestureRef.current;
    if (!current || current.pointerId !== event.pointerId || !image || !preview) return;
    if (current.kind === "draw") {
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

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    const target = event.target as HTMLElement;
    if (target.closest("input, textarea, select, [contenteditable='true']")) return;
    const command = event.metaKey || event.ctrlKey;
    if (command && event.key.toLowerCase() === "z") {
      event.preventDefault();
      if (event.shiftKey) redo(); else undo();
    } else if (command && event.key.toLowerCase() === "y") {
      event.preventDefault();
      redo();
    } else if (event.key === "Delete") {
      event.preventDefault();
      deleteSelected();
    } else if (event.key === "Escape") {
      setCurrentGesture(null);
      setTool({ kind: "select" });
    }
  }

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

  if (!image) {
    return (
      <section className="label-workspace" aria-label="Không gian gán nhãn">
        <div className="panel label-empty-workspace">
          <FileImage size={28} aria-hidden="true" />
          <h1>Chưa có ảnh để gán nhãn</h1>
          <p>Trích xuất frame trước. Ảnh sẽ xuất hiện tại đây để bạn vẽ box thủ công.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="label-workspace" aria-label="Không gian gán nhãn" onKeyDown={onKeyDown}>
      <aside className="panel label-images-panel">
        <div className="panel-heading">
          <div><h1>Ảnh trong dự án</h1><span>{images.length} ảnh · {images.filter((item) => item.status === "reviewed").length} đã review</span></div>
        </div>
        <div className="label-image-list" role="list" aria-label="Danh sách ảnh cần gán nhãn">
          {images.map((item, index) => {
            const count = item.annotations?.filter((annotation) => annotation.review_status !== "rejected").length ?? 0;
            const pendingCount = (item.annotations?.filter((annotation) => annotation.source === "ai" && annotation.review_status === "needs_review").length ?? 0)
              + (item.negative_candidates?.filter((candidate) => candidate.review_status === "needs_review").length ?? 0);
            const current = item.path === image.path;
            return (
              <div role="listitem" key={item.path}>
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
        </div>
      </aside>

      <section className="panel label-canvas-panel" aria-label={`Ảnh đang gán nhãn ${imageName(image.path)}`}>
        <div className="label-canvas-toolbar">
          <div className="label-image-navigation">
            <button type="button" className="label-icon-button" aria-label="Ảnh trước" title="Ảnh trước" disabled={inferenceBusy || imageIndex <= 0} onClick={() => selectImage(images[imageIndex - 1].path)}><ArrowLeft size={14} /></button>
            <span>{imageIndex + 1} / {images.length}</span>
            <button type="button" className="label-icon-button" aria-label="Ảnh tiếp theo" title="Ảnh tiếp theo" disabled={inferenceBusy || imageIndex < 0 || imageIndex >= images.length - 1} onClick={() => selectImage(images[imageIndex + 1].path)}><ArrowRight size={14} /></button>
          </div>
          <div className="label-canvas-zoom" role="group" aria-label="Tỷ lệ ảnh">
            <button type="button" className="label-icon-button" aria-label="Thu nhỏ" title="Thu nhỏ" disabled={zoom <= 0.25} onClick={() => setZoom((current) => Math.max(0.25, Number((current - 0.25).toFixed(2))))}><ZoomOut size={14} /></button>
            <span>{Math.round(zoom * 100)}%</span>
            <button type="button" className="label-icon-button" aria-label="Phóng to" title="Phóng to" disabled={zoom >= 4} onClick={() => setZoom((current) => Math.min(4, Number((current + 0.25).toFixed(2))))}><ZoomIn size={14} /></button>
            <button type="button" className="label-icon-button" aria-label="Vừa khung" title="Vừa khung" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}><Maximize2 size={14} /></button>
          </div>
        </div>

        <div className={`label-canvas-stage${tool.kind === "draw" ? " drawing" : tool.kind === "pan" ? " panning" : ""}`} ref={stageRef}>
          {previewLoading && <div className="label-canvas-state"><LoaderCircle className="spin" size={20} /><span>Đang tải ảnh thật…</span></div>}
          {!previewLoading && previewError && <div className="label-canvas-state label-canvas-error" role="alert"><FileImage size={22} /><strong>Không đọc được ảnh này</strong><span>{previewError}</span><button type="button" className="subtle-button" onClick={refreshPreview}>Thử tải lại</button></div>}
          {!previewLoading && !previewError && preview && fitSize && (
            <div className="label-image-transform" style={{ width: fitSize.width, height: fitSize.height, transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` } as CSSProperties}>
              <img src={preview.data_url} alt={`${imageName(image.path)}, ${preview.width} × ${preview.height} px`} draggable={false} />
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
                  const handles: { name: ResizeHandle; x: number; y: number }[] = [
                    { name: "nw", x, y }, { name: "n", x: x + w / 2, y }, { name: "ne", x: x + w, y },
                    { name: "e", x: x + w, y: y + h / 2 }, { name: "se", x: x + w, y: y + h },
                    { name: "s", x: x + w / 2, y: y + h }, { name: "sw", x, y: y + h }, { name: "w", x, y: y + h / 2 },
                  ];
                  return (
                    <g key={annotation.id} className={`label-box-group${annotation.source === "ai" ? " ai-suggestion" : ""}`} data-class-id={annotation.class_id}>
                      <rect data-annotation-id={annotation.id} x={x} y={y} width={w} height={h} fill={color} fillOpacity={selected ? 0.16 : 0.05} stroke={color} strokeWidth={strokeWidth} strokeDasharray={aiStroke} />
                      <rect className="label-box-title-bg" x={x} y={Math.max(0, y - labelHeight)} width={labelWidth} height={labelHeight} fill={color} pointerEvents="none" />
                      <text className="label-box-title" x={x + 6 / (fitSize.scale || 1)} y={Math.max(labelHeight * 0.72, y - labelHeight * 0.28)} fontSize={11 / (fitSize.scale || 1)} pointerEvents="none">{classNameFor(classes, annotation.class_id)}</text>
                      {selected && handles.map((handle) => <rect
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
                      />)}
                      <title>{`${index + 1}. ${classNameFor(classes, annotation.class_id)}${annotation.source === "ai" ? ` · Gợi ý AI · ${annotation.review_status === "needs_review" ? "cần duyệt" : annotation.review_status === "rejected" ? "đã loại" : "đã duyệt"} · ${annotation.confidence == null ? "confidence không rõ" : `${Math.round(annotation.confidence * 100)}%`}` : ""} · ${Math.round(w)} × ${Math.round(h)} px`}</title>
                    </g>
                  );
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
              <button type="button" className="label-icon-button" aria-label="Làm mới checkpoint" title="Làm mới checkpoint" disabled={modelsLoading || inferenceBusy} onClick={() => void refreshModels()}><RefreshCw size={12} /></button>
            </div>
            <label className="label-ai-model">
              <span>Checkpoint nội bộ</span>
              <select aria-label="Checkpoint inference" value={selectedModelId} disabled={modelsLoading || inferenceBusy || usableModels.length === 0} onChange={(event) => setSelectedModelId(event.target.value)}>
                <option value="">{modelsLoading ? "Đang tải checkpoint…" : "Chọn checkpoint"}</option>
                {usableModels.map((model) => <option key={model.run_id} value={model.run_id}>
                  YOLO11n · {model.effective_imgsz ?? model.requested_imgsz ?? "—"} px · {model.run_id.slice(0, 8)}
                </option>)}
              </select>
            </label>
            {inferenceBusy ? (
              <button type="button" className="subtle-button label-ai-run-button" disabled={inferenceCancelling} onClick={() => void cancelInference()}>
                <X size={13} /> {inferenceCancelling ? "Đang dừng…" : "Dừng inference"}
              </button>
            ) : (
              <>
                <button type="button" className="label-ai-run-button" disabled={!canEdit || !selectedModel || alreadySuggestedByModel} onClick={() => void startInference()}>
                  <Sparkles size={13} /> Gợi ý ảnh này
                </button>
                <button type="button" className="label-ai-run-button label-ai-batch-button" disabled={!selectedModel || inferenceQueue.length === 0} onClick={() => void startBatchInference()}>
                  <Sparkles size={13} /> Gợi ý {inferenceQueue.length} ảnh chưa xử lý
                </button>
              </>
            )}
            {modelsLoading && <small className="label-ai-note">Đang đọc registry checkpoint…</small>}
            {!modelsLoading && usableModels.length === 0 && <small className="label-ai-note">Chưa có checkpoint YOLO11n nội bộ khớp class của dự án.</small>}
            {alreadySuggestedByModel && <small className="label-ai-note">Ảnh đã có gợi ý từ checkpoint này. Duyệt hoặc loại chúng trước khi chạy lại.</small>}
            {inferenceBusy && <small className="label-ai-progress" aria-live="polite"><LoaderCircle className="spin" size={12} /> {inferenceCancelling
              ? "Đang dừng an toàn và lưu phần đã xử lý…"
              : inferenceProgress?.stage?.startsWith("project_")
                ? `Đang xử lý project: ${inferenceProgress.completed_images ?? 0}/${inferenceProgress.total_images ?? inferenceProgress.eligible_images ?? 0} ảnh${inferenceProgress.saved_images ? ` · đã lưu ${inferenceProgress.saved_images}` : ""}`
                : inferenceProgress?.stage === "inference_started" ? "Đang chạy YOLO trên ảnh hiện tại…" : "Đang khởi tạo inference…"}</small>}
            {inferenceMessage && <small className="label-ai-note" aria-live="polite">{inferenceMessage}</small>}
            {modelError && <small className="label-ai-error" role="alert">Không đọc được registry checkpoint: {modelError}</small>}
            {inferenceError && <small className="label-ai-error" role="alert">Inference chưa hoàn tất: {inferenceError}</small>}
            <small className="label-ai-note">Box AI chỉ vào dataset sau khi được duyệt. Ảnh không có detection thành candidate âm tính và phải được xác nhận riêng.</small>
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
                <span><strong>{classNameFor(classes, classId)}</strong><small>Nhấn rồi kéo để vẽ</small></span>
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

          <p className="label-key-help">⌘/Ctrl+Z hoàn tác · ⌘/Ctrl+Shift+Z làm lại · Delete xóa · Esc kết thúc thao tác</p>
          {selectedAnnotation && <p className="label-coordinate-help">Box #{annotations.findIndex((item) => item.id === selectedAnnotation.id) + 1}: x {Math.round(selectedAnnotation.x_min)}, y {Math.round(selectedAnnotation.y_min)} → {Math.round(selectedAnnotation.x_max)}, {Math.round(selectedAnnotation.y_max)} px</p>}
          <p className="label-coordinate-help">Mũi tên di chuyển box · Shift+mũi tên đi 10 px · Alt+mũi tên đổi kích thước.</p>
        </div>
      </aside>
    </section>
  );
}
