import { useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  FileImage,
  Hand,
  LoaderCircle,
  Maximize2,
  MousePointer2,
  Redo2,
  Trash2,
  Undo2,
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
  source: string;
  review_status: "unreviewed" | "reviewed";
};

type LabelImage = {
  path: string;
  width: number;
  height: number;
  status: string;
  annotations?: Annotation[];
};

type LabelPreview = {
  data_url: string;
  width: number;
  height: number;
  image_fingerprint?: string;
};

type SaveImage = (path: string, annotations: Annotation[], status: "ready" | "reviewed", fingerprint: string) => Promise<void>;
type LoadPreview = (path: string) => Promise<LabelPreview>;
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
};

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

export default function ManualLabeling({ images, classNames, loadPreview, saveImage }: Props) {
  const classes = classNames.length ? classNames : ["head", "body"];
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
  const stageRef = useRef<HTMLDivElement | null>(null);
  const gestureRef = useRef<Gesture | null>(null);
  const previewSequence = useRef(0);
  const histories = useRef(new Map<string, History>());
  const saveRevisions = useRef(new Map<string, number>());
  const image = images.find((item) => item.path === activePath) ?? images[0] ?? null;
  const annotations = image?.annotations ?? [];
  const imageIndex = image ? images.findIndex((item) => item.path === image.path) : -1;
  const selectedAnnotation = annotations.find((item) => item.id === selectedAnnotationId) ?? null;
  const saveState = image ? saveStates[image.path] ?? { status: "saved" as const } : { status: "saved" as const };
  const sizeMismatch = Boolean(preview && image && (preview.width !== image.width || preview.height !== image.height));
  const canEdit = Boolean(image && preview && preview.image_fingerprint && !sizeMismatch);
  const fitSize = useMemo(() => {
    if (!preview || !stageSize.width || !stageSize.height) return null;
    const scale = Math.min(stageSize.width / preview.width, stageSize.height / preview.height);
    return { width: preview.width * scale, height: preview.height * scale, scale: scale * zoom };
  }, [preview, stageSize, zoom]);
  const handleSize = fitSize ? 10 / fitSize.scale : 12;
  const strokeWidth = fitSize ? 2 / fitSize.scale : 2;

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

  function commit(nextAnnotations: Annotation[], status: "ready" | "reviewed" = "ready") {
    if (!image || !canEdit || !preview) return;
    const history = histories.current.get(image.path) ?? { past: [], future: [] };
    history.past.push({ annotations: copyAnnotations(annotations), status: image.status === "reviewed" ? "reviewed" : "ready" });
    if (history.past.length > 100) history.past.shift();
    history.future = [];
    histories.current.set(image.path, history);
    const next = nextAnnotations.map((annotation) => ({
      ...annotation,
      review_status: status === "reviewed" ? "reviewed" as const : "unreviewed" as const,
    }));
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
    commit(annotations.filter((annotation) => annotation.id !== selectedAnnotation.id));
    setSelectedAnnotationId(null);
  }

  function selectImage(path: string) {
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
    if (!image || !canEdit) return;
    const status = image.status === "reviewed" ? "ready" : "reviewed";
    commit(annotations, status);
  }

  function changeClass(annotation: Annotation, classId: number) {
    commit(annotations.map((item) => item.id === annotation.id ? { ...item, class_id: classId } : item));
  }

  const history = image ? histories.current.get(image.path) : undefined;
  const visibleAnnotations = useMemo(() => {
    if (!gesture || gesture.kind === "draw" || gesture.kind === "pan") return annotations;
    const updated = gesture.kind === "move"
      ? moveBox(gesture.box, gesture.start, gesture.current, preview?.width ?? image?.width ?? 1, preview?.height ?? image?.height ?? 1)
      : resizeBox(gesture.box, gesture.handle, gesture.current, preview?.width ?? image?.width ?? 1, preview?.height ?? image?.height ?? 1);
    return annotations.map((annotation) => annotation.id === updated.id ? updated : annotation);
  }, [annotations, gesture, image?.height, image?.width, preview?.height, preview?.width]);
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
            const count = item.annotations?.length ?? 0;
            const current = item.path === image.path;
            return (
              <div role="listitem" key={item.path}>
                <button
                  className={`label-image-row${current ? " selected" : ""}`}
                  type="button"
                  aria-current={current ? "true" : undefined}
                  onClick={() => selectImage(item.path)}
                >
                  <span className={`label-review-dot${item.status === "reviewed" ? " reviewed" : ""}`} aria-label={item.status === "reviewed" ? "Đã review" : "Chưa review"} />
                  <span className="label-image-copy"><strong>{imageName(item.path)}</strong><small>{item.width} × {item.height} · {count} box</small></span>
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
            <button type="button" className="label-icon-button" aria-label="Ảnh trước" title="Ảnh trước" disabled={imageIndex <= 0} onClick={() => selectImage(images[imageIndex - 1].path)}><ArrowLeft size={14} /></button>
            <span>{imageIndex + 1} / {images.length}</span>
            <button type="button" className="label-icon-button" aria-label="Ảnh tiếp theo" title="Ảnh tiếp theo" disabled={imageIndex < 0 || imageIndex >= images.length - 1} onClick={() => selectImage(images[imageIndex + 1].path)}><ArrowRight size={14} /></button>
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
                  const handles: { name: ResizeHandle; x: number; y: number }[] = [
                    { name: "nw", x, y }, { name: "n", x: x + w / 2, y }, { name: "ne", x: x + w, y },
                    { name: "e", x: x + w, y: y + h / 2 }, { name: "se", x: x + w, y: y + h },
                    { name: "s", x: x + w / 2, y: y + h }, { name: "sw", x, y: y + h }, { name: "w", x, y: y + h / 2 },
                  ];
                  return (
                    <g key={annotation.id} className="label-box-group" data-class-id={annotation.class_id}>
                      <rect data-annotation-id={annotation.id} x={x} y={y} width={w} height={h} fill={color} fillOpacity={selected ? 0.16 : 0.05} stroke={color} strokeWidth={strokeWidth} />
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
                      <title>{`${index + 1}. ${classNameFor(classes, annotation.class_id)} · ${Math.round(w)} × ${Math.round(h)} px`}</title>
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
          <span>{annotations.length} box</span>
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
        <div className="settings-header"><h2>Công cụ gán nhãn</h2><span>{annotations.length} box</span></div>
        <div className="label-inspector-scroll">
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
            <div className="label-review-heading"><h3>Danh sách box</h3><button type="button" className={`label-review-button${image.status === "reviewed" ? " reviewed" : ""}`} disabled={!canEdit} aria-pressed={image.status === "reviewed"} onClick={markReviewed}>{image.status === "reviewed" ? <><CheckCircle2 size={13} /> Đã review</> : <><Check size={13} /> Đánh dấu đã review</>}</button></div>
            <div className="label-box-list" role="list" aria-label="Box trên ảnh">
              {annotations.map((annotation, index) => (
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
                    commit(annotations.filter((item) => item.id !== annotation.id));
                  }}><Trash2 size={13} /></button>
                </div>
              ))}
              {annotations.length === 0 && <p className="label-box-empty">Chưa có box. Chọn Head hoặc Body rồi kéo trên ảnh.</p>}
            </div>
          </section>

          <p className="label-key-help">⌘/Ctrl+Z hoàn tác · ⌘/Ctrl+Shift+Z làm lại · Delete xóa · Esc kết thúc thao tác</p>
          {selectedAnnotation && <p className="label-coordinate-help">Box #{annotations.findIndex((item) => item.id === selectedAnnotation.id) + 1}: x {Math.round(selectedAnnotation.x_min)}, y {Math.round(selectedAnnotation.y_min)} → {Math.round(selectedAnnotation.x_max)}, {Math.round(selectedAnnotation.y_max)} px</p>}
          <p className="label-coordinate-help">Mũi tên di chuyển box · Shift+mũi tên đi 10 px · Alt+mũi tên đổi kích thước.</p>
        </div>
      </aside>
    </section>
  );
}
