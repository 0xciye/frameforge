import { useEffect, useMemo, useRef, useState } from "react";
import { FileImage, ImagePlus } from "lucide-react";

export type GalleryImage = {
  path: string;
  width: number;
  height: number;
  status: string;
  source_path?: string;
  frame_index?: number;
};

type Props = {
  images: GalleryImage[];
  seedThumbnails: Record<string, string>;
  loadThumbnails(paths: string[]): Promise<Record<string, string | null>>;
  onSelect(image: GalleryImage): void;
};

// Matches .gallery-item width (112px) plus the strip gap (8px).
const ITEM_STRIDE = 120;
const STRIP_PADDING = 9;
const OVERSCAN = 8;
const THUMBNAIL_BATCH = 64;
const MAX_CACHED_THUMBNAILS = 3000;

function fileName(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

export default function ProjectGallery({ images, seedThumbnails, loadThumbnails, onSelect }: Props) {
  const stripRef = useRef<HTMLDivElement | null>(null);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [thumbnails, setThumbnails] = useState<Map<string, string | null>>(() => new Map());
  const pending = useRef(new Set<string>());
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  // Newest extraction first, matching the previous gallery order.
  const ordered = useMemo(() => [...images].reverse(), [images]);

  useEffect(() => {
    const entries = Object.entries(seedThumbnails);
    if (!entries.length) return;
    setThumbnails((current) => {
      const next = new Map(current);
      for (const [path, url] of entries) next.set(path, url);
      return next;
    });
  }, [seedThumbnails]);

  useEffect(() => {
    const node = stripRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setViewportWidth(entry.contentRect.width));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const first = Math.max(0, Math.floor((scrollLeft - STRIP_PADDING) / ITEM_STRIDE) - OVERSCAN);
  const last = Math.min(ordered.length, Math.ceil((scrollLeft + viewportWidth) / ITEM_STRIDE) + OVERSCAN);
  const visible = useMemo(() => ordered.slice(first, last), [ordered, first, last]);

  useEffect(() => {
    const missing = visible
      .map((image) => image.path)
      .filter((path) => !thumbnails.has(path) && !pending.current.has(path))
      .slice(0, THUMBNAIL_BATCH);
    if (!missing.length) return;
    for (const path of missing) pending.current.add(path);
    loadThumbnails(missing)
      .then((loaded) => {
        if (!mounted.current) return;
        setThumbnails((current) => {
          const next = new Map(current);
          for (const path of missing) next.set(path, loaded[path] ?? null);
          // Drop the oldest entries so very large projects do not keep every thumbnail in memory.
          while (next.size > MAX_CACHED_THUMBNAILS) next.delete(next.keys().next().value as string);
          return next;
        });
      })
      .catch(() => undefined)
      .finally(() => {
        for (const path of missing) pending.current.delete(path);
      });
  }, [visible, thumbnails, loadThumbnails]);

  return (
    <div className="gallery-strip" ref={stripRef} onScroll={(event) => setScrollLeft(event.currentTarget.scrollLeft)}>
      {ordered.length === 0 ? (
        <div className="gallery-empty"><ImagePlus size={18} /><span>Frame xuất hiện tại đây sau lần trích đầu tiên.</span></div>
      ) : (
        <div className="gallery-track" style={{ width: ordered.length * ITEM_STRIDE - 8 }}>
          {visible.map((image, offset) => {
            const thumbnail = thumbnails.get(image.path);
            const name = fileName(image.path);
            return (
              <button
                type="button"
                className="gallery-item"
                key={image.path}
                style={{ left: (first + offset) * ITEM_STRIDE }}
                onClick={() => onSelect(image)}
                title={`${name} · ${image.width} × ${image.height}`}
              >
                {thumbnail ? <img src={thumbnail} alt="" /> : <div className="gallery-placeholder"><FileImage size={20} /></div>}
                <span className="gallery-file-name">{name}</span>
                <small>{image.width} × {image.height}{image.status === "reviewed" ? " · đã review" : ""}</small>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
