import { open, confirm } from "@tauri-apps/plugin-dialog";

export type RegisteredModel = {
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

export type ImportedModel = {
  run_id: string;
  checkpoint_path: string;
  class_names: string[];
  architecture: string;
  display_name: string;
};

type EngineRequest = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

/** Usable checkpoints: present, checksum verified, trusted and with a class schema. */
export function availableModels(models: RegisteredModel[]): RegisteredModel[] {
  return models.filter((model) => (
    model.checkpoint_usable && typeof model.checkpoint_path === "string" && (model.class_names?.length ?? 0) > 0
  ));
}

export function modelMatchesClasses(model: RegisteredModel, classes: string[]): boolean {
  return model.class_names?.length === classes.length && model.class_names.every((name, index) => name === classes[index]);
}

/** Picks a `.pt`, asks the user to trust it, then imports it through the verified registry flow. */
export async function pickAndImportCheckpoint(engineRequest: EngineRequest): Promise<ImportedModel | null> {
  const selected = await open({
    title: "Chọn checkpoint YOLO11n",
    multiple: false,
    filters: [{ name: "YOLO checkpoint", extensions: ["pt"] }],
  });
  if (typeof selected !== "string") return null;
  const trusted = await confirm(
    "Checkpoint .pt có thể chứa dữ liệu thực thi mã khi được nạp. Chỉ tiếp tục nếu bạn tự tạo tệp hoặc tin tưởng nguồn của nó. FrameForge sẽ kiểm tra task, kiến trúc YOLO11n và class, rồi sao chép checkpoint vào thư mục quản lý riêng.",
    { title: "Xác nhận tin cậy checkpoint", kind: "warning", okLabel: "Tin cậy và nạp", cancelLabel: "Hủy" },
  );
  if (!trusted) return null;
  return engineRequest<ImportedModel>("import_yolo_checkpoint", { path: selected, trust_source: true });
}
