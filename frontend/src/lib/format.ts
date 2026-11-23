// Định dạng số liệu, thời gian và tách dòng: hàm thuần, không phụ thuộc React hay trình duyệt.

export function gb(bytes: number | null | undefined): string {
  if (bytes == null) return "–";
  if (bytes < 1e9) return (bytes / 1e6).toFixed(0) + " MB";
  return (bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1) + " GB";
}

export function speed(bytesPerSecond: number): string {
  if (bytesPerSecond >= 1e6) return (bytesPerSecond / 1e6).toFixed(1) + " MB/s";
  return (bytesPerSecond / 1e3).toFixed(0) + " KB/s";
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null) return "–";
  const s = Math.max(0, Math.round(seconds));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  if (hours) return `${hours} giờ ${minutes} phút`;
  if (minutes) return `${minutes} phút ${s % 60} giây`;
  return `${s} giây`;
}

export function compact(n: number | null | undefined): string {
  if (n == null) return "–";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

export function clamp01(x: number | null | undefined): number {
  return Math.max(0, Math.min(1, x || 0));
}

export function percent(fraction: number, digits = 1): string {
  return (fraction * 100).toFixed(digits) + "%";
}

/** Chỉ số: unit "%" thì đổi tỉ lệ 0..1 thành phần trăm; còn lại giữ 3 chữ số có nghĩa. */
export function formatMetric(value: number | null | undefined, unit = ""): string {
  if (value == null || Number.isNaN(value)) return "–";
  if (unit === "%") {
    const pct = value * 100;
    return pct.toFixed(Math.abs(pct) < 1 && pct !== 0 ? 2 : 1) + "%";
  }
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 1000) return compact(value);
  if (abs >= 0.001) return String(Number(value.toPrecision(3)));
  return value.toExponential(1);
}

/** Thời điểm (giây Unix) theo giờ Việt Nam. */
export function formatTime(seconds: number | null | undefined): string {
  if (!seconds) return "–";
  return new Date(seconds * 1000).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}

/** Thông báo lỗi để hiện cho người dùng. */
export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function splitLines(text: string): string[] {
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}
