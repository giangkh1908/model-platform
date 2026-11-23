// Lớp HTTP duy nhất của giao diện: mọi module trong src/api/ gọi qua đây, component không gọi fetch hay viết URL.

export class ApiError extends Error {
  /** 0 = mất kết nối */
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
}

interface ErrorBody {
  error?: unknown;
  detail?: unknown;
}

function errorMessage(json: ErrorBody, response: Response): string {
  const detail = json.error ?? json.detail ?? response.statusText;
  if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg ?? "").join("; ");
  if (typeof detail === "string") return detail;
  return JSON.stringify(detail);
}

function init({ method, body, signal }: RequestOptions): RequestInit {
  const options: RequestInit = { method: method ?? (body === undefined ? "GET" : "POST"), signal };
  if (body !== undefined) {
    options.headers = { "content-type": "application/json" };
    options.body = JSON.stringify(body);
  }
  return options;
}

/** Gọi API, trả về JSON. Lỗi HTTP thành ApiError mang thông báo của backend. */
export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init(options));
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") throw e;
    throw new ApiError("Mất kết nối với nền tảng", 0);
  }
  const json: unknown = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(errorMessage(json as ErrorBody, response), response.status);
  return json as T;
}

/** Gọi API trả về luồng (SSE): trả về Response để đọc dần. Lỗi HTTP thành ApiError như `request`. */
export async function requestStream(path: string, options: RequestOptions): Promise<Response> {
  const response = await fetch(path, init(options));
  if (!response.ok) {
    const json: unknown = await response.json().catch(() => ({}));
    throw new ApiError(errorMessage(json as ErrorBody, response), response.status);
  }
  return response;
}

/** Gửi tín hiệu không cần trả lời, vẫn đi được khi trang đang đóng (keepalive). */
export function signal(path: string): void {
  fetch(path, { method: "POST", keepalive: true }).catch(() => {});
}

/** Một đoạn đường dẫn, đã mã hoá. */
export function seg(value: string): string {
  return encodeURIComponent(value);
}

type QueryValue = string | number | boolean | null | undefined;

/** Chuỗi query từ object, bỏ giá trị rỗng. */
export function query(params: Record<string, QueryValue>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (!entries.length) return "";
  return "?" + new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString();
}
