// Thử model đang chạy: chat, decider, embedding, rerank, sinh ảnh, giọng nói.
import type { DecideResponse, EmbedResponse, ImageResponse, RerankResponse, TtsResponse, Voice } from "../types/api.ts";
import { request, requestStream } from "./client.ts";

/** Chat dạng luồng SSE (OpenAI-compatible): trả về Response để đọc dần. */
export function chatStream(body: Record<string, unknown>, abortSignal: AbortSignal): Promise<Response> {
  return requestStream("/api/chat/stream", { body, signal: abortSignal });
}

/** Strands Decider: /v1/systemone. */
export function ask(state: unknown, questions: Record<string, unknown>): Promise<DecideResponse> {
  return request("/api/ask", { body: { state, questions } });
}

export function embed(input: string[]): Promise<EmbedResponse> {
  return request("/api/embed", { body: { input } });
}

export function rerank(queryText: string, documents: string[]): Promise<RerankResponse> {
  return request("/api/rerank", { body: { query: queryText, documents } });
}

export function generateImage(body: { prompt: string; size: string; steps?: number; seed?: number }): Promise<ImageResponse> {
  return request("/api/image", { body });
}

export async function ttsVoices(): Promise<Voice[]> {
  const data = await request<{ data: Voice[] }>("/api/tts/voices");
  return data.data;
}

export function speak(body: Record<string, unknown>): Promise<TtsResponse> {
  return request("/api/tts", { body });
}
