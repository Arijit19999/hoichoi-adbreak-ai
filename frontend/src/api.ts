export type Rules = {
  max_breaks_per_hour: number;
  min_gap_s: number;
  max_ad_load_pct: number;
  no_break_before_s: number;
  no_break_in_last_s: number;
  min_break_score: number;
};

export type Creative = { id: string; duration_sec: number; language?: string; url: string };

export type Brand = {
  brand_id: string;
  display_name: string;
  category: string;
  target_contexts: string[];
  negative_contexts: string[];
  creatives: Creative[];
};

export type Break = {
  id: string;
  time: number;
  score: number;
  selection_score?: number;
  transition: string | null;
  scene_before: number;
  scene_after: number;
  brand_id: string;
  display_name: string;
  category: string;
  creative: Creative;
  fit: number;
  reason: string;
};

export type Plan = {
  duration: number;
  rules: Rules;
  ad_load_s: number;
  ad_load_pct: number;
  breaks: Break[];
};

export type ContextCheck = { context: string; verdict: "absent" | "possible" | "present"; evidence: string };

export type DebugScene = {
  index: number;
  start: number;
  end: number;
  summary: string;
  dominant_activity: string;
  mood: string;
  ending: string;
  interruptibility: number;
  sensitive_events: string[];
  target_contexts_present: string[];
  models?: string[];
  flagged_contexts: ContextCheck[];
  boundary_in: { time: number; proposed_time: number; cut: { kind: string } | null; confidence: number } | null;
};

export type Candidate = {
  time: number;
  scene_before: number;
  scene_after: number;
  rejected: string[];
  notes: string[];
  transition?: string;
  silence_before?: number;
  silence_after?: number | null;
  features?: Record<string, number>;
  score?: number;
  best_fit?: number;
  selection_score?: number;
  safe_brands?: number;
  brand_safety?: Record<string, string[]>;
  brand_ranking?: { brand_id: string; fit: number; llm_fit: number | null; context_overlap: number; reason: string }[];
  audits?: { brand_id: string; model: string; verdict: string; violated_context: string; reason: string }[];
  placement?: Break;
};

export type Debug = {
  rules: Rules;
  scoring_weights: Record<string, number>;
  scenes: DebugScene[];
  candidates: Candidate[];
  plan: Plan;
};

export type VideoSummary = {
  video_id: string;
  source_name: string;
  duration: number;
  ready: boolean;
  scenes: number | null;
  breaks: number | null;
};

export type VideoDetail = {
  meta: VideoSummary;
  plan: Plan | null;
  video_url: string;
  manifest_url: string;
  debug_url: string;
};

export type Job = {
  id: string;
  kind: string;
  label: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  log: string[];
  video_id: string | null;
  error: string | null;
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  videos: () => request<VideoSummary[]>("/api/videos"),
  video: (id: string) => request<VideoDetail>(`/api/videos/${id}`),
  debug: (id: string) => request<Debug>(`/api/videos/${id}/debug.json`),
  manifest: async (id: string) => {
    const res = await fetch(`/api/videos/${id}/manifest.vmap`);
    if (!res.ok) throw new Error(`manifest ${res.status}`);
    return res.text();
  },
  jobs: () => request<Job[]>("/api/jobs"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Job>("/api/videos", { method: "POST", body: form });
  },
  fromUrl: (url: string) => request<Job>("/api/videos/from-url", json({ url })),
  replan: (id: string, rules: Rules) => request<Job>(`/api/videos/${id}/replan`, json(rules)),
  rules: () => request<Rules>("/api/rules"),
  brands: () => request<Brand[]>("/api/brands"),
  saveBrand: (brand: unknown) => request<Brand>("/api/brands", json(brand)),
  deleteBrand: (id: string) => request<{ deleted: string }>(`/api/brands/${id}`, { method: "DELETE" }),
};

export function fmtTime(seconds: number, withMs = false): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  const whole = Math.floor(rest).toString().padStart(2, "0");
  return withMs ? `${m}:${whole}.${Math.round((rest % 1) * 10)}` : `${m}:${whole}`;
}

export type VmapBreak = { id: string; time: number; mediaUrl: string; title: string; duration: number };

function parseOffset(value: string): number {
  const [h, m, s] = value.split(":");
  return Number(h) * 3600 + Number(m) * 60 + Number(s);
}

/** The player reads the VMAP itself, so what plays is exactly what the manifest says. */
export function parseVmap(xml: string): VmapBreak[] {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  const ns = "http://www.iab.net/videosuite/vmap";
  return Array.from(doc.getElementsByTagNameNS(ns, "AdBreak"))
    .map((el) => ({
      id: el.getAttribute("breakId") ?? "",
      time: parseOffset(el.getAttribute("timeOffset") ?? "0:0:0"),
      mediaUrl: el.getElementsByTagName("MediaFile")[0]?.textContent?.trim() ?? "",
      title: el.getElementsByTagName("AdTitle")[0]?.textContent?.trim() ?? "Ad",
      duration: parseOffset(el.getElementsByTagName("Duration")[0]?.textContent?.trim() ?? "0:0:0"),
    }))
    .filter((b) => b.mediaUrl)
    .sort((a, b) => a.time - b.time);
}
