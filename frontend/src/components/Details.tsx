import { fmtTime, type Candidate, type DebugScene } from "../api";

const FEATURE_LABELS: Record<string, string> = {
  interruptibility: "Natural pause (AI)",
  ending: "Scene ending",
  transition: "Shot transition",
  silence_before: "Silence before cut",
  silence_after: "Silence after cut",
  loudness_dip: "Audio dip",
  boundary_confidence: "Boundary confidence",
};

function Bar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full rounded bg-white/10">
      <div className="h-1.5 rounded bg-emerald-400" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

export function BreakCard({ c, scenes, onWatch }: { c: Candidate; scenes: DebugScene[]; onWatch: (t: number) => void }) {
  const p = c.placement!;
  const before = scenes[c.scene_before];
  const after = scenes[c.scene_after];
  const blocked = Object.entries(c.brand_safety ?? {}).filter(([, reasons]) => reasons.length);
  return (
    <div className="rounded-xl bg-zinc-900 p-4 ring-1 ring-white/10">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-xs uppercase tracking-wide text-amber-400">
            {p.id} · {fmtTime(c.time, true)} · {c.kind.replace("_", " ")}
          </div>
          <div className="text-lg font-semibold">
            {p.display_name} <span className="text-sm font-normal text-zinc-400">· {p.category} · {p.creative.duration_sec}s</span>
          </div>
        </div>
        <button
          onClick={() => onWatch(Math.max(0, c.time - 6))}
          className="rounded-md bg-amber-400 px-3 py-1.5 text-sm font-semibold text-black hover:bg-amber-300"
        >
          ▶ Watch the cut
        </button>
      </div>

      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-semibold text-zinc-300">
            Where: break score {c.score?.toFixed(2)} · {c.transition?.replace("_", " ")} · silence {c.silence_before?.toFixed(1)}s before
          </div>
          <div className="space-y-1">
            {Object.entries(c.features ?? {}).map(([k, v]) => (
              <div key={k} className="grid grid-cols-[9rem_1fr_2.5rem] items-center gap-2 text-[11px] text-zinc-400">
                <span>{FEATURE_LABELS[k] ?? k}</span>
                <Bar value={v} />
                <span className="text-right">{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="mt-2 text-[11px] text-zinc-400">
            <b className="text-zinc-300">Before:</b> {before?.dominant_activity} — {before?.summary}
          </div>
          <div className="mt-1 text-[11px] text-zinc-400">
            <b className="text-zinc-300">After:</b> {after?.dominant_activity} — {after?.summary}
          </div>
        </div>

        <div>
          <div className="mb-1 text-xs font-semibold text-zinc-300">What: brand fit {p.fit.toFixed(2)}</div>
          <p className="text-[12px] text-zinc-300">{p.reason}</p>
          <div className="mt-2 space-y-0.5 text-[11px]">
            {c.brand_ranking?.map((r) => (
              <div key={r.brand_id} className="flex justify-between text-zinc-400">
                <span className={r.brand_id === p.brand_id ? "font-semibold text-amber-300" : ""}>{r.brand_id}</span>
                <span>fit {r.fit.toFixed(2)} (AI {r.llm_fit ?? "–"}/10, keywords {r.context_overlap.toFixed(1)})</span>
              </div>
            ))}
          </div>
          {blocked.length > 0 && (
            <details className="mt-2 text-[11px] text-rose-300">
              <summary className="cursor-pointer">{blocked.length} brand(s) hard-blocked by negative contexts</summary>
              {blocked.map(([id, reasons]) => (
                <div key={id} className="mt-1">
                  <b>{id}</b>: {reasons.join("; ")}
                </div>
              ))}
            </details>
          )}
          {c.audits?.map((a, i) => (
            <div key={i} className={`mt-2 text-[11px] ${a.verdict === "safe" ? "text-emerald-300" : "text-rose-300"}`}>
              Independent audit ({a.model}) on {a.brand_id}: <b>{a.verdict}</b> — {a.reason}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function CandidatesTable({ candidates, onWatch }: { candidates: Candidate[]; onWatch: (t: number) => void }) {
  return (
    <div className="overflow-x-auto rounded-xl ring-1 ring-white/10">
      <table className="w-full text-left text-[12px]">
        <thead className="bg-zinc-900 text-zinc-400">
          <tr>
            <th className="px-3 py-2">Cut</th>
            <th className="px-3 py-2">Score</th>
            <th className="px-3 py-2">+ fit</th>
            <th className="px-3 py-2">Safe brands</th>
            <th className="px-3 py-2">Decision</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c) => (
            <tr key={c.time} className="border-t border-white/5 align-top">
              <td className="px-3 py-1.5">
                <button className="text-sky-300 hover:underline" onClick={() => onWatch(Math.max(0, c.time - 6))}>
                  {fmtTime(c.time, true)}
                </button>
                <div className="text-[10px] text-zinc-500">{c.kind.replace("_", " ")}</div>
              </td>
              <td className="px-3 py-1.5">{c.score?.toFixed(2) ?? "–"}</td>
              <td className="px-3 py-1.5">{c.selection_score?.toFixed(2) ?? "–"}</td>
              <td className="px-3 py-1.5">{c.safe_brands ?? "–"}</td>
              <td className="px-3 py-1.5">
                {c.placement ? (
                  <span className="font-semibold text-amber-300">PLACED · {c.placement.display_name}</span>
                ) : (
                  <span className="text-zinc-400">{c.rejected[0] ?? "eligible"}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ScenesTable({ scenes, onWatch }: { scenes: DebugScene[]; onWatch: (t: number) => void }) {
  return (
    <div className="overflow-x-auto rounded-xl ring-1 ring-white/10">
      <table className="w-full text-left text-[12px]">
        <thead className="bg-zinc-900 text-zinc-400">
          <tr>
            <th className="px-3 py-2">#</th>
            <th className="px-3 py-2">Time</th>
            <th className="px-3 py-2">Dominant activity · summary</th>
            <th className="px-3 py-2">Ending</th>
            <th className="px-3 py-2">Negative contexts</th>
          </tr>
        </thead>
        <tbody>
          {scenes.map((s) => (
            <tr key={s.index} className="border-t border-white/5 align-top">
              <td className="px-3 py-1.5 text-zinc-500">{s.index}</td>
              <td className="px-3 py-1.5 whitespace-nowrap">
                <button className="text-sky-300 hover:underline" onClick={() => onWatch(s.start)}>
                  {fmtTime(s.start)}–{fmtTime(s.end)}
                </button>
                <div className="text-[10px] text-zinc-500">{s.boundary_in?.cut?.kind?.replace("_", " ") ?? (s.index ? "no clean cut" : "start")}</div>
              </td>
              <td className="px-3 py-1.5">
                <div className="font-medium text-zinc-200">{s.dominant_activity} <span className="font-normal text-zinc-500">· {s.mood}</span></div>
                <div className="text-zinc-400">{s.summary}</div>
              </td>
              <td className="px-3 py-1.5 whitespace-nowrap text-zinc-400">
                {s.ending}
                <div className="text-[10px]">pause {s.interruptibility}/10</div>
              </td>
              <td className="px-3 py-1.5">
                <div className="flex flex-wrap gap-1">
                  {s.flagged_contexts.map((c) => (
                    <span
                      key={c.context}
                      title={c.evidence}
                      className={`rounded px-1.5 py-0.5 text-[10px] ${
                        c.verdict === "present" ? "bg-rose-900 text-rose-100" : "bg-amber-900 text-amber-100"
                      }`}
                    >
                      {c.context}
                    </span>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
