import { fmtTime, type Break, type Candidate, type DebugScene } from "../api";

type Props = {
  duration: number;
  scenes: DebugScene[];
  candidates: Candidate[];
  breaks: Break[];
  current: number;
  onSeek: (t: number) => void;
};

function sceneTone(s: DebugScene, i: number): string {
  if (s.flagged_contexts.some((c) => c.verdict === "present")) return "bg-rose-900/70 hover:bg-rose-800";
  if (s.flagged_contexts.some((c) => c.verdict === "possible")) return "bg-amber-900/60 hover:bg-amber-800";
  return i % 2 ? "bg-zinc-700 hover:bg-zinc-600" : "bg-zinc-600 hover:bg-zinc-500";
}

export function Timeline({ duration, scenes, candidates, breaks, current, onSeek }: Props) {
  const pct = (t: number) => `${(100 * t) / duration}%`;
  const ticks = Array.from({ length: Math.floor(duration / 300) + 1 }, (_, i) => i * 300);

  return (
    <div className="select-none">
      <div className="relative h-6">
        {breaks.map((b) => (
          <button
            key={b.id}
            onClick={() => onSeek(Math.max(0, b.time - 5))}
            className="absolute -translate-x-1/2 whitespace-nowrap rounded bg-amber-400 px-1.5 text-[10px] font-bold text-black hover:bg-amber-300"
            style={{ left: pct(b.time) }}
            title={`${b.id} at ${fmtTime(b.time, true)}: ${b.display_name}. Click to watch from 5s before.`}
          >
            ▼ {b.display_name}
          </button>
        ))}
      </div>
      <div
        className="relative h-12 cursor-pointer overflow-hidden rounded-md ring-1 ring-white/10"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          onSeek(((e.clientX - rect.left) / rect.width) * duration);
        }}
      >
        {scenes.map((s, i) => (
          <div
            key={s.index}
            className={`absolute top-0 h-full border-r border-black/60 ${sceneTone(s, i)}`}
            style={{ left: pct(s.start), width: pct(s.end - s.start) }}
            title={`Scene ${s.index} · ${fmtTime(s.start)}–${fmtTime(s.end)}\n${s.dominant_activity} · ${s.mood}\n${s.summary}${
              s.flagged_contexts.length
                ? "\n⚠ " + s.flagged_contexts.map((c) => `${c.context} (${c.verdict})`).join(", ")
                : ""
            }`}
          >
            <span className="pointer-events-none absolute left-1 top-1 text-[10px] text-white/70">{s.index}</span>
          </div>
        ))}
        {candidates.map((c) => (
          <div
            key={c.time}
            className={`pointer-events-none absolute top-0 h-full w-0.5 ${
              c.placement ? "bg-amber-400" : c.rejected.length ? "bg-black/70" : "bg-emerald-400"
            }`}
            style={{ left: pct(c.time) }}
          />
        ))}
        <div className="pointer-events-none absolute top-0 h-full w-0.5 bg-white" style={{ left: pct(current) }} />
      </div>
      <div className="relative mt-1 h-4 text-[10px] text-zinc-500">
        {ticks.map((t) => (
          <span key={t} className="absolute -translate-x-1/2" style={{ left: pct(t) }}>
            {fmtTime(t)}
          </span>
        ))}
      </div>
      <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-zinc-400">
        <span><i className="mr-1 inline-block h-2 w-3 bg-zinc-600" />scene</span>
        <span><i className="mr-1 inline-block h-2 w-3 bg-rose-900" />negative context present</span>
        <span><i className="mr-1 inline-block h-2 w-3 bg-amber-900" />negative context possible</span>
        <span><i className="mr-1 inline-block h-2 w-0.5 bg-amber-400" /> placed break</span>
        <span><i className="mr-1 inline-block h-2 w-0.5 bg-black ring-1 ring-zinc-500" /> rejected candidate</span>
      </div>
    </div>
  );
}
