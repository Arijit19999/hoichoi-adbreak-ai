import { useEffect, useState } from "react";
import { api, type Brand, type Rules } from "../api";

const EXAMPLE_BRAND = {
  brand_id: "brand_i",
  display_name: "Brand I",
  category: "home appliances",
  target_contexts: ["kitchen", "cooking", "household chores", "cleaning", "home", "family at home", "refrigerator"],
  negative_contexts: ["funeral", "grief", "violence", "accident", "hospital", "fire"],
};

const RULE_FIELDS: { key: keyof Rules; label: string; step: number }[] = [
  { key: "max_breaks_per_hour", label: "Max breaks / hour", step: 1 },
  { key: "min_gap_s", label: "Min gap between breaks (s)", step: 30 },
  { key: "max_ad_load_pct", label: "Max ad load (%)", step: 1 },
  { key: "no_break_before_s", label: "No break in first (s)", step: 30 },
  { key: "no_break_in_last_s", label: "No break in last (s)", step: 30 },
  { key: "min_break_score", label: "Min break score", step: 0.05 },
];

export function RulesPanel({ initial, onReplan, busy }: { initial: Rules; onReplan: (r: Rules) => void; busy: boolean }) {
  const [rules, setRules] = useState<Rules>(initial);
  useEffect(() => setRules(initial), [initial]);
  return (
    <div className="rounded-xl bg-zinc-900 p-4 ring-1 ring-white/10">
      <div className="mb-3 text-sm font-semibold">Pacing rules</div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {RULE_FIELDS.map((f) => (
          <label key={f.key} className="text-[11px] text-zinc-400">
            {f.label}
            <input
              type="number"
              step={f.step}
              value={rules[f.key]}
              onChange={(e) => setRules({ ...rules, [f.key]: Number(e.target.value) })}
              className="mt-1 w-full rounded-md bg-black/40 px-2 py-1 text-sm text-white ring-1 ring-white/10"
            />
          </label>
        ))}
      </div>
      <button
        disabled={busy}
        onClick={() => onReplan(rules)}
        className="mt-3 rounded-md bg-brand px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
      >
        {busy ? "Re-planning…" : "Re-plan breaks with these rules"}
      </button>
      <p className="mt-2 text-[11px] text-zinc-500">
        Re-planning reuses the cached scene analysis: only break scoring, brand matching, audit and the manifest are recomputed.
      </p>
    </div>
  );
}

export function BrandsPanel({ onChanged }: { onChanged: () => void }) {
  const [brands, setBrands] = useState<Brand[]>([]);
  const [draft, setDraft] = useState(JSON.stringify(EXAMPLE_BRAND, null, 2));
  const [message, setMessage] = useState("");

  const load = () => api.brands().then(setBrands).catch((e) => setMessage(String(e)));
  useEffect(() => {
    load();
  }, []);

  const save = async () => {
    try {
      const saved = await api.saveBrand(JSON.parse(draft));
      setMessage(`Saved ${saved.brand_id}. Re-plan a video to use it: no code changes needed.`);
      await load();
      onChanged();
    } catch (e) {
      setMessage(String(e));
    }
  };

  const remove = async (id: string) => {
    await api.deleteBrand(id);
    await load();
    onChanged();
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
      <div className="space-y-2">
        {brands.map((b) => (
          <div key={b.brand_id} className="rounded-lg bg-zinc-900 p-3 ring-1 ring-white/10">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">
                {b.display_name} <span className="font-normal text-zinc-400">· {b.category}</span>
              </div>
              <button onClick={() => remove(b.brand_id)} className="text-[11px] text-zinc-500 hover:text-rose-300">
                remove
              </button>
            </div>
            <div className="mt-1 flex flex-wrap gap-1">
              {b.target_contexts.map((c) => (
                <span key={c} className="rounded bg-emerald-950 px-1.5 py-0.5 text-[10px] text-emerald-200">{c}</span>
              ))}
            </div>
            <div className="mt-1 flex flex-wrap gap-1">
              {b.negative_contexts.map((c) => (
                <span key={c} className="rounded bg-rose-950 px-1.5 py-0.5 text-[10px] text-rose-200">⛔ {c}</span>
              ))}
            </div>
            <div className="mt-1 text-[10px] text-zinc-500">
              creatives: {b.creatives.map((c) => `${c.duration_sec}s`).join(", ")}
            </div>
          </div>
        ))}
      </div>
      <div className="rounded-xl bg-zinc-900 p-4 ring-1 ring-white/10">
        <div className="text-sm font-semibold">Add / update a brand</div>
        <p className="mt-1 text-[11px] text-zinc-400">
          Same schema as brands.json. Matching uses only these context lists, so an unseen brand works without code changes.
          Missing creatives get a placeholder video.
        </p>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          spellCheck={false}
          className="mt-2 h-72 w-full rounded-md bg-black/50 p-2 font-mono text-[11px] text-zinc-200 ring-1 ring-white/10"
        />
        <button onClick={save} className="mt-2 rounded-md bg-brand px-3 py-1.5 text-sm font-semibold text-white">
          Save brand
        </button>
        {message && <p className="mt-2 text-[11px] text-zinc-300">{message}</p>}
      </div>
    </div>
  );
}
