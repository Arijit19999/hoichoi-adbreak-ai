import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, fmtTime, parseVmap, type Brand, type Debug, type Job, type Rules, type VideoDetail, type VideoSummary, type VmapBreak } from "./api";
import { BreakCard, CandidatesTable, ScenesTable } from "./components/Details";
import { BrandsPanel, RulesPanel } from "./components/Panels";
import { Player, type PlayerHandle } from "./components/Player";
import { Timeline } from "./components/Timeline";

type Tab = "breaks" | "scenes" | "candidates" | "brands" | "rules";

function Sidebar({
  videos,
  jobs,
  selected,
  onSelect,
  onJob,
}: {
  videos: VideoSummary[];
  jobs: Job[];
  selected: string | null;
  onSelect: (id: string) => void;
  onJob: () => void;
}) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setError("");
    setUploading(true);
    try {
      await api.upload(file);
      onJob();
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
    }
  };

  return (
    <aside className="space-y-4">
      <div className="rounded-xl bg-zinc-900 p-4 ring-1 ring-white/10">
        <div className="text-sm font-semibold">Process a video</div>
        <label className="mt-2 block cursor-pointer rounded-lg border border-dashed border-white/20 p-3 text-center text-xs text-zinc-400 hover:border-white/40">
          {uploading ? "Uploading…" : "Upload an episode (.mp4)"}
          <input type="file" accept="video/*" className="hidden" onChange={(e) => upload(e.target.files?.[0])} />
        </label>
        <div className="mt-2 flex gap-2">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="…or a video / Google Drive link"
            className="min-w-0 flex-1 rounded-md bg-black/40 px-2 py-1 text-xs ring-1 ring-white/10"
          />
          <button
            onClick={async () => {
              setError("");
              try {
                await api.fromUrl(url);
                setUrl("");
                onJob();
              } catch (e) {
                setError(String(e));
              }
            }}
            disabled={!url}
            className="rounded-md bg-brand px-2 text-xs font-semibold disabled:opacity-40"
          >
            Go
          </button>
        </div>
        {error && <p className="mt-2 text-[11px] text-rose-300">{error}</p>}
      </div>

      {jobs.length > 0 && (
        <div className="rounded-xl bg-zinc-900 p-4 ring-1 ring-white/10">
          <div className="text-sm font-semibold">Jobs</div>
          {jobs.slice(0, 4).map((j) => (
            <div key={j.id} className="mt-2 text-[11px]">
              <div className="flex justify-between gap-2">
                <span className="truncate text-zinc-300">{j.kind}: {j.label}</span>
                <span className={j.status === "error" ? "text-rose-300" : j.status === "done" ? "text-emerald-300" : "text-amber-300"}>
                  {j.status === "running" ? `${j.stage}…` : j.status}
                </span>
              </div>
              {j.status === "running" && <div className="truncate text-zinc-500">{j.log[j.log.length - 1]}</div>}
              {j.error && <div className="text-rose-300">{j.error}</div>}
            </div>
          ))}
        </div>
      )}

      <div className="rounded-xl bg-zinc-900 p-4 ring-1 ring-white/10">
        <div className="text-sm font-semibold">Processed videos</div>
        {videos.length === 0 && <p className="mt-2 text-[11px] text-zinc-500">None yet: upload an episode above.</p>}
        {videos.map((v) => (
          <button
            key={v.video_id}
            onClick={() => onSelect(v.video_id)}
            className={`mt-2 block w-full rounded-lg px-3 py-2 text-left text-xs ring-1 ${
              v.video_id === selected ? "bg-white/10 ring-amber-400/60" : "ring-white/10 hover:bg-white/5"
            }`}
          >
            <div className="truncate font-medium">{v.source_name}</div>
            <div className="text-[10px] text-zinc-500">
              {fmtTime(v.duration)} · {v.ready ? `${v.scenes} scenes · ${v.breaks} breaks` : "processing…"}
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}

export default function App() {
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<string | null>(() => new URLSearchParams(location.search).get("video"));
  const [detail, setDetail] = useState<VideoDetail | null>(null);
  const [debug, setDebug] = useState<Debug | null>(null);
  const [vmapBreaks, setVmapBreaks] = useState<VmapBreak[]>([]);
  const [defaultRules, setDefaultRules] = useState<Rules | null>(null);
  const [brands, setBrands] = useState<Brand[]>([]);
  const [tab, setTab] = useState<Tab>("breaks");
  const [current, setCurrent] = useState(0);
  const [reloadKey, setReloadKey] = useState(0);
  const player = useRef<PlayerHandle>(null);
  const seenDone = useRef<Set<string>>(new Set());

  const refreshVideos = useCallback(() => api.videos().then(setVideos).catch(() => undefined), []);

  useEffect(() => {
    refreshVideos();
    api.rules().then(setDefaultRules).catch(() => undefined);
  }, [refreshVideos]);

  useEffect(() => {
    if (!selected && videos.length) setSelected(videos.find((v) => v.ready)?.video_id ?? null);
  }, [videos, selected]);

  // Poll jobs; when one finishes, refresh the library and the open video.
  const jobsRef = useRef<Job[] | null>(null);
  const pollJobs = useCallback(async () => {
    const list = await api.jobs().catch(() => [] as Job[]);
    const firstPoll = jobsRef.current === null;
    jobsRef.current = list;
    setJobs(list);
    for (const j of list) {
      if (j.status === "done" && !seenDone.current.has(j.id)) {
        seenDone.current.add(j.id);
        if (firstPoll) continue; // finished before this page was opened: don't steal the selection
        await refreshVideos();
        if (j.video_id) {
          setSelected(j.video_id);
          setReloadKey((k) => k + 1);
        }
      }
    }
  }, [refreshVideos]);

  // One self-rescheduling loop: fast while a job runs, slow otherwise.
  useEffect(() => {
    let timer = 0;
    let stopped = false;
    const loop = async () => {
      await pollJobs();
      if (stopped) return;
      const busy = (jobsRef.current ?? []).some((j) => j.status === "running" || j.status === "queued");
      timer = window.setTimeout(loop, busy ? 2000 : 10000);
    };
    loop();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [pollJobs]);

  useEffect(() => {
    if (!selected) return;
    history.replaceState(null, "", `?video=${selected}`);
    setDebug(null);
    api.video(selected).then(setDetail).catch(() => setDetail(null));
    api.debug(selected).then(setDebug).catch(() => setDebug(null));
    api.manifest(selected).then((xml) => setVmapBreaks(parseVmap(xml))).catch(() => setVmapBreaks([]));
    api.brands().then(setBrands).catch(() => setBrands([]));
  }, [selected, reloadKey]);

  const placed = useMemo(() => debug?.candidates.filter((c) => c.placement) ?? [], [debug]);
  const busyReplan = jobs.some((j) => j.kind === "replan" && (j.status === "running" || j.status === "queued"));
  const watch = (t: number) => player.current?.seek(t);

  const replan = async (rules: Rules) => {
    if (!selected) return;
    await api.replan(selected, rules);
    pollJobs();
  };

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-5">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold tracking-tight">
            <span className="text-brand">AdBreak</span> AI
          </h1>
          <p className="text-xs text-zinc-400">
            Scene-aware ad breaks for Bengali drama · where to cut, whether to cut, which brand
          </p>
        </div>
        {detail?.plan && (
          <div className="flex gap-2 text-xs">
            <a className="rounded-md bg-white/10 px-3 py-1.5 hover:bg-white/20" href={`/${detail.manifest_url}`} target="_blank" rel="noreferrer">
              VMAP manifest
            </a>
            <a className="rounded-md bg-white/10 px-3 py-1.5 hover:bg-white/20" href={`/${detail.debug_url}`} target="_blank" rel="noreferrer">
              debug.json
            </a>
          </div>
        )}
      </header>

      <div className="grid gap-5 lg:grid-cols-[18rem_1fr]">
        <Sidebar videos={videos} jobs={jobs} selected={selected} onSelect={setSelected} onJob={pollJobs} />

        <main className="min-w-0 space-y-4">
          {!detail || !debug ? (
            <div className="grid aspect-video place-items-center rounded-xl bg-zinc-900 text-sm text-zinc-500 ring-1 ring-white/10">
              {selected ? "Loading…" : "Upload or pick a video to see its scenes and ad breaks."}
            </div>
          ) : (
            <>
              <Player ref={player} src={`/${detail.video_url}`} breaks={vmapBreaks} onTime={setCurrent} />
              <div className="flex flex-wrap gap-4 text-xs text-zinc-400">
                <span><b className="text-zinc-200">{debug.scenes.length}</b> scenes</span>
                <span><b className="text-zinc-200">{debug.candidates.length}</b> boundary candidates</span>
                <span><b className="text-zinc-200">{placed.length}</b> breaks placed</span>
                <span>ad load <b className="text-zinc-200">{debug.plan.ad_load_pct}%</b> ({debug.plan.ad_load_s}s)</span>
              </div>
              <Timeline
                duration={debug.plan.duration}
                scenes={debug.scenes}
                candidates={debug.candidates}
                breaks={debug.plan.breaks}
                brands={brands}
                current={current}
                onSeek={watch}
              />

              <nav className="flex flex-wrap gap-1 border-b border-white/10 text-sm">
                {(["breaks", "scenes", "candidates", "brands", "rules"] as Tab[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`-mb-px border-b-2 px-3 py-2 capitalize ${
                      tab === t ? "border-brand text-white" : "border-transparent text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    {t === "candidates" ? "all candidates" : t}
                  </button>
                ))}
              </nav>

              {tab === "breaks" &&
                (placed.length ? (
                  <div className="space-y-3">
                    {placed.map((c) => (
                      <BreakCard key={c.time} c={c} scenes={debug.scenes} onWatch={watch} />
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-zinc-400">No break met every rule for this video. See "all candidates" for why.</p>
                ))}
              {tab === "scenes" && <ScenesTable scenes={debug.scenes} onWatch={watch} />}
              {tab === "candidates" && <CandidatesTable candidates={debug.candidates} onWatch={watch} />}
              {tab === "brands" && <BrandsPanel onChanged={() => undefined} />}
              {tab === "rules" && (
                <RulesPanel initial={debug.plan.rules ?? defaultRules!} onReplan={replan} busy={busyReplan} />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
