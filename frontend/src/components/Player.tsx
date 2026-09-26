import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { fmtTime, type VmapBreak } from "../api";

export type PlayerHandle = { seek: (t: number) => void };

type Props = {
  src: string;
  breaks: VmapBreak[];
  onTime?: (t: number) => void;
};

// Stop this far before the cut so not a single frame of the next scene shows before the ad.
const LEAD_S = 0.04;

export const Player = forwardRef<PlayerHandle, Props>(function Player({ src, breaks, onTime }, ref) {
  const main = useRef<HTMLVideoElement>(null);
  const ad = useRef<HTMLVideoElement>(null);
  const played = useRef<Set<string>>(new Set());
  const lastT = useRef(0);
  const resumeAt = useRef(0);
  const internalSeek = useRef(false);
  const activeRef = useRef<VmapBreak | null>(null);
  const [active, setActive] = useState<VmapBreak | null>(null);
  const [adLeft, setAdLeft] = useState(0);
  const [events, setEvents] = useState<string[]>([]);

  const log = (line: string) => setEvents((e) => [line, ...e].slice(0, 6));

  useEffect(() => {
    played.current = new Set();
    lastT.current = 0;
    activeRef.current = null;
    setActive(null);
    setEvents([]);
  }, [src, breaks]);

  const setMainTime = (t: number) => {
    const v = main.current;
    if (!v) return;
    internalSeek.current = true;
    v.currentTime = t;
    lastT.current = t;
  };

  const startAd = useCallback((b: VmapBreak, resume: number, why: string) => {
    const v = main.current;
    if (!v) return;
    v.pause();
    setMainTime(Math.max(0, b.time - LEAD_S));
    played.current.add(b.id);
    resumeAt.current = resume;
    activeRef.current = b;
    setActive(b);
    log(`${fmtTime(b.time, true)}  ${why}: ${b.title} (${b.duration}s)`);
  }, []);

  const finishAd = useCallback((skipped: boolean) => {
    const b = activeRef.current;
    activeRef.current = null;
    setActive(null);
    const v = main.current;
    if (!v || !b) return;
    setMainTime(resumeAt.current);
    void v.play();
    log(`${fmtTime(resumeAt.current, true)}  ${skipped ? "ad skipped (demo)" : "ad finished"}, content resumed`);
  }, []);

  // Frame-accurate break trigger: poll every animation frame while the episode plays.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const v = main.current;
      if (v && !v.paused && !activeRef.current) {
        const t = v.currentTime;
        const due = breaks.find(
          (b) => !played.current.has(b.id) && lastT.current <= b.time && t >= b.time - LEAD_S && t - lastT.current < 1.5,
        );
        if (due) startAd(due, due.time, "cut to ad");
        else lastT.current = t;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [breaks, startAd]);

  useEffect(() => {
    const a = ad.current;
    if (active && a) {
      a.src = active.mediaUrl;
      a.currentTime = 0;
      void a.play();
    }
  }, [active]);

  // Seeking forward over an unplayed break plays it first (snap-back), then continues at the target.
  const onSeeking = () => {
    const v = main.current;
    if (!v) return;
    if (internalSeek.current) {
      internalSeek.current = false;
      return;
    }
    const target = v.currentTime;
    if (target > lastT.current) {
      const skipped = breaks.filter((b) => !played.current.has(b.id) && b.time > lastT.current && b.time <= target);
      if (skipped.length) {
        startAd(skipped[skipped.length - 1], target, "seek crossed a break");
        return;
      }
    } else {
      breaks.forEach((b) => b.time > target && played.current.delete(b.id));
    }
    lastT.current = target;
  };

  useImperativeHandle(ref, () => ({
    seek: (t: number) => {
      if (activeRef.current) finishAd(true);
      breaks.forEach((b) => (b.time < t ? played.current.add(b.id) : played.current.delete(b.id)));
      setMainTime(t);
      void main.current?.play();
    },
  }));

  return (
    <div>
      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-black ring-1 ring-white/10">
        <video
          ref={main}
          src={src}
          controls={!active}
          preload="metadata"
          className="h-full w-full"
          onSeeking={onSeeking}
          onTimeUpdate={(e) => onTime?.(e.currentTarget.currentTime)}
        />
        {active && (
          <div className="absolute inset-0 bg-black">
            <video
              ref={ad}
              className="h-full w-full"
              playsInline
              onEnded={() => finishAd(false)}
              onTimeUpdate={(e) => setAdLeft(Math.max(0, active.duration - e.currentTarget.currentTime))}
            />
            <div className="absolute left-3 top-3 rounded-md bg-amber-400 px-2 py-0.5 text-xs font-bold text-black">
              AD · {active.title}
            </div>
            <div className="absolute bottom-3 left-3 rounded-md bg-black/70 px-2 py-1 text-xs text-white">
              {active.id} · resumes in {Math.ceil(adLeft)}s
            </div>
            <button
              onClick={() => finishAd(true)}
              className="absolute bottom-3 right-3 rounded-md bg-white/15 px-3 py-1 text-xs text-white hover:bg-white/25"
            >
              Skip ad (demo) ⏭
            </button>
          </div>
        )}
      </div>
      <div className="mt-2 min-h-6 space-y-0.5 font-mono text-[11px] text-zinc-400">
        {events.length === 0 ? (
          <div>Player reads the VMAP manifest: {breaks.length} mid-roll break(s) loaded.</div>
        ) : (
          events.map((e, i) => <div key={i} className={i === 0 ? "text-zinc-200" : ""}>{e}</div>)
        )}
      </div>
    </div>
  );
});
