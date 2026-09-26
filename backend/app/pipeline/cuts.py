"""Frame-accurate shot-cut refinement around an approximate timestamp.

The scene model gives boundaries at ~1 s precision. Cutting to an ad a few frames off a real
shot change shows a flash of the next scene, so each proposed boundary is snapped to the
nearest hard cut or fade-to-black by decoding only a few seconds around it.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .media import MediaInfo, read_gray_frames

HARD_CUT_MIN_DIFF = 18.0     # mean abs luma difference (0-255) between consecutive frames
HARD_CUT_MIN_RATIO = 4.0     # ... and this many times the window's typical motion
BLACK_LUMA = 18.0            # mean luma below this counts as a black frame


@dataclass
class Cut:
    time: float          # first frame of the new shot (or middle of the black run)
    kind: str            # "hard_cut" | "fade_black"
    strength: float      # diff / median diff, or black-run length in seconds
    offset: float        # distance from the proposed time

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


def find_cut_near(video: Path, t: float, info: MediaInfo, window: float = 2.5) -> Cut | None:
    times, frames = read_gray_frames(video, t - window, 2 * window, info)
    if len(frames) < 3:
        return None

    luma = frames.reshape(len(frames), -1).mean(axis=1)
    black = luma < BLACK_LUMA
    if black.any():
        # Longest run of black frames: a fade-out/in is the most natural break there is.
        best, run_start = (0, 0), None
        for i, is_black in enumerate([*black, False]):
            if is_black and run_start is None:
                run_start = i
            elif not is_black and run_start is not None:
                if i - run_start > best[1] - best[0]:
                    best = (run_start, i)
                run_start = None
        mid = times[(best[0] + best[1] - 1) // 2]
        length = (best[1] - best[0]) / info.fps
        if length >= 0.2:
            return Cut(float(mid), "fade_black", float(length), float(abs(mid - t)))

    diffs = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2))
    typical = float(np.median(diffs)) + 1.0
    # Prefer the strongest cut, lightly penalised by distance from the proposed time.
    distance = np.abs(times[1:] - t)
    score = diffs / typical - distance
    i = int(np.argmax(score))
    if diffs[i] >= HARD_CUT_MIN_DIFF and diffs[i] / typical >= HARD_CUT_MIN_RATIO:
        return Cut(float(times[i + 1]), "hard_cut", float(diffs[i] / typical), float(abs(times[i + 1] - t)))
    return None
