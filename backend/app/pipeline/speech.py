"""Speech / silence timeline from the 16 kHz WAV.

Silero VAD (ONNX, bundled with faster-whisper, no torch) runs over the audio in chunks so
memory stays flat regardless of episode length. The same pass records loudness per hop.
"""

import wave
from pathlib import Path

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

CHUNK_S = 300
OVERLAP_S = 5
LOUDNESS_HOP_S = 0.5
MERGE_GAP_S = 0.15
MIN_GAP_S = 0.3

VAD_OPTIONS = VadOptions(
    threshold=0.45,             # slightly below default: missing quiet speech is worse than a false alarm
    min_speech_duration_ms=150,
    min_silence_duration_ms=200,
    speech_pad_ms=80,
)


def _merge(intervals: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def detect_speech(wav_path: Path) -> dict:
    speech: list[tuple[float, float]] = []
    loudness: list[float] = []

    with wave.open(str(wav_path), "rb") as wav:
        sr = wav.getframerate()
        total = wav.getnframes()
        hop = int(LOUDNESS_HOP_S * sr)

        # Loudness: one sequential pass
        while frames := wav.readframes(hop * 240):
            block = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            usable = len(block) // hop * hop
            if usable:
                rms = np.sqrt(np.mean(block[:usable].reshape(-1, hop) ** 2, axis=1))
                loudness.extend(np.round(20 * np.log10(rms + 1e-6), 1).tolist())

        # VAD: overlapping chunks, results shifted to absolute time and merged
        step, overlap = CHUNK_S * sr, OVERLAP_S * sr
        for chunk_start in range(0, total, step):
            read_from = max(0, chunk_start - overlap)
            wav.setpos(read_from)
            frames = wav.readframes(min(total - read_from, step + 2 * overlap))
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            for seg in get_speech_timestamps(audio, VAD_OPTIONS, sampling_rate=sr):
                speech.append(((read_from + seg["start"]) / sr, (read_from + seg["end"]) / sr))

    duration = total / sr
    speech = _merge(speech, MERGE_GAP_S)
    gaps, cursor = [], 0.0
    for start, end in speech:
        if start - cursor >= MIN_GAP_S:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= MIN_GAP_S:
        gaps.append((cursor, duration))

    return {
        "duration": round(duration, 3),
        "speech": [[round(s, 3), round(e, 3)] for s, e in speech],
        "gaps": [[round(s, 3), round(e, 3)] for s, e in gaps],
        "speech_ratio": round(sum(e - s for s, e in speech) / duration, 3) if duration else 0.0,
        "loudness_db": {"hop_s": LOUDNESS_HOP_S, "values": loudness},
    }


def gap_at(gaps: list[list[float]], t: float) -> tuple[float, float] | None:
    """The silence gap containing time t, if any."""
    for start, end in gaps:
        if start <= t <= end:
            return start, end
        if start > t:
            break
    return None
