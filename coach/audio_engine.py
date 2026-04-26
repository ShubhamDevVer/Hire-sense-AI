"""
Audio Engine — VAD, tone analysis, and Groq Whisper transcription.

Ported from the Streamlit app.py. Framework-agnostic: no Django or Streamlit
imports. The Django consumer calls these functions from its async context.
"""

import io
import os
import re
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully
# ---------------------------------------------------------------------------
try:
    import librosa
except ImportError:
    librosa = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# ---------------------------------------------------------------------------
# Whisper prompt — STYLE EXAMPLE, not instructions
# ---------------------------------------------------------------------------
WHISPER_FILLER_PROMPT = (
    "Um, so I think, uh, the main point here is, ah, that we need to "
    "consider all the, mm, different options before making a decision."
)

# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------
HALLUCINATION_PHRASES = [
    "thank you", "thanks for watching", "subscribe", "like and subscribe",
    "please subscribe", "transcribe", "transcript", "no clear human speech",
    "return an empty", "will return nothing", "transcription by",
    "subtitles by", "translated by", "captions by", "amara.org", "the end", "you",
]

# Regex patterns
FILLER_PATTERN = re.compile(r"\b(?:u+m+|u+h+|a+h+|m{2,})\b", re.IGNORECASE)
WORD_SANITIZE_PATTERN = re.compile(r"[^a-z0-9']+")

# ---------------------------------------------------------------------------
# VAD configuration
# ---------------------------------------------------------------------------
VAD_FRAME_SECONDS = 0.1
VAD_SILENCE_THRESHOLD_RMS = 0.01
VAD_MAX_PAUSE_SECONDS = 1.0
VAD_MIN_PHRASE_SECONDS = 0.5
VAD_MAX_PHRASE_SECONDS = 15.0
VAD_MAX_SILENCE_SECONDS = 5.0

# ---------------------------------------------------------------------------
# Audio transcription gate thresholds
# ---------------------------------------------------------------------------
LOOKBACK_SECONDS = 1.2
MAX_BUFFER_SECONDS = 45.0
MIN_TRANSCRIBE_RMS = 0.0025
MIN_TRANSCRIBE_PEAK = 0.015
MIN_TRANSCRIBE_ACTIVE_PCT = 15.0
MAX_TRANSCRIBE_DEAD_AIR_PCT = 85.0


@dataclass
class AudioChunk:
    """A small microphone chunk kept entirely in RAM."""
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


@dataclass
class RollingAudioBuffer:
    """Thread-safe rolling buffer with Voice Activity Detection."""

    max_seconds: float = MAX_BUFFER_SECONDS
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    samples: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), init=False
    )
    sample_rate: Optional[int] = field(default=None, init=False)
    absolute_start: int = field(default=0, init=False)
    next_chunk_start: int = field(default=0, init=False)

    _vad_scan_cursor: int = field(default=0, init=False)
    _vad_speaking: bool = field(default=False, init=False)
    _vad_silence_count: int = field(default=0, init=False)
    _vad_speech_frames: int = field(default=0, init=False)
    _vad_total_frames: int = field(default=0, init=False)

    def append(self, new_samples: np.ndarray, sample_rate: int) -> None:
        """Append mono float32 samples into the rolling RAM buffer."""
        if new_samples.size == 0:
            return

        with self.lock:
            if self.sample_rate is not None and self.sample_rate != sample_rate:
                self._reset_locked(sample_rate)
            if self.sample_rate is None:
                self.sample_rate = sample_rate

            self.samples = np.concatenate(
                [self.samples, new_samples.astype(np.float32)]
            )
            max_samples = int(self.max_seconds * self.sample_rate)
            if len(self.samples) > max_samples:
                drop_count = len(self.samples) - max_samples
                self.samples = self.samples[drop_count:]
                self.absolute_start += drop_count
                self.next_chunk_start = max(self.next_chunk_start, self.absolute_start)
                self._vad_scan_cursor = max(
                    self._vad_scan_cursor, self.absolute_start
                )

    def pop_vad_phrase(self) -> Optional[AudioChunk]:
        """Slice out complete phrases using Voice Activity Detection."""
        with self.lock:
            if self.sample_rate is None:
                return None
            sr = self.sample_rate
            self._vad_scan_cursor = max(
                self._vad_scan_cursor, self.absolute_start, self.next_chunk_start,
            )
            scan_rel = self._vad_scan_cursor - self.absolute_start
            frame_size = int(VAD_FRAME_SECONDS * sr)
            if len(self.samples) - scan_rel < frame_size:
                return None
            new_samples = self.samples[scan_rel:].copy()
            scan_abs_start = self._vad_scan_cursor

        max_pause_frames = int(VAD_MAX_PAUSE_SECONDS / VAD_FRAME_SECONDS)
        max_phrase_frames = int(VAD_MAX_PHRASE_SECONDS / VAD_FRAME_SECONDS)
        flush_silence_frames = int(VAD_MAX_SILENCE_SECONDS / VAD_FRAME_SECONDS)

        boundary_offset = None
        frames_scanned = 0

        for i in range(0, len(new_samples) - frame_size + 1, frame_size):
            frame = new_samples[i : i + frame_size]
            rms = float(np.sqrt(np.mean(np.square(frame))))
            frames_scanned = i + frame_size
            self._vad_total_frames += 1

            if rms > VAD_SILENCE_THRESHOLD_RMS:
                if not self._vad_speaking:
                    self._vad_speaking = True
                self._vad_silence_count = 0
                self._vad_speech_frames += 1
                if self._vad_total_frames >= max_phrase_frames:
                    boundary_offset = i + frame_size
                    break
            else:
                self._vad_silence_count += 1
                if self._vad_speaking:
                    if self._vad_silence_count >= max_pause_frames:
                        boundary_offset = i + frame_size
                        break
                else:
                    if self._vad_silence_count >= flush_silence_frames:
                        boundary_offset = i + frame_size
                        break

        self._vad_scan_cursor = scan_abs_start + frames_scanned

        if boundary_offset is not None:
            boundary_abs = scan_abs_start + boundary_offset
            with self.lock:
                chunk_start_rel = self.next_chunk_start - self.absolute_start
                chunk_end_rel = boundary_abs - self.absolute_start
                if chunk_start_rel < 0 or chunk_end_rel > len(self.samples):
                    self._reset_vad_state_locked()
                    return None

                chunk_data = self.samples[chunk_start_rel:chunk_end_rel].copy()
                duration = len(chunk_data) / float(sr)

                if self._vad_speech_frames > 0 and (
                    self._vad_speech_frames * VAD_FRAME_SECONDS
                ) < VAD_MIN_PHRASE_SECONDS:
                    return None

                self.next_chunk_start = self.absolute_start + chunk_end_rel
                self._reset_vad_state_locked()
                return AudioChunk(
                    samples=chunk_data, sample_rate=sr, duration_seconds=duration,
                )
        return None

    def buffered_seconds(self) -> float:
        with self.lock:
            if self.sample_rate is None:
                return 0.0
            return len(self.samples) / float(self.sample_rate)

    def _reset_vad_state_locked(self) -> None:
        self._vad_scan_cursor = self.next_chunk_start
        self._vad_speaking = False
        self._vad_silence_count = 0
        self._vad_speech_frames = 0
        self._vad_total_frames = 0

    def _reset_locked(self, sample_rate: int) -> None:
        self.samples = np.empty(0, dtype=np.float32)
        self.sample_rate = sample_rate
        self.absolute_start = 0
        self.next_chunk_start = 0
        self._reset_vad_state_locked()


class AudioTranscriber:
    """Groq Whisper wrapper for exact, filler-aware transcription."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3", language: str = "en"):
        if Groq is None:
            raise ImportError("Install the groq package to enable transcription.")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY.")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.language = language

    def transcribe_wav_bytes(self, wav_bytes: bytes, duration_seconds: float, context_words: str = "") -> Dict:
        if context_words:
            prompt = f"{context_words} {WHISPER_FILLER_PROMPT}"
        else:
            prompt = WHISPER_FILLER_PROMPT

        wav_file = io.BytesIO(wav_bytes)
        wav_file.name = "hire_sense_audio_chunk.wav"
        transcription = self.client.audio.transcriptions.create(
            file=wav_file, model=self.model, prompt=prompt,
            response_format="json", language=self.language, temperature=0.0,
        )
        raw_text = (getattr(transcription, "text", "") or "").strip()
        text = filter_hallucinated_text(raw_text)
        filler_counts = count_filler_words(text)
        filler_total = int(sum(filler_counts.values()))
        filler_per_minute = filler_total / max(duration_seconds / 60.0, 1e-6)
        return {
            "text": text, "filler_counts": filler_counts,
            "filler_total": filler_total, "filler_per_minute": filler_per_minute,
        }


class ToneAnalyzer:
    """Local librosa analyzer for pauses and pitch stability."""

    def __init__(self, target_sample_rate: int = 16000, silence_top_db: int = 30):
        if librosa is None:
            raise ImportError("Install librosa to enable local tone analysis.")
        self.target_sample_rate = target_sample_rate
        self.silence_top_db = silence_top_db

    def prepare_audio(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        y = np.asarray(samples, dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if sample_rate != self.target_sample_rate:
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=self.target_sample_rate)
        return np.ascontiguousarray(y, dtype=np.float32)

    def analyze(self, samples: np.ndarray, sample_rate: int) -> Dict:
        y = np.asarray(samples, dtype=np.float32)
        duration_seconds = len(y) / float(sample_rate)
        if len(y) < sample_rate * 0.5 or np.max(np.abs(y)) < 1e-5:
            return self._empty_result(duration_seconds)

        intervals = librosa.effects.split(y, top_db=self.silence_top_db, frame_length=2048, hop_length=512)
        voiced_samples = int(sum(end - start for start, end in intervals))
        dead_air_pct = 100.0 * (1.0 - voiced_samples / max(len(y), 1))
        long_pause_count = self._count_long_pauses(intervals, len(y), sample_rate)

        try:
            f0, _, _ = librosa.pyin(
                y, sr=sample_rate,
                fmin=librosa.note_to_hz("C2"),
                fmax=min(librosa.note_to_hz("C7"), sample_rate / 2.0 - 1.0),
            )
            valid_f0 = f0[np.isfinite(f0)]
        except Exception:
            valid_f0 = np.array([], dtype=np.float32)

        if len(valid_f0) >= 3:
            mean_f0 = float(np.mean(valid_f0))
            median_f0 = float(np.median(valid_f0))
            f0_variance = float(np.var(valid_f0))
            f0_std = float(np.std(valid_f0))
            f0_cv = f0_std / max(mean_f0, 1e-6)
            pitch_stability = float(np.clip(100.0 - f0_cv * 180.0, 0.0, 100.0))
            voice_heaviness_index = float(np.clip((250.0 - median_f0) / 2.0, 0.0, 100.0))
        else:
            mean_f0 = median_f0 = f0_variance = f0_std = f0_cv = 0.0
            pitch_stability = voice_heaviness_index = 0.0

        return {
            "duration_seconds": duration_seconds,
            "dead_air_pct": float(np.clip(dead_air_pct, 0.0, 100.0)),
            "long_pause_count": int(long_pause_count),
            "mean_f0_hz": mean_f0, "median_f0_hz": median_f0,
            "f0_variance": f0_variance, "f0_std": f0_std, "f0_cv": f0_cv,
            "pitch_stability": pitch_stability,
            "voice_heaviness_index": voice_heaviness_index,
        }

    def _count_long_pauses(self, intervals, total_samples, sample_rate, min_pause_seconds=0.35):
        if len(intervals) == 0:
            return 1
        min_pause_samples = int(min_pause_seconds * sample_rate)
        pause_count = 0
        cursor = 0
        for start, end in intervals:
            if start - cursor >= min_pause_samples:
                pause_count += 1
            cursor = end
        if total_samples - cursor >= min_pause_samples:
            pause_count += 1
        return pause_count

    @staticmethod
    def _empty_result(duration_seconds: float) -> Dict:
        return {
            "duration_seconds": duration_seconds, "dead_air_pct": 100.0,
            "long_pause_count": 0, "mean_f0_hz": 0.0, "median_f0_hz": 0.0,
            "f0_variance": 0.0, "f0_std": 0.0, "f0_cv": 0.0,
            "pitch_stability": 0.0, "voice_heaviness_index": 0.0,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def count_filler_words(text: str) -> Dict[str, int]:
    counts = {"um": 0, "uh": 0, "ah": 0, "mm": 0}
    for match in FILLER_PATTERN.findall(text.lower()):
        if match.startswith("um"):
            counts["um"] += 1
        elif match.startswith("uh"):
            counts["uh"] += 1
        elif match.startswith("ah"):
            counts["ah"] += 1
        elif match.startswith("mm"):
            counts["mm"] += 1
    return counts


def filter_hallucinated_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
    for phrase in HALLUCINATION_PHRASES:
        if cleaned == phrase or cleaned.startswith(phrase):
            return ""
    word_count = len(cleaned.split())
    if word_count <= 2 and len(cleaned) < 15:
        return ""
    return text


def _normalize_word(word: str) -> str:
    return WORD_SANITIZE_PATTERN.sub("", word.lower()).strip()


def trim_transcript_overlap(previous_text: str, current_text: str, max_overlap_words: int = 14) -> str:
    prev_words = [w for w in previous_text.strip().split() if _normalize_word(w)]
    cur_words = [w for w in current_text.strip().split() if _normalize_word(w)]
    if not prev_words or not cur_words:
        return current_text.strip()
    max_n = min(max_overlap_words, len(prev_words), len(cur_words))
    for n in range(max_n, 0, -1):
        prev_slice = [_normalize_word(w) for w in prev_words[-n:]]
        cur_slice = [_normalize_word(w) for w in cur_words[:n]]
        if prev_slice == cur_slice:
            return " ".join(cur_words[n:]).strip()
    return " ".join(cur_words).strip()


def float32_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    y = np.asarray(samples, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -1.0, 1.0)
    pcm16 = (y * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def measure_audio_signal(samples: np.ndarray, tone: Dict) -> Dict:
    y = np.asarray(samples, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    dbfs = 20.0 * np.log10(max(rms, 1e-9))
    dead_air_pct = float(tone.get("dead_air_pct", 100.0))
    active_pct = float(np.clip(100.0 - dead_air_pct, 0.0, 100.0))
    return {"rms": rms, "peak": peak, "dbfs": float(dbfs), "active_pct": active_pct, "dead_air_pct": dead_air_pct}


def should_transcribe_audio(audio_quality: Dict) -> tuple:
    rms = audio_quality["rms"]
    peak = audio_quality["peak"]
    active_pct = audio_quality["active_pct"]
    dead_air_pct = audio_quality["dead_air_pct"]
    if rms < MIN_TRANSCRIBE_RMS:
        return False, f"Skipped: RMS too low ({rms:.4f})."
    if peak < MIN_TRANSCRIBE_PEAK:
        return False, f"Skipped: peak too low ({peak:.4f})."
    if active_pct < MIN_TRANSCRIBE_ACTIVE_PCT or dead_air_pct > MAX_TRANSCRIBE_DEAD_AIR_PCT:
        return False, f"Skipped: mostly silence ({dead_air_pct:.1f}% dead air)."
    return True, "Chunk passed speech gate."


def compute_audio_performance_metrics(tone: Dict, transcription: Dict) -> Dict:
    dead_air_pct = tone.get("dead_air_pct", 0.0)
    filler_per_minute = transcription.get("filler_per_minute", 0.0)
    vocal_smoothness = 100.0 - dead_air_pct * 1.1 - tone.get("long_pause_count", 0) * 2.0
    clarity = 100.0 - filler_per_minute * 7.0 - dead_air_pct * 0.35
    tone_stability = tone.get("pitch_stability", 0.0)
    return {
        "vocal_smoothness": float(np.clip(vocal_smoothness, 0.0, 100.0)),
        "clarity": float(np.clip(clarity, 0.0, 100.0)),
        "tone_stability": float(np.clip(tone_stability, 0.0, 100.0)),
    }


def process_audio_chunk(
    chunk: AudioChunk,
    tone_analyzer: Optional[ToneAnalyzer],
    transcriber: Optional[AudioTranscriber],
    lookback_samples: Optional[np.ndarray] = None,
    lookback_seconds: float = LOOKBACK_SECONDS,
    context_words: str = "",
) -> Dict:
    """Full pipeline for one audio chunk."""
    if tone_analyzer is None:
        tone = ToneAnalyzer._empty_result(chunk.duration_seconds)
        prepared_audio = chunk.samples
        prepared_sr = chunk.sample_rate
    else:
        prepared_audio = tone_analyzer.prepare_audio(chunk.samples, chunk.sample_rate)
        prepared_sr = tone_analyzer.target_sample_rate
        tone = tone_analyzer.analyze(prepared_audio, prepared_sr)

    audio_quality = measure_audio_signal(prepared_audio, tone)
    can_transcribe, gate_reason = should_transcribe_audio(audio_quality)

    lookback = np.asarray(lookback_samples, dtype=np.float32) if lookback_samples is not None else np.empty(0, dtype=np.float32)
    transcribe_audio = np.concatenate([lookback, prepared_audio]) if lookback.size > 0 else prepared_audio

    if transcriber is not None and can_transcribe:
        wav_bytes = float32_to_wav_bytes(transcribe_audio, prepared_sr)
        transcription = transcriber.transcribe_wav_bytes(
            wav_bytes=wav_bytes,
            duration_seconds=len(transcribe_audio) / float(prepared_sr),
            context_words=context_words,
        )
        transcription["skipped"] = False
        transcription["gate_reason"] = gate_reason
    else:
        wav_bytes = float32_to_wav_bytes(transcribe_audio, prepared_sr)
        transcription = {
            "text": "", "filler_counts": {"um": 0, "uh": 0, "ah": 0, "mm": 0},
            "filler_total": 0, "filler_per_minute": 0.0, "skipped": True,
            "gate_reason": gate_reason if transcriber is not None else "Groq unavailable.",
        }

    lookback_len = int(max(0.2, lookback_seconds) * prepared_sr)
    next_lookback = prepared_audio[-lookback_len:].copy() if lookback_len > 0 else np.empty(0, dtype=np.float32)

    metrics = compute_audio_performance_metrics(tone, transcription)
    return {
        "tone": tone, "transcription": transcription, "metrics": metrics,
        "audio_quality": audio_quality, "next_lookback": next_lookback,
        "prepared_sample_rate": prepared_sr,
    }
