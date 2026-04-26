"""
WebSocket consumers for the Vision and Audio engines.

VideoConsumer: receives JPEG frames, returns emotion/score JSON.
AudioConsumer: receives PCM audio chunks, returns transcript/metrics JSON.
"""

import asyncio
import base64
import json
import struct
import time

import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

from .audio_engine import (
    AudioChunk,
    AudioTranscriber,
    RollingAudioBuffer,
    ToneAnalyzer,
    compute_audio_performance_metrics,
    float32_to_wav_bytes,
    measure_audio_signal,
    process_audio_chunk,
    should_transcribe_audio,
    trim_transcript_overlap,
    LOOKBACK_SECONDS,
    librosa,
    Groq,
)
from .vision_engine import (
    load_emotion_model,
    load_face_detector,
    process_video_frame,
)


class VideoConsumer(AsyncWebsocketConsumer):
    """
    Receives JPEG frames from the browser webcam, runs emotion detection,
    and sends back annotated frame + scores as JSON.
    """

    async def connect(self):
        await self.accept()
        # Load model in a thread so we don't block the event loop
        self.model = await asyncio.to_thread(
            load_emotion_model, str(settings.EMOTION_MODEL_PATH)
        )
        self.face_detector = await asyncio.to_thread(load_face_detector)
        await self.send(text_data=json.dumps({"type": "status", "message": "Vision Engine ready."}))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        """
        Expects either:
          - text_data: JSON with { "frame": "<base64 JPEG>" }
          - bytes_data: raw JPEG bytes
        """
        try:
            if bytes_data:
                jpeg_bytes = bytes_data
            elif text_data:
                data = json.loads(text_data)
                frame_b64 = data.get("frame", "")
                if not frame_b64:
                    return
                jpeg_bytes = base64.b64decode(frame_b64)
            else:
                return

            result = await asyncio.to_thread(
                process_video_frame, jpeg_bytes, self.model, self.face_detector
            )

            await self.send(text_data=json.dumps({
                "type": "vision_result",
                "emotion": result["emotion"],
                "probability": result["probability"],
                "score": result["score"],
                "faces_detected": result["faces_detected"],
                "annotated_frame": result["annotated_frame"],
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": str(e),
            }))


class AudioConsumer(AsyncWebsocketConsumer):
    """
    Receives PCM audio chunks from the browser microphone, runs VAD + tone
    analysis + Groq transcription, and sends back metrics/transcript JSON.
    """

    async def connect(self):
        await self.accept()

        self.audio_buffer = RollingAudioBuffer()
        self.tone_analyzer = None
        self.transcriber = None
        self.transcript_lines = []
        self.last_transcript_text = ""
        self.lookback_samples = np.empty(0, dtype=np.float32)
        self.chunks_processed = 0
        self.chunks_skipped = 0
        self.groq_requests = 0
        self._worker_running = True

        # Initialize optional components
        if librosa is not None:
            self.tone_analyzer = ToneAnalyzer(target_sample_rate=16000)

        api_key = settings.GROQ_API_KEY
        if api_key and Groq is not None:
            try:
                self.transcriber = AudioTranscriber(api_key=api_key)
            except Exception:
                self.transcriber = None

        # Start the background VAD worker
        self._worker_task = asyncio.create_task(self._vad_worker())

        status = {
            "type": "status",
            "message": "Audio Engine ready.",
            "has_transcriber": self.transcriber is not None,
            "has_tone_analyzer": self.tone_analyzer is not None,
        }
        await self.send(text_data=json.dumps(status))

    async def disconnect(self, close_code):
        self._worker_running = False
        if hasattr(self, "_worker_task"):
            self._worker_task.cancel()

    async def receive(self, text_data=None, bytes_data=None):
        """
        Expects bytes_data: raw PCM int16 mono audio at 16kHz.
        First 4 bytes = sample rate as uint32 little-endian.
        """
        if bytes_data:
            if len(bytes_data) < 6:
                return
            sample_rate = struct.unpack("<I", bytes_data[:4])[0]
            pcm_bytes = bytes_data[4:]
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            float_samples = pcm_array.astype(np.float32) / 32768.0
            self.audio_buffer.append(float_samples, sample_rate)

    async def _vad_worker(self):
        """Background loop: poll VAD, process complete phrases."""
        while self._worker_running:
            try:
                chunk = self.audio_buffer.pop_vad_phrase()

                if chunk is None:
                    await asyncio.sleep(0.1)
                    continue

                context = ""
                if self.last_transcript_text:
                    words = self.last_transcript_text.strip().split()
                    context = " ".join(words[-15:])

                result = await asyncio.to_thread(
                    process_audio_chunk,
                    chunk,
                    self.tone_analyzer,
                    self.transcriber,
                    lookback_samples=self.lookback_samples,
                    lookback_seconds=LOOKBACK_SECONDS,
                    context_words=context,
                )

                text = result["transcription"].get("text", "")
                self.lookback_samples = result.get("next_lookback", np.empty(0, dtype=np.float32))
                self.chunks_processed += 1

                if result["transcription"].get("skipped"):
                    self.chunks_skipped += 1
                elif self.transcriber is not None:
                    self.groq_requests += 1

                if text:
                    stitched = trim_transcript_overlap(self.last_transcript_text, text)
                    if stitched:
                        self.transcript_lines.append(stitched)
                        self.transcript_lines = self.transcript_lines[-30:]
                    self.last_transcript_text = text

                await self.send(text_data=json.dumps({
                    "type": "audio_result",
                    "metrics": result["metrics"],
                    "tone": result["tone"],
                    "transcription": {
                        "text": result["transcription"].get("text", ""),
                        "filler_counts": result["transcription"].get("filler_counts", {}),
                        "filler_total": result["transcription"].get("filler_total", 0),
                        "filler_per_minute": result["transcription"].get("filler_per_minute", 0.0),
                        "skipped": result["transcription"].get("skipped", True),
                        "gate_reason": result["transcription"].get("gate_reason", ""),
                    },
                    "audio_quality": result["audio_quality"],
                    "transcript_lines": self.transcript_lines,
                    "chunks_processed": self.chunks_processed,
                    "chunks_skipped": self.chunks_skipped,
                    "groq_requests": self.groq_requests,
                    "buffered_seconds": self.audio_buffer.buffered_seconds(),
                }))

            except asyncio.CancelledError:
                break
            except Exception as e:
                try:
                    await self.send(text_data=json.dumps({
                        "type": "error",
                        "message": str(e),
                    }))
                except Exception:
                    pass
                await asyncio.sleep(0.5)
