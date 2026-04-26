/**
 * Audio Stream — Captures microphone via getUserMedia + AudioWorklet,
 * sends PCM chunks over WebSocket.
 *
 * Wire protocol: first 4 bytes = sample rate (uint32 LE), rest = PCM int16.
 */

class AudioStream {
    constructor(onResult, onStatus) {
        this.onResult = onResult;
        this.onStatus = onStatus;

        this.ws = null;
        this.stream = null;
        this.audioCtx = null;
        this.processor = null;
        this.running = false;
        this.targetSampleRate = 16000;
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: { ideal: 16000 },
                }
            });

            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.targetSampleRate,
            });

            const source = this.audioCtx.createMediaStreamSource(this.stream);

            // Use ScriptProcessorNode (widely supported) for capturing PCM
            // Buffer size 4096 at 16kHz ≈ 256ms chunks
            this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);

            this.processor.onaudioprocess = (e) => {
                if (!this.running || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

                const float32Data = e.inputBuffer.getChannelData(0);
                const int16Data = new Int16Array(float32Data.length);

                for (let i = 0; i < float32Data.length; i++) {
                    const s = Math.max(-1, Math.min(1, float32Data[i]));
                    int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                // Prepend sample rate (4 bytes, uint32 little-endian)
                const srBuffer = new ArrayBuffer(4);
                new DataView(srBuffer).setUint32(0, this.audioCtx.sampleRate, true);

                const payload = new Uint8Array(4 + int16Data.byteLength);
                payload.set(new Uint8Array(srBuffer), 0);
                payload.set(new Uint8Array(int16Data.buffer), 4);

                this.ws.send(payload.buffer);
            };

            source.connect(this.processor);
            this.processor.connect(this.audioCtx.destination);

            // Connect WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/audio/`);

            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => {
                this.running = true;
                this.onStatus('connected');
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.onResult(data);
                } catch (e) {
                    console.error('Audio parse error:', e);
                }
            };

            this.ws.onclose = () => {
                this.running = false;
                this.onStatus('disconnected');
            };

            this.ws.onerror = () => {
                this.onStatus('error');
            };

        } catch (err) {
            console.error('Audio Engine error:', err);
            this.onStatus('error');
        }
    }

    stop() {
        this.running = false;

        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }

        if (this.audioCtx) {
            this.audioCtx.close();
            this.audioCtx = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.onStatus('disconnected');
    }
}
