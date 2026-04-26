/**
 * Video Stream — Captures webcam via getUserMedia, sends JPEG frames
 * over WebSocket, and displays the annotated response.
 */

class VideoStream {
    constructor(canvasEl, placeholderEl, onResult, onStatus) {
        this.canvas = canvasEl;
        this.ctx = canvasEl.getContext('2d');
        this.placeholder = placeholderEl;
        this.onResult = onResult;
        this.onStatus = onStatus;

        this.video = document.createElement('video');
        this.video.setAttribute('playsinline', '');
        this.video.setAttribute('autoplay', '');
        this.video.muted = true;

        this.ws = null;
        this.stream = null;
        this.running = false;
        this._waitingForResponse = false;  // ping-pong gate
        this._captureCanvas = document.createElement('canvas');
        this._captureCtx = this._captureCanvas.getContext('2d');
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
            });
            this.video.srcObject = this.stream;
            await this.video.play();

            // Size canvas to video
            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;

            // Connect WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/video/`);

            this.ws.onopen = () => {
                this.running = true;
                this._waitingForResponse = false;
                this.placeholder.style.display = 'none';
                this.canvas.style.display = 'block';
                this.onStatus('connected');
                // Kick off the ping-pong loop with the very first frame
                this._sendNextFrame();
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'vision_result') {
                    // Draw annotated frame returned by server
                    if (data.annotated_frame) {
                        const img = new Image();
                        img.onload = () => {
                            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                        };
                        img.src = 'data:image/jpeg;base64,' + data.annotated_frame;
                    }
                    this.onResult(data);
                }
                // Server has finished — release the gate and send the next frame.
                // The small timeout (~30ms) caps us at ~30 FPS max and gives the
                // browser a moment to paint before we capture the next frame.
                this._waitingForResponse = false;
                if (this.running) {
                    setTimeout(() => this._sendNextFrame(), 30);
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
            console.error('Vision Engine error:', err);
            this.onStatus('error');
        }
    }

    _sendNextFrame() {
        // Guard: skip if stopped, WS not open, or still waiting for a response.
        // This is the core of the ping-pong pattern — only ONE frame is ever
        // in-flight at a time, so the queue can never build up.
        if (!this.running || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this._waitingForResponse) return;

        const w = this.video.videoWidth || 640;
        const h = this.video.videoHeight || 480;
        this._captureCanvas.width = w;
        this._captureCanvas.height = h;
        this._captureCtx.drawImage(this.video, 0, 0, w, h);

        this._captureCanvas.toBlob((blob) => {
            if (blob && this.ws && this.ws.readyState === WebSocket.OPEN) {
                this._waitingForResponse = true;  // Lock the gate until server replies
                blob.arrayBuffer().then(buf => this.ws.send(buf));
            }
        }, 'image/jpeg', 0.7);
    }

    stop() {
        this.running = false;
        this._waitingForResponse = false;

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.canvas.style.display = 'none';
        this.placeholder.style.display = 'flex';
        this.onStatus('disconnected');
    }
}
