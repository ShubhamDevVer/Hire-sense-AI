/**
 * Dashboard Orchestrator — Wires up VideoStream and AudioStream to the UI.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── DOM References ──────────────────────────────────────────────────
    const videoCanvas      = document.getElementById('video-canvas');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const btnVideo         = document.getElementById('btn-video');
    const btnAudio         = document.getElementById('btn-audio');
    const btnClear         = document.getElementById('btn-clear');

    // Vision metrics
    const scoreValue       = document.getElementById('score-value');
    const emotionValue     = document.getElementById('emotion-value');
    const emotionProb      = document.getElementById('emotion-prob');
    const facesValue       = document.getElementById('faces-value');

    // Audio metrics
    const smoothnessValue  = document.getElementById('smoothness-value');
    const smoothnessFill   = document.getElementById('smoothness-fill');
    const clarityValue     = document.getElementById('clarity-value');
    const clarityFill      = document.getElementById('clarity-fill');
    const toneValue        = document.getElementById('tone-value');
    const toneFill         = document.getElementById('tone-fill');

    // Transcript
    const transcriptBox    = document.getElementById('transcript-box');

    // Details
    const audioDetails     = document.getElementById('audio-details');
    const errorBanner      = document.getElementById('error-banner');

    // Status dots
    const videoDot         = document.getElementById('video-status-dot');
    const audioDot         = document.getElementById('audio-status-dot');

    // ── State ───────────────────────────────────────────────────────────
    let videoStream = null;
    let audioStream = null;
    let videoActive = false;
    let audioActive = false;

    // ── Score color helper ──────────────────────────────────────────────
    function scoreClass(score) {
        if (score >= 8) return 'high';
        if (score >= 5) return 'mid';
        return 'low';
    }

    // ── Vision callbacks ────────────────────────────────────────────────
    function onVisionResult(data) {
        scoreValue.textContent = data.score + '/10';
        scoreValue.className = 'score-card__value ' + scoreClass(data.score);
        emotionValue.textContent = data.emotion;
        emotionProb.textContent = (data.probability * 100).toFixed(1) + '%';
        facesValue.textContent = data.faces_detected;
    }

    function onVisionStatus(status) {
        videoDot.className = 'status-dot' + (status === 'connected' ? ' connected' : '');
    }

    // ── Audio callbacks ─────────────────────────────────────────────────
    function onAudioResult(data) {
        if (data.type === 'error') {
            showError(data.message);
            return;
        }

        if (data.type === 'status') return;

        if (data.type !== 'audio_result') return;

        // Metrics bars
        const m = data.metrics || {};
        updateBar(smoothnessValue, smoothnessFill, m.vocal_smoothness || 0);
        updateBar(clarityValue, clarityFill, m.clarity || 0);
        updateBar(toneValue, toneFill, m.tone_stability || 0);

        // Transcript
        const lines = data.transcript_lines || [];
        if (lines.length > 0) {
            transcriptBox.innerHTML = lines.map(l => `<p>${escapeHtml(l)}</p>`).join('');
            transcriptBox.scrollTop = transcriptBox.scrollHeight;
        }

        // Details
        const t = data.tone || {};
        const q = data.audio_quality || {};
        const tr = data.transcription || {};
        audioDetails.innerHTML = `
            <strong>Status:</strong> ${tr.gate_reason || 'Waiting...'}<br>
            <strong>Filler Words:</strong> ${tr.filler_total || 0} total, ${(tr.filler_per_minute || 0).toFixed(1)}/min<br>
            <strong>Dead Air:</strong> ${(t.dead_air_pct || 0).toFixed(1)}%<br>
            <strong>Median F0:</strong> ${(t.median_f0_hz || 0).toFixed(1)} Hz<br>
            <strong>Pitch Stability:</strong> ${(t.pitch_stability || 0).toFixed(1)}%<br>
            <strong>Audio Level:</strong> RMS ${(q.rms || 0).toFixed(4)}, Peak ${(q.peak || 0).toFixed(4)}<br>
            <strong>Chunks:</strong> ${data.chunks_processed || 0} processed, ${data.chunks_skipped || 0} skipped, ${data.groq_requests || 0} Groq calls<br>
            <strong>Buffered:</strong> ${(data.buffered_seconds || 0).toFixed(1)}s
        `;

        hideError();
    }

    function onAudioStatus(status) {
        audioDot.className = 'status-dot' + (status === 'connected' ? ' connected' : '');
    }

    // ── UI Helpers ──────────────────────────────────────────────────────
    function updateBar(valueEl, fillEl, pct) {
        const clamped = Math.max(0, Math.min(100, pct));
        valueEl.textContent = clamped.toFixed(0) + '%';
        fillEl.style.width = clamped + '%';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function showError(msg) {
        errorBanner.textContent = msg;
        errorBanner.classList.add('visible');
    }

    function hideError() {
        errorBanner.classList.remove('visible');
    }

    // ── Button Handlers ─────────────────────────────────────────────────
    btnVideo.addEventListener('click', () => {
        if (!videoActive) {
            videoStream = new VideoStream(videoCanvas, videoPlaceholder, onVisionResult, onVisionStatus);
            videoStream.start();
            videoActive = true;
            btnVideo.classList.add('active');
            btnVideo.querySelector('.btn__label').textContent = 'Stop Camera';
        } else {
            if (videoStream) videoStream.stop();
            videoActive = false;
            btnVideo.classList.remove('active');
            btnVideo.querySelector('.btn__label').textContent = 'Start Camera';
            scoreValue.textContent = '0/10';
            scoreValue.className = 'score-card__value low';
            emotionValue.textContent = 'Waiting';
            emotionProb.textContent = '-';
            facesValue.textContent = '0';
        }
    });

    btnAudio.addEventListener('click', () => {
        if (!audioActive) {
            audioStream = new AudioStream(onAudioResult, onAudioStatus);
            audioStream.start();
            audioActive = true;
            btnAudio.classList.add('active');
            btnAudio.querySelector('.btn__label').textContent = 'Stop Microphone';
        } else {
            if (audioStream) audioStream.stop();
            audioActive = false;
            btnAudio.classList.remove('active');
            btnAudio.querySelector('.btn__label').textContent = 'Start Microphone';
        }
    });

    btnClear.addEventListener('click', () => {
        transcriptBox.innerHTML = '<p class="transcript-placeholder">Listening... transcript will appear as you speak.</p>';
        updateBar(smoothnessValue, smoothnessFill, 0);
        updateBar(clarityValue, clarityFill, 0);
        updateBar(toneValue, toneFill, 0);
        audioDetails.innerHTML = '<strong>Status:</strong> Cleared.';
        hideError();
    });
});
