/**
 * interview.js — Orchestrates the full interview test page.
 *
 * Responsibilities:
 *  1. Auto-start VideoStream + AudioStream when the page loads.
 *  2. Forward live vision scores to the AudioConsumer for averaging.
 *  3. Update all UI elements (question, transcript, metrics, scores).
 *  4. Handle Submit Answer:
 *       a. Send {type:'submit_answer'} over the Audio WebSocket.
 *       b. Wait for {type:'final_transcript'} message back.
 *       c. Call /grade/ with the full transcript.
 *       d. Show the grading overlay with the score.
 *
 * KEY FIX: AudioStream.onResult is always called for every incoming message,
 * including final_transcript. We no longer try to re-wrap onmessage.
 */

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ─────────────────────────────────────────────────────────
    const videoCanvas       = document.getElementById('video-canvas');
    const videoPlaceholder  = document.getElementById('video-placeholder');
    const videoDot          = document.getElementById('video-status-dot');
    const audioDot          = document.getElementById('audio-status-dot');

    const scoreVal          = document.getElementById('score-val');
    const emotionVal        = document.getElementById('emotion-val');
    const facesVal          = document.getElementById('faces-val');

    const transcriptBox     = document.getElementById('transcript-box');
    const smoothnessValue   = document.getElementById('smoothness-value');
    const smoothnessFill    = document.getElementById('smoothness-fill');
    const clarityValue      = document.getElementById('clarity-value');
    const clarityFill       = document.getElementById('clarity-fill');
    const toneValue         = document.getElementById('tone-value');
    const toneFill          = document.getElementById('tone-fill');

    const btnSubmit         = document.getElementById('btn-submit');

    const gradingOverlay    = document.getElementById('grading-overlay');
    const gradingSpinner    = document.getElementById('grading-spinner');
    const gradingResult     = document.getElementById('grading-result');
    const gradingScore      = document.getElementById('grading-score');
    const gradingMsg        = document.getElementById('grading-msg');

    // ── State ─────────────────────────────────────────────────────────────
    let videoStream     = null;
    let audioStream     = null;
    let submitted       = false;
    let localTranscript = [];  // accumulate lines locally as a fallback

    // ── Helper: progress bars ─────────────────────────────────────────────
    function updateBar(valueEl, fillEl, pct) {
        const v = Math.max(0, Math.min(100, pct || 0));
        if (valueEl) valueEl.textContent = v.toFixed(0) + '%';
        if (fillEl)  fillEl.style.width  = v + '%';
    }

    function escHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function getCookie(name) {
        const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
        return match ? decodeURIComponent(match[2]) : '';
    }

    // ── Vision callbacks ──────────────────────────────────────────────────
    function onVisionResult(data) {
        if (scoreVal)   scoreVal.textContent   = (data.score || 0) + '/10';
        if (emotionVal) emotionVal.textContent = data.emotion || '—';
        if (facesVal)   facesVal.textContent   = data.faces_detected ?? '0';

        // Forward vision score to AudioConsumer for session average
        if (audioStream && audioStream.ws && audioStream.ws.readyState === WebSocket.OPEN && data.score > 0) {
            audioStream.ws.send(JSON.stringify({ type: 'vision_score', score: data.score }));
        }
    }

    function onVisionStatus(status) {
        if (videoDot) {
            videoDot.className = 'dot' + (status === 'connected' ? ' connected' : '');
            videoDot.closest('.status-pill')?.classList.toggle('live', status === 'connected');
        }
        if (videoCanvas)      videoCanvas.style.display      = status === 'connected' ? 'block' : 'none';
        if (videoPlaceholder) videoPlaceholder.style.display = status === 'connected' ? 'none'  : 'flex';
    }

    // ── Audio callbacks ───────────────────────────────────────────────────
    function onAudioResult(data) {
        if (!data || !data.type) return;

        if (data.type === 'audio_result') {
            // Update live transcript display
            const lines = data.transcript_lines || [];
            if (lines.length > 0) {
                // Keep a local copy as fallback in case final_transcript is empty
                localTranscript = lines;
                transcriptBox.innerHTML = lines.map(l => `<p>${escHtml(l)}</p>`).join('');
                transcriptBox.scrollTop = transcriptBox.scrollHeight;
            }
            // Metrics bars
            const m = data.metrics || {};
            updateBar(smoothnessValue, smoothnessFill, m.vocal_smoothness);
            updateBar(clarityValue,    clarityFill,    m.clarity);
            updateBar(toneValue,       toneFill,       m.tone_stability);
            return;
        }

        if (data.type === 'final_transcript') {
            // ✅ Backend confirmed submission — now grade it
            // If backend returns empty string, fall back to locally accumulated lines
            const transcript = data.transcript || localTranscript.join(' ');
            const avgVision  = data.avg_vision_score;
            callGradeEndpoint(transcript, avgVision);
            return;
        }

        if (data.type === 'error') {
            console.error('AudioConsumer error:', data.message);
        }
        // status messages are silently ignored
    }

    function onAudioStatus(status) {
        if (audioDot) {
            audioDot.className = 'dot' + (status === 'connected' ? ' connected' : '');
            audioDot.closest('.status-pill')?.classList.toggle('live', status === 'connected');
        }
    }

    // ── Auto-start both engines on page load ──────────────────────────────
    async function init() {
        videoStream = new VideoStream(videoCanvas, videoPlaceholder, onVisionResult, onVisionStatus);
        await videoStream.start();

        // AudioStream: onAudioResult is passed as the result callback.
        // AudioStream's internal onmessage calls this.onResult(data) for every
        // message, so final_transcript will always be handled here correctly.
        audioStream = new AudioStream(onAudioResult, onAudioStatus);
        await audioStream.start();
    }

    // ── Submit button ─────────────────────────────────────────────────────
    btnSubmit.addEventListener('click', () => {
        if (submitted) return;
        submitted = true;
        btnSubmit.disabled = true;
        btnSubmit.textContent = 'Submitting…';

        // Show spinner immediately so the user knows something is happening
        gradingOverlay.classList.add('visible');
        gradingSpinner.style.display = 'block';
        gradingResult.style.display  = 'none';

        // Stop video stream now — we no longer need the camera
        if (videoStream) videoStream.stop();

        const ws = audioStream && audioStream.ws;

        if (ws && ws.readyState === WebSocket.OPEN) {
            // Happy path: tell the AudioConsumer to freeze and return the transcript.
            // onAudioResult will receive {type:'final_transcript'} and call callGradeEndpoint.
            ws.send(JSON.stringify({ type: 'submit_answer' }));

            // Safety fallback: if we haven't received final_transcript within 8 seconds,
            // grade using whatever we accumulated locally.
            setTimeout(() => {
                if (gradingResult.style.display === 'none') {
                    console.warn('final_transcript timeout — grading with local transcript');
                    if (audioStream) audioStream.stop();
                    callGradeEndpoint(localTranscript.join(' '), null);
                }
            }, 8000);
        } else {
            // WebSocket not available — grade immediately with local transcript
            if (audioStream) audioStream.stop();
            callGradeEndpoint(localTranscript.join(' '), null);
        }
    });

    // ── Grade endpoint call ───────────────────────────────────────────────
    function callGradeEndpoint(transcript, avgVisionScore) {
        // Stop audio stream now that we have the transcript
        if (audioStream) audioStream.stop();

        const streamId = document.getElementById('stream-id').value;
        const question = document.getElementById('question-text').textContent.trim();

        console.log('Grading — stream:', streamId, '| transcript length:', transcript.length);

        fetch('/grade/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({
                stream_id:        streamId,
                question:         question,
                transcript:       transcript,
                avg_vision_score: avgVisionScore,
            }),
        })
        .then(r => {
            if (!r.ok) {
                return r.text().then(t => { throw new Error(`HTTP ${r.status}: ${t.slice(0, 200)}`); });
            }
            return r.json();
        })
        .then(data => {
            gradingSpinner.style.display = 'none';
            gradingResult.style.display  = 'block';
            gradingScore.textContent = (data.score !== undefined ? data.score : '—') + '/10';
            gradingMsg.textContent   = data.feedback || 'Interview complete!';
        })
        .catch(err => {
            console.error('Grade fetch error:', err);
            gradingSpinner.style.display = 'none';
            gradingResult.style.display  = 'block';
            gradingScore.textContent     = '—';
            gradingMsg.textContent       = 'Network error: ' + err.message;
        });
    }

    // Kick it off
    init();
});
