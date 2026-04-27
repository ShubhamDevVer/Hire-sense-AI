"""Views for Hire Sense AI — authentication + dashboards + LLM pipeline."""

import json
import logging
import re

import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from .models import InterviewResult, TestStream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared OpenRouter helper
# ---------------------------------------------------------------------------

def _call_openrouter(system_prompt: str, user_content: str) -> str:
    """
    Send a single chat completion request to OpenRouter's API.
    Returns the model's response text.
    Raises ValueError on non-200 responses.
    """
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hiresenseai.local",  # required by OpenRouter
        "X-Title": "Hire Sense AI",
    }
    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    response = requests.post(
        settings.OPENROUTER_BASE_URL,
        headers=headers,
        json=payload,
        timeout=30,
    )
    if response.status_code != 200:
        raise ValueError(
            f"OpenRouter error {response.status_code}: {response.text[:300]}"
        )
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def register_view(request):
    """New candidate registration."""
    if request.user.is_authenticated:
        return redirect("candidate_dashboard")

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Welcome to Hire Sense AI, {user.username}!")
            return redirect("candidate_dashboard")
    else:
        form = UserCreationForm()

    return render(request, "coach/register.html", {"form": form})


def login_view(request):
    """Candidate login."""
    if request.user.is_authenticated:
        return redirect("candidate_dashboard")

    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            next_url = request.GET.get("next", "candidate_dashboard")
            return redirect(next_url)
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    return render(request, "coach/login.html", {"form": form})


def logout_view(request):
    """Log out and return to login page."""
    logout(request)
    return redirect("login")


# ---------------------------------------------------------------------------
# Candidate Dashboard
# ---------------------------------------------------------------------------

@login_required
def candidate_dashboard(request):
    """
    Main candidate landing page.
    Left sidebar: profile + past InterviewResult history.
    Main area: TestStream grid for choosing a new interview.
    """
    past_results = (
        InterviewResult.objects
        .filter(user=request.user)
        .select_related("stream")
        .order_by("-completed_at")[:20]
    )
    streams = TestStream.objects.filter(is_active=True)

    context = {
        "past_results": past_results,
        "streams": streams,
        "result_count": past_results.count(),
        "best_score": max(
            (r.llm_score for r in past_results if r.llm_score is not None),
            default=None,
        ),
    }
    return render(request, "coach/candidate_dashboard.html", context)


# ---------------------------------------------------------------------------
# Interview Test Page
# ---------------------------------------------------------------------------

@login_required
def interview_view(request, stream_id):
    """
    Renders the interview test page for a given TestStream.
    The LLM question is fetched client-side via /generate-question/.
    """
    stream = get_object_or_404(TestStream, id=stream_id, is_active=True)
    return render(request, "coach/interview_test.html", {"stream": stream})


# ---------------------------------------------------------------------------
# LLM Pipeline — Question Generation
# ---------------------------------------------------------------------------

@login_required
@require_GET
def generate_question(request):
    """
    GET /generate-question/?stream_id=<id>

    Calls the Minimax M2 model on OpenRouter to generate one fresh,
    intermediate-level interview question for the requested stream.

    Returns: { "question": "..." }
    On OpenRouter failure returns a sensible fallback question so the
    interview page never gets stuck.
    """
    stream_id = request.GET.get("stream_id")
    if not stream_id:
        return JsonResponse({"error": "stream_id is required."}, status=400)

    stream = get_object_or_404(TestStream, id=stream_id, is_active=True)

    system_prompt = (
        "You are an expert technical interviewer with years of industry experience. "
        "Generate a single, challenging, intermediate-level interview question "
        "for the topic provided by the user. "
        "Output ONLY the question text. "
        "Do not include greetings, numbering, labels, or any extra formatting."
    )
    user_content = f"Topic: {stream.name}"

    try:
        question = _call_openrouter(system_prompt, user_content)
        return JsonResponse({"question": question})
    except Exception as exc:
        logger.exception("OpenRouter question generation failed: %s", exc)
        fallback = (
            f"Explain a key concept in {stream.name} that you consider "
            "most important in a professional setting."
        )
        return JsonResponse({"question": fallback, "fallback": True})


# ---------------------------------------------------------------------------
# LLM Pipeline — Answer Grading
# ---------------------------------------------------------------------------

@login_required
@require_POST
def grade_answer(request):
    """
    POST /grade/
    Body (JSON): { stream_id, question, transcript, avg_vision_score }

    1. Sends the question + transcript to Minimax M2 on OpenRouter for scoring.
    2. Parses the integer score (1-10) from the response.
    3. Saves an InterviewResult row to the DB.
    4. Returns: { "score": <int 1-10>, "feedback": "<message>" }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    stream_id        = body.get("stream_id")
    question         = body.get("question", "").strip()
    transcript       = body.get("transcript", "").strip()
    avg_vision_score = body.get("avg_vision_score")  # float or None

    if not stream_id:
        return JsonResponse({"error": "stream_id is required."}, status=400)

    stream = get_object_or_404(TestStream, id=stream_id, is_active=True)

    # ── Grade via OpenRouter ──────────────────────────────────────────────
    # Always call OpenRouter — even on empty transcript.
    # If silent, the model will naturally return a very low score (1-2).
    llm_score = None

    system_prompt = (
        "You are a strict but fair technical interviewer. "
        "Evaluate the candidate's spoken answer based solely on its "
        "technical accuracy and relevance to the question. "
        "Ignore filler words (um, uh, like) and speech disfluencies. "
        "If the answer is blank or empty, give a score of 1. "
        "Output ONLY a single integer from 1 to 10. "
        "No explanations, no punctuation, just the number."
    )
    candidate_answer = transcript if transcript else "(No answer provided — the candidate did not speak.)"
    user_content = (
        f"Interview question: {question}\n\n"
        f"Candidate's answer: {candidate_answer}"
    )
    try:
        raw = _call_openrouter(system_prompt, user_content)
        logger.info("OpenRouter grading raw response: %r", raw[:100])
        match = re.search(r"\b(10|[1-9])\b", raw)
        if match:
            llm_score = int(match.group(1))
        else:
            logger.warning("Could not parse score from: %r", raw[:100])
            llm_score = 1  # unparseable response → minimum score
    except Exception as exc:
        logger.exception("OpenRouter grading failed: %s", exc)
        llm_score = 1  # API failure → always give a score, never None

    # ── Persist to DB ─────────────────────────────────────────────────────
    try:
        vision_float = float(avg_vision_score) if avg_vision_score is not None else None
    except (TypeError, ValueError):
        vision_float = None

    InterviewResult.objects.create(
        user=request.user,
        stream=stream,
        generated_question=question,
        candidate_transcript=transcript,
        llm_score=llm_score,
        vision_confidence_score=vision_float,
    )

    # ── Build feedback message ────────────────────────────────────────────
    if llm_score is None:
        feedback = "Grading could not be completed. Your session has been recorded."
    elif llm_score >= 8:
        feedback = "Excellent answer! You demonstrated strong technical knowledge."
    elif llm_score >= 6:
        feedback = "Good effort! A few areas could be strengthened with more depth."
    elif llm_score >= 4:
        feedback = "Decent attempt. Consider revisiting core concepts in this area."
    else:
        feedback = "Keep practicing! Review the fundamentals and try again."

    return JsonResponse({
        "score":    llm_score if llm_score is not None else "—",
        "feedback": feedback,
    })


# ---------------------------------------------------------------------------
# Real-time coach dashboard (Vision + Audio engines page)
# ---------------------------------------------------------------------------

@login_required
def dashboard(request):
    """Render the real-time interview coach dashboard."""
    return render(request, "coach/dashboard.html")
