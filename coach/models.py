"""
Database models for Hire Sense AI.

TestStream  — represents a topic category (Python, SQL, etc.)
InterviewResult — one completed interview session per candidate.
"""

from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


class TestStream(models.Model):
    """A topic/domain the candidate can be interviewed on."""

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(
        max_length=10,
        blank=True,
        default="💻",
        help_text="Single emoji shown on the dashboard card.",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class InterviewResult(models.Model):
    """One completed interview session stored against a user."""

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="interview_results",
    )
    stream = models.ForeignKey(
        TestStream,
        on_delete=models.SET_NULL,
        null=True,
        related_name="results",
    )

    # LLM-generated question shown to the candidate
    generated_question = models.TextField(blank=True)

    # Whisper transcript of the candidate's spoken answer
    candidate_transcript = models.TextField(blank=True)

    # Score given by the grading LLM (1–10)
    llm_score = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(10)],
    )

    # Average emotion confidence score from the Vision Engine (0.0–10.0)
    vision_confidence_score = models.FloatField(null=True, blank=True)

    # When the interview was completed
    completed_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-completed_at"]

    def __str__(self):
        stream_name = self.stream.name if self.stream else "Unknown"
        return f"{self.user.username} — {stream_name} — {self.completed_at:%Y-%m-%d %H:%M}"

    @property
    def overall_score(self):
        """
        Blended score shown on the dashboard.
        70% LLM answer quality, 30% vision confidence.
        Returns None if either component is missing.
        """
        if self.llm_score is None or self.vision_confidence_score is None:
            return self.llm_score  # Fall back to LLM score alone
        return round(self.llm_score * 0.7 + (self.vision_confidence_score) * 0.3, 1)
