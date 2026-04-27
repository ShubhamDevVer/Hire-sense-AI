"""HTTP URL patterns for the coach app."""

from django.urls import path

from . import views

urlpatterns = [
    # Authentication
    path("register/", views.register_view,      name="register"),
    path("login/",    views.login_view,          name="login"),
    path("logout/",   views.logout_view,         name="logout"),

    # Candidate portal
    path("candidate/", views.candidate_dashboard, name="candidate_dashboard"),

    # Interview test page
    path("interview/<int:stream_id>/", views.interview_view,   name="interview"),

    # LLM pipeline
    path("generate-question/",         views.generate_question, name="generate_question"),
    path("grade/",                     views.grade_answer,      name="grade_answer"),

    # Real-time coach dashboard (Vision + Audio engines)
    path("coach/",   views.dashboard,            name="dashboard"),

    # Root → redirect to candidate dashboard
    path("",         lambda r: __import__("django.shortcuts", fromlist=["redirect"]).redirect("candidate_dashboard"), name="home"),
]
