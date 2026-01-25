from django.urls import path
from .views import (
    home_view,
    explain_view,
    summarize_view,
    quiz_view,
    flashcards_view,
)

urlpatterns = [
    path("", home_view, name="home"),

    path("explain/", explain_view, name="explain"),
    path("summarize/", summarize_view, name="summarize"),

    path("quiz/", quiz_view, name="quiz"),
    path("flashcards/", flashcards_view, name="flashcards"),
]
