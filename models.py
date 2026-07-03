"""
models.py

Pydantic data models used throughout the AI HR Recruitment Assistant.
These models define the shape of data passed between agents, tasks,
the FastAPI backend, and the Streamlit frontend.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent 1: Resume Screening
# ---------------------------------------------------------------------------

class CandidateScreen(BaseModel):
    """Structured screening output for a single candidate."""

    candidate_name: str = Field(..., description="Candidate's full name or filename if unknown")
    file_name: str = Field(..., description="Original resume file name")
    matched_skills: List[str] = Field(default_factory=list, description="Skills that match the job description")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills not found in resume")
    certifications: List[str] = Field(default_factory=list, description="Certifications held by the candidate")
    awards: List[str] = Field(default_factory=list, description="Awards or honors received")
    notable_projects: List[str] = Field(default_factory=list, description="Notable projects mentioned")
    experience_summary: str = Field(default="", description="Summary of relevant work experience")
    education_summary: str = Field(default="", description="Summary of education background")
    overall_impression: str = Field(default="", description="Screening agent's overall impression")


class ScreeningReport(BaseModel):
    """Collection of all candidate screening results."""

    candidates: List[CandidateScreen] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 2: Candidate Ranking
# ---------------------------------------------------------------------------

class CandidateRanking(BaseModel):
    """Ranking output for a single candidate."""

    candidate_name: str
    file_name: str
    score: float = Field(..., ge=0, le=100, description="Overall candidate score out of 100")
    rank: int = Field(..., ge=1, description="Rank position, 1 = best")
    justification: str = Field(default="", description="Explanation for the assigned score/rank")
    confidence_level: str = Field(default="Medium", description="Low / Medium / High confidence in the ranking")
    fairness_note: str = Field(
        default="", description="Self-review note confirming the ranking was made without bias"
    )


class RankingReport(BaseModel):
    """Collection of all candidate rankings, sorted best to worst."""

    rankings: List[CandidateRanking] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 3: Interview Question Generation
# ---------------------------------------------------------------------------

class CandidateInterview(BaseModel):
    """Interview questions generated for a single (top) candidate."""

    candidate_name: str
    file_name: str
    questions: List[str] = Field(default_factory=list, description="5-7 tailored interview questions")


class InterviewReport(BaseModel):
    """Collection of interview question sets for top candidates."""

    interviews: List[CandidateInterview] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 4: Hiring Recommendation
# ---------------------------------------------------------------------------

class CandidateRecommendation(BaseModel):
    """Final hiring recommendation for a single candidate."""

    candidate_name: str
    file_name: str
    verdict: str = Field(..., description="Strong Hire / Hire / Maybe / No Hire")
    summary: str = Field(default="", description="Short summary supporting the verdict")
    key_strengths: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)


class RecommendationReport(BaseModel):
    """Collection of hiring recommendations for all candidates."""

    recommendations: List[CandidateRecommendation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Merged / Final Output
# ---------------------------------------------------------------------------

class CandidateFullProfile(BaseModel):
    """Fully merged view of a single candidate across all four agents."""

    candidate_name: str
    file_name: str

    # From screening
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)
    notable_projects: List[str] = Field(default_factory=list)
    experience_summary: str = ""
    education_summary: str = ""
    overall_impression: str = ""

    # From ranking
    score: Optional[float] = None
    rank: Optional[int] = None
    ranking_justification: str = ""
    confidence_level: str = ""
    fairness_note: str = ""

    # From interview generation (only present for top candidates)
    interview_questions: List[str] = Field(default_factory=list)

    # From recommendation
    verdict: Optional[str] = None
    recommendation_summary: str = ""
    key_strengths: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)


class AgentTraceStep(BaseModel):
    """A single step in the agent execution trace, shown in the UI."""

    agent_name: str
    status: str = Field(default="completed", description="completed / failed / skipped")
    duration_seconds: Optional[float] = None
    message: str = ""


class FinalReport(BaseModel):
    """Top-level report returned by the /analyze endpoint."""

    job_description_excerpt: str = ""
    candidates_processed: int = 0
    candidates: List[CandidateFullProfile] = Field(default_factory=list)
    agent_trace: List[AgentTraceStep] = Field(default_factory=list)
    skipped_files: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)