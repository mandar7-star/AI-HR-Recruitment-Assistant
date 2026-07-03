"""
tasks.py

Orchestrates the four-agent CrewAI pipeline sequentially:
  1. Screening Agent   -> CandidateScreen per resume
  2. Ranking Agent      -> CandidateRanking per resume
  3. Interview Agent    -> CandidateInterview for top 3 candidates
  4. Recommendation Agent -> CandidateRecommendation per resume

Design choices (per project constraints):
  - Agents run strictly SEQUENTIALLY, one Crew/Task execution at a time.
  - A 5-10 second delay is inserted between agent stages to respect
    Groq's free-tier rate limits (30 RPM / 12,000 TPM).
  - Context is NOT chained between agents as raw conversation history;
    instead, each agent receives a compact, programmatically-built
    summary of only what it needs. This prevents exponential token growth.
  - The job description is truncated to 500 characters before being
    embedded in any prompt.
  - Outputs are parsed as JSON and merged into a single FinalReport
    programmatically (not by an LLM), for reliability.
  - Rate limit errors (HTTP 429) are retried with exponential backoff.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from crewai import Crew, Task, Process

from agents import (
    interview_agent,
    ranking_agent,
    recommendation_agent,
    screening_agent,
)
from models import (
    AgentTraceStep,
    CandidateFullProfile,
    CandidateInterview,
    CandidateRanking,
    CandidateRecommendation,
    CandidateScreen,
    FinalReport,
)

logger = logging.getLogger(__name__)

INTER_AGENT_DELAY_RANGE = (5, 10)  # seconds
MAX_RETRIES = 3
JOB_DESC_MAX_CHARS = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_job_description(job_description: str) -> str:
    text = job_description.strip()
    if len(text) <= JOB_DESC_MAX_CHARS:
        return text
    return text[:JOB_DESC_MAX_CHARS].rsplit(" ", 1)[0] + "..."


def _sleep_between_agents() -> None:
    delay = random.uniform(*INTER_AGENT_DELAY_RANGE)
    logger.info("Waiting %.1fs before next agent stage to respect rate limits...", delay)
    time.sleep(delay)


def _extract_json(raw_output: str):
    """
    Extract the first valid JSON object/array from an LLM's raw text output,
    tolerating markdown code fences and surrounding commentary.
    """
    if raw_output is None:
        raise ValueError("Empty output from agent.")

    text = raw_output.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the widest {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from agent output: {text[:300]}")


def _run_crew_with_retry(crew: Crew, stage_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Kick off a single-task Crew, retrying on rate-limit / transient errors
    with exponential backoff. Returns (raw_output, error_message).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = crew.kickoff()
            raw = str(result)
            return raw, None
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            is_rate_limit = "429" in message or "rate" in message.lower()
            if attempt < MAX_RETRIES and is_rate_limit:
                backoff = (2 ** attempt) + random.uniform(0, 2)
                logger.warning(
                    "[%s] Rate limit hit (attempt %d/%d). Backing off %.1fs.",
                    stage_name, attempt, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                continue
            logger.exception("[%s] Agent stage failed on attempt %d.", stage_name, attempt)
            return None, message
    return None, "Max retries exceeded."


# ---------------------------------------------------------------------------
# Stage 1: Screening
# ---------------------------------------------------------------------------

def run_screening_stage(
    job_description: str, resumes: Dict[str, str]
) -> Tuple[List[CandidateScreen], AgentTraceStep]:
    started = time.time()
    jd_excerpt = _truncate_job_description(job_description)

    resume_list_text = "\n\n".join(
        f"File: {name}\n{text[:3000]}" for name, text in resumes.items()
    )

    task = Task(
        description=(
            f"Job description (excerpt): {jd_excerpt}\n\n"
            f"Below are {len(resumes)} candidate resumes. For EACH resume, use the "
            "Resume Retrieval Tool if needed to confirm details, then extract a "
            "structured profile.\n\n"
            f"{resume_list_text}\n\n"
            "Return ONLY a JSON array, one object per candidate, with EXACTLY these "
            "keys: candidate_name, file_name, matched_skills (list), missing_skills "
            "(list), certifications (list), awards (list), notable_projects (list), "
            "experience_summary (string), education_summary (string), "
            "overall_impression (string). Do not include any text outside the JSON array."
        ),
        expected_output="A JSON array of candidate screening profiles.",
        agent=screening_agent,
    )
    crew = Crew(agents=[screening_agent], tasks=[task], process=Process.sequential, verbose=True)

    raw, error = _run_crew_with_retry(crew, "Screening")
    duration = time.time() - started

    if error:
        return [], AgentTraceStep(
            agent_name="Resume Screening Agent", status="failed",
            duration_seconds=duration, message=error,
        )

    try:
        data = _extract_json(raw)
        screens = [CandidateScreen(**item) for item in data]
        return screens, AgentTraceStep(
            agent_name="Resume Screening Agent", status="completed",
            duration_seconds=duration,
            message=f"Screened {len(screens)} candidate(s).",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse screening output.")
        return [], AgentTraceStep(
            agent_name="Resume Screening Agent", status="failed",
            duration_seconds=duration, message=f"Parse error: {exc}",
        )


# ---------------------------------------------------------------------------
# Stage 2: Ranking
# ---------------------------------------------------------------------------

def run_ranking_stage(
    job_description: str, screens: List[CandidateScreen]
) -> Tuple[List[CandidateRanking], AgentTraceStep]:
    started = time.time()
    jd_excerpt = _truncate_job_description(job_description)

    profiles_json = json.dumps([s.model_dump() for s in screens], indent=None)

    task = Task(
        description=(
            f"Job description (excerpt): {jd_excerpt}\n\n"
            f"Candidate screening profiles:\n{profiles_json}\n\n"
            "Score each candidate 0-100 based only on job fit, then rank them "
            "(1 = best). For each candidate, write a brief justification, a "
            "confidence_level ('Low', 'Medium', or 'High'), and a fairness_note "
            "explicitly confirming your score was based only on job-relevant "
            "qualifications and not on name, gender, age, or ethnicity.\n\n"
            "Return ONLY a JSON array with EXACTLY these keys per object: "
            "candidate_name, file_name, score (number), rank (integer), "
            "justification (string), confidence_level (string), fairness_note (string)."
        ),
        expected_output="A JSON array of candidate rankings sorted best to worst.",
        agent=ranking_agent,
    )
    crew = Crew(agents=[ranking_agent], tasks=[task], process=Process.sequential, verbose=True)

    raw, error = _run_crew_with_retry(crew, "Ranking")
    duration = time.time() - started

    if error:
        return [], AgentTraceStep(
            agent_name="Candidate Ranking Agent", status="failed",
            duration_seconds=duration, message=error,
        )

    try:
        data = _extract_json(raw)
        rankings = [CandidateRanking(**item) for item in data]
        rankings.sort(key=lambda r: r.rank)
        return rankings, AgentTraceStep(
            agent_name="Candidate Ranking Agent", status="completed",
            duration_seconds=duration,
            message=f"Ranked {len(rankings)} candidate(s).",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse ranking output.")
        return [], AgentTraceStep(
            agent_name="Candidate Ranking Agent", status="failed",
            duration_seconds=duration, message=f"Parse error: {exc}",
        )


# ---------------------------------------------------------------------------
# Stage 3: Interview Questions (top 3 only)
# ---------------------------------------------------------------------------

def run_interview_stage(
    job_description: str,
    rankings: List[CandidateRanking],
    screens_by_file: Dict[str, CandidateScreen],
) -> Tuple[List[CandidateInterview], AgentTraceStep]:
    started = time.time()
    jd_excerpt = _truncate_job_description(job_description)

    top_candidates = rankings[:3]
    if not top_candidates:
        return [], AgentTraceStep(
            agent_name="Interview Question Generator", status="skipped",
            duration_seconds=0.0, message="No ranked candidates available.",
        )

    context_items = []
    for r in top_candidates:
        screen = screens_by_file.get(r.file_name)
        context_items.append({
            "candidate_name": r.candidate_name,
            "file_name": r.file_name,
            "matched_skills": screen.matched_skills if screen else [],
            "missing_skills": screen.missing_skills if screen else [],
            "experience_summary": screen.experience_summary if screen else "",
        })

    task = Task(
        description=(
            f"Job description (excerpt): {jd_excerpt}\n\n"
            f"Top candidates:\n{json.dumps(context_items)}\n\n"
            "For EACH top candidate, write 5 to 7 interview questions mixing "
            "technical, behavioral, and situational types. Questions must be "
            "specific to both the role and the candidate's background "
            "(e.g. probe missing skills, validate matched skills).\n\n"
            "Return ONLY a JSON array with EXACTLY these keys per object: "
            "candidate_name, file_name, questions (list of strings)."
        ),
        expected_output="A JSON array of interview question sets for the top candidates.",
        agent=interview_agent,
    )
    crew = Crew(agents=[interview_agent], tasks=[task], process=Process.sequential, verbose=True)

    raw, error = _run_crew_with_retry(crew, "Interview")
    duration = time.time() - started

    if error:
        return [], AgentTraceStep(
            agent_name="Interview Question Generator", status="failed",
            duration_seconds=duration, message=error,
        )

    try:
        data = _extract_json(raw)
        interviews = [CandidateInterview(**item) for item in data]
        return interviews, AgentTraceStep(
            agent_name="Interview Question Generator", status="completed",
            duration_seconds=duration,
            message=f"Generated questions for {len(interviews)} top candidate(s).",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse interview output.")
        return [], AgentTraceStep(
            agent_name="Interview Question Generator", status="failed",
            duration_seconds=duration, message=f"Parse error: {exc}",
        )


# ---------------------------------------------------------------------------
# Stage 4: Hiring Recommendation
# ---------------------------------------------------------------------------

def run_recommendation_stage(
    job_description: str, rankings: List[CandidateRanking]
) -> Tuple[List[CandidateRecommendation], AgentTraceStep]:
    started = time.time()
    jd_excerpt = _truncate_job_description(job_description)

    rankings_json = json.dumps([r.model_dump() for r in rankings])

    task = Task(
        description=(
            f"Job description (excerpt): {jd_excerpt}\n\n"
            f"Candidate rankings:\n{rankings_json}\n\n"
            "For EACH candidate, issue a final hiring verdict: one of "
            "'Strong Hire', 'Hire', 'Maybe', or 'No Hire'. Provide a concise "
            "summary, key_strengths (list), and key_risks (list).\n\n"
            "Return ONLY a JSON array with EXACTLY these keys per object: "
            "candidate_name, file_name, verdict, summary, key_strengths (list), "
            "key_risks (list)."
        ),
        expected_output="A JSON array of hiring recommendations for all candidates.",
        agent=recommendation_agent,
    )
    crew = Crew(agents=[recommendation_agent], tasks=[task], process=Process.sequential, verbose=True)

    raw, error = _run_crew_with_retry(crew, "Recommendation")
    duration = time.time() - started

    if error:
        return [], AgentTraceStep(
            agent_name="Hiring Recommendation Agent", status="failed",
            duration_seconds=duration, message=error,
        )

    try:
        data = _extract_json(raw)
        recs = [CandidateRecommendation(**item) for item in data]
        return recs, AgentTraceStep(
            agent_name="Hiring Recommendation Agent", status="completed",
            duration_seconds=duration,
            message=f"Produced recommendations for {len(recs)} candidate(s).",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse recommendation output.")
        return [], AgentTraceStep(
            agent_name="Hiring Recommendation Agent", status="failed",
            duration_seconds=duration, message=f"Parse error: {exc}",
        )


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    job_description: str, resumes: Dict[str, str], skipped_files: Optional[List[str]] = None
) -> FinalReport:
    """
    Run the full sequential 4-agent pipeline and merge results into a
    single FinalReport.

    Args:
        job_description: raw job description text.
        resumes: mapping of {file_name: extracted_resume_text}.
        skipped_files: file names that failed to parse before reaching this stage.

    Returns:
        FinalReport with merged candidate profiles and an agent execution trace.
    """
    trace: List[AgentTraceStep] = []
    errors: List[str] = []

    # --- Stage 1: Screening ---
    screens, step1 = run_screening_stage(job_description, resumes)
    trace.append(step1)
    if step1.status == "failed":
        errors.append(f"Screening stage failed: {step1.message}")
    screens_by_file = {s.file_name: s for s in screens}

    if screens:
        _sleep_between_agents()

    # --- Stage 2: Ranking ---
    rankings, step2 = run_ranking_stage(job_description, screens) if screens else (
        [], AgentTraceStep(agent_name="Candidate Ranking Agent", status="skipped",
                            message="No screened candidates to rank.")
    )
    trace.append(step2)
    if step2.status == "failed":
        errors.append(f"Ranking stage failed: {step2.message}")

    if rankings:
        _sleep_between_agents()

    # --- Stage 3: Interview Questions (top 3) ---
    interviews, step3 = run_interview_stage(job_description, rankings, screens_by_file) if rankings else (
        [], AgentTraceStep(agent_name="Interview Question Generator", status="skipped",
                            message="No rankings available.")
    )
    trace.append(step3)
    if step3.status == "failed":
        errors.append(f"Interview stage failed: {step3.message}")
    interviews_by_file = {i.file_name: i for i in interviews}

    if rankings:
        _sleep_between_agents()

    # --- Stage 4: Hiring Recommendation ---
    recommendations, step4 = run_recommendation_stage(job_description, rankings) if rankings else (
        [], AgentTraceStep(agent_name="Hiring Recommendation Agent", status="skipped",
                            message="No rankings available.")
    )
    trace.append(step4)
    if step4.status == "failed":
        errors.append(f"Recommendation stage failed: {step4.message}")
    recs_by_file = {r.file_name: r for r in recommendations}

    # --- Merge everything programmatically ---
    rankings_by_file = {r.file_name: r for r in rankings}
    all_file_names = set(screens_by_file) | set(rankings_by_file) | set(recs_by_file)

    profiles: List[CandidateFullProfile] = []
    for file_name in all_file_names:
        screen = screens_by_file.get(file_name)
        ranking = rankings_by_file.get(file_name)
        interview = interviews_by_file.get(file_name)
        rec = recs_by_file.get(file_name)

        name = (
            (screen.candidate_name if screen else None)
            or (ranking.candidate_name if ranking else None)
            or (rec.candidate_name if rec else None)
            or file_name
        )

        profiles.append(CandidateFullProfile(
            candidate_name=name,
            file_name=file_name,
            matched_skills=screen.matched_skills if screen else [],
            missing_skills=screen.missing_skills if screen else [],
            certifications=screen.certifications if screen else [],
            awards=screen.awards if screen else [],
            notable_projects=screen.notable_projects if screen else [],
            experience_summary=screen.experience_summary if screen else "",
            education_summary=screen.education_summary if screen else "",
            overall_impression=screen.overall_impression if screen else "",
            score=ranking.score if ranking else None,
            rank=ranking.rank if ranking else None,
            ranking_justification=ranking.justification if ranking else "",
            confidence_level=ranking.confidence_level if ranking else "",
            fairness_note=ranking.fairness_note if ranking else "",
            interview_questions=interview.questions if interview else [],
            verdict=rec.verdict if rec else None,
            recommendation_summary=rec.summary if rec else "",
            key_strengths=rec.key_strengths if rec else [],
            key_risks=rec.key_risks if rec else [],
        ))

    profiles.sort(key=lambda p: (p.rank is None, p.rank if p.rank is not None else 0))

    return FinalReport(
        job_description_excerpt=_truncate_job_description(job_description),
        candidates_processed=len(resumes),
        candidates=profiles,
        agent_trace=trace,
        skipped_files=skipped_files or [],
        errors=errors,
    )