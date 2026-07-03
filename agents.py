"""
agents.py

Defines the four CrewAI agents that make up the HR Recruitment Assistant
pipeline, all backed by Groq's Llama 3.3 70B model via LiteLLM.

IMPORTANT — Groq / CrewAI compatibility patch
----------------------------------------------
CrewAI's internal LLM caching layer (crewai.llms.cache) may attach a
`cache_breakpoint` field to outgoing requests. Groq's API does not
support this field and will reject requests that include it. The patch
below strips `cache_breakpoint` from any params dict before it reaches
the Groq/LiteLLM call, restoring compatibility without disabling CrewAI's
caching behaviour outright.
"""

from __future__ import annotations

import logging
import os

from crewai import Agent, LLM
from dotenv import load_dotenv

from tools import resume_retrieval_tool

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Critical patch: remove 'cache_breakpoint' from CrewAI's LLM cache layer
# since Groq's API rejects unknown fields like this one.
# ---------------------------------------------------------------------------

def _patch_crewai_cache_breakpoint() -> None:
    """
    Monkey-patch crewai.llms.cache so that any dict/object it produces
    never carries a 'cache_breakpoint' key/attribute, which Groq's API
    does not accept.
    """
    try:
        import crewai.llms.cache as crewai_cache
    except ImportError:
        logger.warning("crewai.llms.cache module not found; skipping cache_breakpoint patch.")
        return

    # Case 1: module exposes a function that builds cache control params.
    for attr_name in dir(crewai_cache):
        attr = getattr(crewai_cache, attr_name, None)
        if not callable(attr):
            continue

        original_fn = attr

        def _make_wrapper(fn):
            def _wrapped(*args, **kwargs):
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and "cache_breakpoint" in result:
                    result = {k: v for k, v in result.items() if k != "cache_breakpoint"}
                elif hasattr(result, "cache_breakpoint"):
                    try:
                        delattr(result, "cache_breakpoint")
                    except AttributeError:
                        pass
                return result
            return _wrapped

        try:
            setattr(crewai_cache, attr_name, _make_wrapper(original_fn))
        except (AttributeError, TypeError):
            # Some attributes (classes, constants) can't/shouldn't be wrapped.
            continue

    logger.info("Applied CrewAI cache_breakpoint compatibility patch for Groq.")


_patch_crewai_cache_breakpoint()


# ---------------------------------------------------------------------------
# Groq LLM configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY is not set. Set it in your .env file before running the pipeline."
    )


def build_groq_llm() -> LLM:
    """Construct the shared Groq-backed LLM used by all four agents."""
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=2000,
        max_retries=2,
    )


groq_llm = build_groq_llm()


# ---------------------------------------------------------------------------
# Agent 1: Resume Screening Agent
# ---------------------------------------------------------------------------

screening_agent = Agent(
    role="Resume Screening Specialist",
    goal=(
        "Carefully analyze each candidate's resume against the job description "
        "and extract a structured, accurate profile covering matched and missing "
        "skills, certifications, awards, notable projects, experience, and education."
    ),
    backstory=(
        "You are a meticulous technical recruiter with over a decade of experience "
        "screening resumes for engineering and business roles. You read resumes "
        "closely, avoid assumptions not supported by the text, and always ground "
        "your findings in what is actually written in the candidate's resume."
    ),
    tools=[resume_retrieval_tool],
    llm=groq_llm,
    allow_delegation=False,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Agent 2: Candidate Ranking Agent
# ---------------------------------------------------------------------------

ranking_agent = Agent(
    role="Candidate Ranking Analyst",
    goal=(
        "Objectively score and rank candidates from 0-100 based solely on how "
        "well their qualifications match the job requirements. Provide a clear "
        "justification for every score and perform an explicit fairness self-review "
        "to ensure the ranking is free of bias related to name, gender, age, "
        "ethnicity, or any factor unrelated to job qualifications."
    ),
    backstory=(
        "You are an experienced, impartial hiring analyst known for data-driven, "
        "defensible rankings. You explicitly double-check your own reasoning for "
        "signs of unconscious bias before finalizing any ranking, and you document "
        "that self-review as part of your output."
    ),
    tools=[],
    llm=groq_llm,
    allow_delegation=False,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Agent 3: Interview Question Generator
# ---------------------------------------------------------------------------

interview_agent = Agent(
    role="Interview Question Designer",
    goal=(
        "Design 5-7 tailored interview questions for each of the top 3 ranked "
        "candidates, mixing technical, behavioral, and situational questions "
        "that probe both the role's requirements and the candidate's specific "
        "background, including any gaps identified during screening."
    ),
    backstory=(
        "You are a senior interview panel lead who crafts precise, role-specific "
        "and candidate-specific questions that reveal real signal about a "
        "candidate's ability to succeed in the role."
    ),
    tools=[],
    llm=groq_llm,
    allow_delegation=False,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Agent 4: Hiring Recommendation Agent
# ---------------------------------------------------------------------------

recommendation_agent = Agent(
    role="Hiring Recommendation Lead",
    goal=(
        "Synthesize the screening and ranking results into a final hiring "
        "verdict for each candidate — one of 'Strong Hire', 'Hire', 'Maybe', "
        "or 'No Hire' — along with a concise summary, key strengths, and key risks."
    ),
    backstory=(
        "You are a hiring committee chair responsible for making the final "
        "call on every candidate. You weigh the evidence presented by the "
        "screening and ranking analysts and communicate clear, actionable "
        "verdicts to the hiring manager."
    ),
    tools=[],
    llm=groq_llm,
    allow_delegation=False,
    verbose=True,
)