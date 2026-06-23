"""
assessment.py — Credehub AI Assessment Module
=============================================
Handles:
  1. MCQ generation  → generate_assessment()
  2. Report creation → generate_report()

To add a new subject in future:
  - Add its FAISS retriever in main.py
  - Pass it to generate_assessment() via the retriever arguments
  - No changes needed here

To tune difficulty mix: change DIFFICULTY_RATIO below.
To change Karachi Board prompt style: edit MCQ_SYSTEM_PROMPT / MCQ_USER_PROMPT.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from groq import Groq
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# CONFIG — change these without touching any logic
# ─────────────────────────────────────────────────────────────

# All 7 units of Class 9 CS (Karachi Board)
UNITS_CS9: List[str] = [
    "Unit 1 - Fundamentals of Computer",
    "Unit 2 - Fundamentals of Operating System",
    "Unit 3 - Office Automation",
    "Unit 4 - Data Communication and Computer Networks",
    "Unit 5 - Computer Security and Ethics",
    "Unit 6 - Web Development",
    "Unit 7 - Introduction to Database System",
]

# Difficulty ratios — Karachi Board exam pattern
# Easy:  direct recall, definitions, full forms
# Medium: conceptual, "NOT", comparisons
# Hard:  application, multi-step, edge cases
DIFFICULTY_RATIO = {"easy": 0.40, "medium": 0.40, "hard": 0.20}

# Thresholds for unit performance labels
STRONG_THRESHOLD  = 70.0   # ≥70% → Strong
AVERAGE_THRESHOLD = 40.0   # 40–69% → Average  |  <40% → Weak

# Groq model used for assessment (can be swapped here only)
GROQ_MODEL = "openai/gpt-oss-120b"

# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────

class AssessmentRequest(BaseModel):
    """Input for starting an assessment session."""
    num_questions: int = Field(default=20, ge=5, le=50,
                               description="Total MCQs (5–50)")
    subject: str = Field(default="Computer Science")
    grade: int   = Field(default=9)


class MCQOption(BaseModel):
    A: str
    B: str
    C: str
    D: str


class MCQQuestion(BaseModel):
    """A single MCQ question with metadata."""
    id:             int
    question:       str
    options:        MCQOption
    correct_answer: str        # "A" | "B" | "C" | "D"
    unit:           str
    topic:          str
    difficulty:     str        # "easy" | "medium" | "hard"
    explanation:    str        # shown on result screen


class AssessmentResponse(BaseModel):
    """Response returned by /assessment/generate."""
    total_questions: int
    subject:         str
    grade:           int
    questions:       List[MCQQuestion]


class StudentAnswer(BaseModel):
    """One student answer for one question."""
    question_id: int
    selected:    str           # "A" | "B" | "C" | "D"


class ReportRequest(BaseModel):
    """Input for /assessment/report endpoint."""
    questions:       List[MCQQuestion]
    student_answers: List[StudentAnswer]
    student_name:    str = Field(default="")


class UnitResult(BaseModel):
    """Per-unit performance breakdown."""
    unit:      str
    total:     int
    correct:   int
    score_pct: float
    status:    str             # "Strong" | "Average" | "Weak"


class AssessmentReport(BaseModel):
    """Complete assessment report returned by /assessment/report."""
    student_name:          str
    total_score:           int
    total_questions:       int
    percentage:            float
    grade_label:           str     # A+ / A / B / C / D / Fail
    unit_results:          List[UnitResult]
    strong_units:          List[str]
    average_units:         List[str]
    weak_units:            List[str]
    ai_narrative:          str     # AI-written personal report
    study_recommendations: List[str]


# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────

MCQ_SYSTEM_PROMPT = (
    "You are an MCQ question generator for Karachi Board Class 9 examinations. "
    "Return ONLY valid JSON — no markdown, no code fences, no extra text, no think tags."
)

MCQ_USER_PROMPT = """\
Generate exactly {count} MCQ questions for the following unit of Class {grade} {subject} (Karachi Board):

UNIT: {unit}

DIFFICULTY DISTRIBUTION (follow exactly):
- Easy   ({easy_count} questions) : Direct definitions, full forms/abbreviations, simple recall facts
- Medium ({medium_count} questions): Conceptual — "Which of the following is NOT...", comparisons, purpose-based
- Hard   ({hard_count} questions) : Application-based, exception cases, multi-step reasoning

KARACHI BOARD EXAM STYLE:
- Questions should match the style of Karachi Board objective paper (MCQs section)
- Use precise book-accurate language
- Distractors must be plausible but clearly incorrect
- Avoid trick questions

CURRICULUM CONTENT (generate questions ONLY from this):
{context}

OUTPUT — Return ONLY this JSON structure:
{{
  "mcqs": [
    {{
      "question": "What does ALU stand for?",
      "options": {{
        "A": "Arithmetic Logic Unit",
        "B": "Arithmetic Logical Unit",
        "C": "Automated Logic Unit",
        "D": "Arithmetic Load Unit"
      }},
      "correct_answer": "A",
      "topic": "CPU Components",
      "difficulty": "easy",
      "explanation": "ALU stands for Arithmetic Logic Unit. It performs arithmetic and logical operations inside the CPU."
    }}
  ]
}}
"""

REPORT_SYSTEM_PROMPT = (
    "You are an experienced academic counselor writing a student assessment report. "
    "Be honest but encouraging. Write in clear English paragraphs."
)

REPORT_USER_PROMPT = """\
Write a detailed assessment report for the following student:

Student: {student_name}
Subject: Class 9 Computer Science (Karachi Board)
Score: {total_score}/{total_questions} ({percentage:.1f}%) — Grade: {grade_label}

Unit-wise Performance:
{unit_performance}

Strong Units  (≥70%): {strong_units}
Average Units (40–69%): {average_units}
Weak Units    (<40%): {weak_units}

Write 4 paragraphs:
1. Overall performance summary
2. Strong areas — what the student has mastered
3. Weak areas — specific knowledge gaps that need attention
4. Study plan & Karachi Board exam tips — what to focus on for each weak unit, how to prepare

End with a short motivational closing line.
Be specific about units and topics, not generic. Tone: professional teacher writing a report card comment.
"""


# ─────────────────────────────────────────────────────────────
# HELPER: DISTRIBUTE QUESTIONS ACROSS UNITS
# ─────────────────────────────────────────────────────────────

def _distribute(total: int, units: List[str]) -> List[int]:
    """
    Evenly spread `total` questions across `units`.
    Remainder questions go to the first units.
    Example: 20 questions, 7 units → [3,3,3,3,3,3,2]
    """
    n = len(units)
    base = total // n
    remainder = total % n
    counts = [base] * n
    for i in range(remainder):
        counts[i] += 1
    return counts


def _difficulty_counts(total: int) -> tuple[int, int, int]:
    """Return (easy, medium, hard) counts for a given total."""
    easy   = max(1, round(total * DIFFICULTY_RATIO["easy"]))
    hard   = max(1, round(total * DIFFICULTY_RATIO["hard"]))
    medium = max(1, total - easy - hard)
    return easy, medium, hard


# ─────────────────────────────────────────────────────────────
# HELPER: CLEAN AI OUTPUT
# ─────────────────────────────────────────────────────────────

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> and unclosed <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*",          "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json(raw: str) -> dict:
    """Pull the first {...} JSON block from the AI response."""
    raw = _strip_think_tags(raw)
    # Remove markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in AI response")
    return json.loads(match.group())


# ─────────────────────────────────────────────────────────────
# CORE: GENERATE MCQs FOR ONE UNIT
# ─────────────────────────────────────────────────────────────

def _fetch_unit_context(unit: str, vector_store, bm25_retriever) -> str:
    """Pull curriculum content for a unit from FAISS + BM25."""
    # Primary: semantic search filtered by unit
    try:
        docs = vector_store.similarity_search(unit, k=10, filter={"unit": unit})
    except Exception:
        docs = []

    # Fallback: BM25 keyword search
    if not docs:
        docs = bm25_retriever.invoke(unit)

    return "\n\n".join(d.page_content for d in docs[:8]) if docs else ""


def _generate_unit_mcqs(
    unit:           str,
    count:          int,
    grade:          int,
    subject:        str,
    vector_store,
    bm25_retriever,
    client:         Groq,
) -> List[dict]:
    """
    Generate `count` MCQs for a single `unit`.
    Returns a list of raw MCQ dicts (from JSON).
    Returns [] on any failure — the caller skips empty units gracefully.
    """
    context = _fetch_unit_context(unit, vector_store, bm25_retriever)
    if not context:
        return []

    easy, medium, hard = _difficulty_counts(count)

    user_msg = MCQ_USER_PROMPT.format(
        count=count,
        grade=grade,
        subject=subject,
        unit=unit,
        easy_count=easy,
        medium_count=medium,
        hard_count=hard,
        context=context,
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=3500,
            messages=[
                {"role": "system", "content": MCQ_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw  = resp.choices[0].message.content
        data = _extract_json(raw)
        mcqs = data.get("mcqs", [])

        # Attach unit to each MCQ (AI may omit it)
        for mcq in mcqs:
            mcq["unit"] = unit

        return mcqs[:count]  # Never return more than requested

    except Exception as exc:
        # Log and continue — one bad unit shouldn't kill the whole assessment
        print(f"  [assessment] MCQ generation failed for {unit}: {exc}")
        return []


# ─────────────────────────────────────────────────────────────
# PUBLIC: GENERATE FULL ASSESSMENT
# ─────────────────────────────────────────────────────────────

def generate_assessment(
    request:        AssessmentRequest,
    vector_store,
    bm25_retriever,
    chunks,
    client:         Groq,
) -> AssessmentResponse:
    """
    Generate a full set of MCQs spread across all units.
    Called by POST /assessment/generate in main.py.
    """
    units  = UNITS_CS9  # Future: make this subject-aware
    counts = _distribute(request.num_questions, units)

    all_mcqs: List[MCQQuestion] = []
    question_id = 1

    for unit, count in zip(units, counts):
        if count == 0:
            continue

        raw_mcqs = _generate_unit_mcqs(
            unit=unit,
            count=count,
            grade=request.grade,
            subject=request.subject,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            client=client,
        )

        for raw in raw_mcqs:
            try:
                opts = raw.get("options", {})
                mcq = MCQQuestion(
                    id=question_id,
                    question=raw.get("question", "").strip(),
                    options=MCQOption(
                        A=opts.get("A", ""),
                        B=opts.get("B", ""),
                        C=opts.get("C", ""),
                        D=opts.get("D", ""),
                    ),
                    correct_answer=str(raw.get("correct_answer", "A")).upper(),
                    unit=raw.get("unit", unit),
                    topic=raw.get("topic", unit),
                    difficulty=raw.get("difficulty", "medium").lower(),
                    explanation=raw.get("explanation", "").strip(),
                )
                all_mcqs.append(mcq)
                question_id += 1
            except Exception as exc:
                print(f"  [assessment] Skipping malformed MCQ: {exc}")
                continue

    return AssessmentResponse(
        total_questions=len(all_mcqs),
        subject=request.subject,
        grade=request.grade,
        questions=all_mcqs,
    )


# ─────────────────────────────────────────────────────────────
# PUBLIC: GENERATE ASSESSMENT REPORT
# ─────────────────────────────────────────────────────────────

def _grade_label(pct: float) -> str:
    if pct >= 90: return "A+"
    if pct >= 80: return "A"
    if pct >= 70: return "B"
    if pct >= 60: return "C"
    if pct >= 50: return "D"
    return "Fail"


def _calc_unit_results(
    questions:       List[MCQQuestion],
    student_answers: List[StudentAnswer],
) -> List[UnitResult]:
    """Score each unit based on student answers."""
    ans_map = {a.question_id: a.selected.upper() for a in student_answers}

    unit_data: dict[str, dict] = {}
    for q in questions:
        entry = unit_data.setdefault(q.unit, {"total": 0, "correct": 0})
        entry["total"] += 1
        if ans_map.get(q.id, "") == q.correct_answer.upper():
            entry["correct"] += 1

    results: List[UnitResult] = []
    for unit, data in unit_data.items():
        pct = round(data["correct"] / data["total"] * 100, 1) if data["total"] else 0.0
        if pct >= STRONG_THRESHOLD:
            status = "Strong"
        elif pct >= AVERAGE_THRESHOLD:
            status = "Average"
        else:
            status = "Weak"
        results.append(UnitResult(
            unit=unit,
            total=data["total"],
            correct=data["correct"],
            score_pct=pct,
            status=status,
        ))

    return results


def generate_report(
    request: ReportRequest,
    client:  Groq,
) -> AssessmentReport:
    """
    Score the assessment and produce an AI-generated report.
    Called by POST /assessment/report in main.py.
    """
    unit_results  = _calc_unit_results(request.questions, request.student_answers)

    total_correct = sum(r.correct for r in unit_results)
    total_qs      = sum(r.total   for r in unit_results)
    pct           = round(total_correct / total_qs * 100, 1) if total_qs else 0.0

    strong_units  = [r.unit for r in unit_results if r.status == "Strong"]
    average_units = [r.unit for r in unit_results if r.status == "Average"]
    weak_units    = [r.unit for r in unit_results if r.status == "Weak"]

    grade         = _grade_label(pct)
    student_name  = request.student_name.strip() or "Student"

    # Build unit performance string for the prompt
    unit_perf = "\n".join(
        f"  • {r.unit}: {r.correct}/{r.total} ({r.score_pct}%) — {r.status}"
        for r in unit_results
    )

    user_msg = REPORT_USER_PROMPT.format(
        student_name=student_name,
        total_score=total_correct,
        total_questions=total_qs,
        percentage=pct,
        grade_label=grade,
        unit_performance=unit_perf,
        strong_units=", ".join(strong_units)  or "None",
        average_units=", ".join(average_units) or "None",
        weak_units=", ".join(weak_units)       or "None",
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        narrative = _strip_think_tags(resp.choices[0].message.content)
    except Exception as exc:
        print(f"  [assessment] Report generation failed: {exc}")
        narrative = (
            f"Assessment complete. Score: {total_correct}/{total_qs} ({pct}%). "
            f"Strong: {', '.join(strong_units) or 'None'}. "
            f"Weak: {', '.join(weak_units) or 'None'}."
        )

    # Extract bullet-point recommendations from narrative
    recommendations: List[str] = []
    for line in narrative.splitlines():
        line = line.strip()
        if line and line[0] in ("-", "•", "*", "→", "►") and len(line) > 8:
            recommendations.append(re.sub(r"^[-•*→►]\s*", "", line).strip())

    return AssessmentReport(
        student_name=student_name,
        total_score=total_correct,
        total_questions=total_qs,
        percentage=pct,
        grade_label=grade,
        unit_results=unit_results,
        strong_units=strong_units,
        average_units=average_units,
        weak_units=weak_units,
        ai_narrative=narrative,
        study_recommendations=recommendations,
    )
