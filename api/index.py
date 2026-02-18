from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import re

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# OpenAI setup
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is NOT set!")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------
# Input model (ONLY jd + resume)
# -------------------------------
class ResumeRequest(BaseModel):
    jd: str
    resume: str

# -------------------------------
# Default safe response
# -------------------------------
def default_response():
    return {
        "Score": 0,
        "SkillsetMatch": "",
        "MissingSkills": "",
        "Summary": "Evaluation could not be completed.",
        "Recommendation": "Reject",
        "Name": "",
        "Email": ""
    }

# -------------------------------
# Score Benchmark Logic
# -------------------------------
def decide_recommendation(score: int):
    if score >= 70:
        return "Shortlist"
    return "Reject"

# -------------------------------
# Clean HTML Summary formatter
# -------------------------------
def format_summary(matched, missing, reasoning=""):
    matched_html = "".join(
        [f"• <span style='color:green'>{s}</span><br>" for s in matched]
    )
    missing_html = "".join(
        [f"• <span style='color:red'>{s}</span><br>" for s in missing]
    )

    summary = "<b>Matched Skills:</b><br>"
    summary += matched_html if matched_html else "None<br>"

    summary += "<br><b>Missing Skills:</b><br>"
    summary += missing_html if missing_html else "None<br>"

    if reasoning:
        summary += "<br><b>Reasoning:</b><br>" + reasoning

    return summary

# -------------------------------
# Main Endpoint
# -------------------------------
@app.post("/")
def analyze_resume(data: ResumeRequest):

    if not client:
        return {"error": "OPENAI_API_KEY not configured"}

    jd_text = data.jd[:1500]
    resume_text = data.resume[:1500]

    # -------------------------------
    # GPT CALL (fully protected)
    # -------------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON-only HR assistant. Never output text outside JSON."
                },
                {
                    "role": "user",
                    "content": f"""
Evaluate the following resume against the job description.

Resume:
{resume_text}

Job Description:
{jd_text}

Instructions:
1. Extract candidate Name and Email from resume.
2. Extract matching skills.
3. Extract missing skills.
4. Calculate a match score between 0-100.
5. Provide recommendation: Shortlist if score >=70 else Reject.

Return STRICT JSON only in this format:

{{
  "Score": 0,
  "SkillsetMatch": [],
  "MissingSkills": [],
  "Summary": "",
  "Recommendation": "",
  "Name": "",
  "Email": ""
}}
"""
                }
            ],
        )

        raw_output = response.choices[0].message.content.strip()

    except Exception as e:
        print("OPENAI ERROR:", str(e))
        return default_response()

    if not raw_output:
        print("Empty GPT response")
        return default_response()

    # -------------------------------
    # Safe JSON Parsing
    # -------------------------------
    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        parsed = json.loads(raw_output[start:end])
    except Exception as e:
        print("JSON PARSE ERROR:", str(e))
        print("RAW OUTPUT:", raw_output)
        return default_response()

    # -------------------------------
    # Extract & sanitize fields
    # -------------------------------
    score = parsed.get("Score", 0)
    try:
        score = int(float(score))
        score = max(0, min(100, score))
    except:
        score = 0

    matched = parsed.get("SkillsetMatch", [])
    missing = parsed.get("MissingSkills", [])

    if not isinstance(matched, list):
        matched = [s.strip() for s in str(matched).split(",") if s.strip()]

    if not isinstance(missing, list):
        missing = [s.strip() for s in str(missing).split(",") if s.strip()]

    reasoning = parsed.get("Summary", "")

    recommendation = decide_recommendation(score)

    name = parsed.get("Name", "")
    email = parsed.get("Email", "")

    # -------------------------------
    # Clean formatted summary
    # -------------------------------
    summary = format_summary(matched, missing, reasoning)

    # -------------------------------
    # Final Safe Output
    # -------------------------------
    return {
        "Score": score,
        "SkillsetMatch": ", ".join(matched),
        "MissingSkills": ", ".join(missing),
        "Summary": summary,
        "Recommendation": recommendation,
        "Name": name,
        "Email": email
    }
 
