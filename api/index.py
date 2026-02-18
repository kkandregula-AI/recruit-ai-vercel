from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import re

# ---------------------------------
# FastAPI Setup
# ---------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# OpenAI Setup
# ---------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is NOT set!")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------------------
# Input Model (ONLY jd + resume)
# ---------------------------------
class ResumeRequest(BaseModel):
    jd: str
    resume: str

# ---------------------------------
# Safe Default Response
# ---------------------------------
def default_response():
    return {
        "Candidate": {
            "Name": "",
            "Email": "",
            "Role": "Not Specified"
        },
        "Evaluation": {
            "Score": 0,
            "Recommendation": "Reject"
        },
        "Skills": {
            "Matched": "",
            "Missing": ""
        },
        "Summary": "Evaluation could not be completed."
    }

# ---------------------------------
# Recommendation Logic (Deterministic)
# ---------------------------------
def decide_recommendation(score: int):
    return "Shortlist" if score >= 70 else "Reject"

# ---------------------------------
# Extract Role Fallback (if GPT misses)
# ---------------------------------
def extract_role_from_jd(jd_text):
    role_pattern = r"(Senior|Junior|Lead)?\s?[A-Za-z ]+(Developer|Engineer|Executive|Manager|Analyst|Consultant|Specialist)"
    match = re.search(role_pattern, jd_text, re.IGNORECASE)
    return match.group(0) if match else "Not Specified"

# ---------------------------------
# Main Endpoint
# ---------------------------------
@app.post("/")
def analyze_resume(data: ResumeRequest):

    if not client:
        return {"error": "OPENAI_API_KEY not configured"}

    jd_text = data.jd[:2000]
    resume_text = data.resume[:2000]

    # ---------------------------------
    # GPT Call (Fully Protected)
    # ---------------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON-only HR screening assistant. Never output text outside JSON."
                },
                {
                    "role": "user",
                    "content": f"""
Evaluate the resume against the job description.

Resume:
{resume_text}

Job Description:
{jd_text}

Instructions:
1. Identify the target Role from the Job Description.
2. Extract candidate Name and Email from resume.
3. Extract matching skills.
4. Extract missing skills.
5. Assign a match score between 0-100 based on skill overlap.
6. Provide a professional HR summary (brief paragraph).

Return STRICT JSON only in this format:

{{
  "Score": 0,
  "Role": "",
  "SkillsetMatch": [],
  "MissingSkills": [],
  "Summary": "",
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

    # ---------------------------------
    # Safe JSON Parsing
    # ---------------------------------
    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        parsed = json.loads(raw_output[start:end])
    except Exception as e:
        print("JSON PARSE ERROR:", str(e))
        print("RAW OUTPUT:", raw_output)
        return default_response()

    # ---------------------------------
    # Sanitize & Extract Fields
    # ---------------------------------
    score = parsed.get("Score", 0)
    try:
        score = int(float(score))
        score = max(0, min(100, score))
    except:
        score = 0

    role = parsed.get("Role", "")
    if not role:
        role = extract_role_from_jd(jd_text)

    matched = parsed.get("SkillsetMatch", [])
    missing = parsed.get("MissingSkills", [])

    if not isinstance(matched, list):
        matched = [s.strip() for s in str(matched).split(",") if s.strip()]

    if not isinstance(missing, list):
        missing = [s.strip() for s in str(missing).split(",") if s.strip()]

    summary_text = parsed.get("Summary", "")
    name = parsed.get("Name", "")
    email = parsed.get("Email", "")

    recommendation = decide_recommendation(score)

    # ---------------------------------
    # Final Structured Output
    # ---------------------------------
    return {
        "Candidate": {
            "Name": name,
            "Email": email,
            "Role": role
        },
        "Evaluation": {
            "Score": score,
            "Recommendation": recommendation
        },
        "Skills": {
            "Matched": ", ".join(matched),
            "Missing": ", ".join(missing)
        },
        "Summary": summary_text
    }
