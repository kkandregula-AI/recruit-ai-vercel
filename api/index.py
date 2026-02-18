from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import re
from functools import lru_cache

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
# Input model
# -------------------------------
class ResumeRequest(BaseModel):
    jd: str
    resume: str

# -------------------------------
# Fallback response
# -------------------------------
def default_response():
    return {
        "Score": 0,
        "SkillsetMatch": "",
        "MissingSkills": "",
        "Summary": "",
        "Recommendation": "",
        "Name": "",
        "Email": ""
    }

# -------------------------------
# Dynamic weight calculation
# -------------------------------
def dynamic_weight(skill_text: str):
    must_keywords = ["must-have", "required", "essential", "mandatory"]
    nice_keywords = ["nice-to-have", "desirable", "optional", "preferable"]
    must_count = sum(skill_text.lower().count(k) for k in must_keywords)
    nice_count = sum(skill_text.lower().count(k) for k in nice_keywords)
    total = must_count + nice_count
    if total == 0:
        return {"must_have_weight": 50, "nice_to_have_weight": 20}
    must_weight = round(70 * (must_count / total), 2)
    nice_weight = round(30 * (nice_count / total), 2)
    return {"must_have_weight": must_weight, "nice_to_have_weight": nice_weight}

# -------------------------------
# Recommendation helper
# -------------------------------
score_benchmarks = {
    "Strong Shortlist": (85, 100),
    "Shortlist": (70, 84),
    "Borderline": (50, 69),
    "Reject": (0, 49)
}

def decide_recommendation(score):
    for decision, (low, high) in score_benchmarks.items():
        if low <= score <= high:
            return "Shortlist" if decision != "Reject" else "Reject"
    return "Reject"

# -------------------------------
# Format clean Summary with colored inline bullets
# -------------------------------
def format_summary_html(matched_skills, missing_skills, reasoning_text=""):
    matched_html = "".join([f"• <span style='color:green'>{s}</span><br>" for s in matched_skills])
    missing_html = "".join([f"• <span style='color:red'>{s}</span><br>" for s in missing_skills])
    summary = "<b>Matched Skills:</b><br>" + (matched_html if matched_html else "None<br>")
    summary += "<b>Missing Skills:</b><br>" + (missing_html if missing_html else "None<br>")
    if reasoning_text:
        summary += "<b>Reasoning:</b><br>" + reasoning_text
    return summary

# -------------------------------
# Cached GPT evaluation
# -------------------------------
@lru_cache(maxsize=128)
def cached_gpt_evaluate(jd_text: str, resume_text: str, must_weight: float, nice_weight: float):
    prompt = f"""
You are an expert HR Recruitment AI.

Resume:
{resume_text}

Job Description:
{jd_text}

Instructions:
1. Extract the candidate's Name and Email from the resume.
2. Extract must-have and nice-to-have skills from JD and candidate.
3. Calculate a matching score (0-100) using adaptive weights:
   Must-have skills weight: {must_weight}%
   Nice-to-have skills weight: {nice_weight}%
4. Provide a recommendation: "Shortlist" or "Reject"
5. Generate a summary with inline bullets highlighting skills:
   - Matched skills: green
   - Missing skills: red
6. Return strictly JSON with fields:
   Score, SkillsetMatch, MissingSkills, Summary, Recommendation, Name, Email
7. Do not output any text outside JSON.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON-only HR assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error in GPT evaluation:", e)
        return None

# -------------------------------
# Fallback skill extraction
# -------------------------------
def fallback_skills(text):
    common_skills = ["Python", "Java", "C++", "Machine Learning", "AI", "FastAPI", "DevOps", "Generative AI", "Agentic AI",
                     "Sales experience", "Understanding client needs", "Communicating product value",
                     "Building sales pipelines", "Nurturing leads", "Closing deals"]
    return [skill for skill in common_skills if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]

# -------------------------------
# POST endpoint
# -------------------------------
@app.post("/")
def analyze_resume(data: ResumeRequest):
    if not OPENAI_API_KEY or not client:
        return {"error": "OPENAI_API_KEY is not set."}

    jd_text = data.jd[:1000]
    resume_text = data.resume[:1000]

    # Adaptive weights
    weights = dynamic_weight(jd_text)
    must_weight = weights["must_have_weight"]
    nice_weight = weights["nice_to_have_weight"]

    # Single GPT call
    try:
        raw_output = cached_gpt_evaluate(jd_text, resume_text, must_weight, nice_weight)
    except Exception as e:
        print("GPT call failed:", e)
        raw_output = None

    if not raw_output:
        # fallback skill match
        jd_skills = fallback_skills(jd_text)
        candidate_skills = fallback_skills(resume_text)
        matching = [s for s in candidate_skills if s in jd_skills]
        missing = [s for s in jd_skills if s not in candidate_skills]
        score = round((len(matching)/max(1,len(jd_skills)))*100, 2)
        recommendation = decide_recommendation(score)
        summary = format_summary_html(matching, missing, reasoning_text="Fallback: GPT unavailable, basic skill matching used.")
        return {
            "Score": score,
            "SkillsetMatch": ", ".join(matching),
            "MissingSkills": ", ".join(missing),
            "Summary": summary,
            "Recommendation": recommendation,
            "Name": "",
            "Email": ""
        }

    # Parse GPT JSON safely
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start == -1 or end == -1:
            return default_response()
        raw_json = raw_output[start:end]
        result = json.loads(raw_json)
    except Exception as e:
        print("Failed to parse GPT output:", e)
        return default_response()

    # Ensure all fields exist
    for key in ["Score", "SkillsetMatch", "MissingSkills", "Summary", "Recommendation", "Name", "Email"]:
        if key not in result or result[key] is None:
            result[key] = "" if key != "Score" else 0

    # Convert lists to strings
    matched_skills = result.get("SkillsetMatch", [])
    if isinstance(matched_skills, list):
        result["SkillsetMatch"] = ", ".join(matched_skills)
    else:
        matched_skills = [s.strip() for s in str(result.get("SkillsetMatch", "")).split(",") if s.strip()]
        result["SkillsetMatch"] = ", ".join(matched_skills)

    missing_skills = result.get("MissingSkills", [])
    if isinstance(missing_skills, list):
        result["MissingSkills"] = ", ".join(missing_skills)
    else:
        missing_skills = [s.strip() for s in str(result.get("MissingSkills", "")).split(",") if s.strip()]
        result["MissingSkills"] = ", ".join(missing_skills)

    # Clean inline Summary
    reasoning_text = result.get("Summary", "")
    result["Summary"] = format_summary_html(matched_skills, missing_skills, reasoning_text)

    # Validate score & recommendation
    try:
        result["Score"] = max(0, min(100, round(float(result.get("Score", 0)))))
    except:
        result["Score"] = 0
    if result.get("Recommendation") not in ["Shortlist", "Reject"]:
        result["Recommendation"] = decide_recommendation(result["Score"])

    # Name and Email extracted from GPT resume parsing
    result["Name"] = result.get("Name", "")
    result["Email"] = result.get("Email", "")

    return result
