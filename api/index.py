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
    candidate_name: str = ""
    candidate_email: str = ""

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
# Dynamic weight calculation based on JD keywords
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
# Cached GPT evaluation (single call)
# -------------------------------
@lru_cache(maxsize=128)
def cached_gpt_evaluate(jd_text: str, resume_text: str, candidate_name: str, candidate_email: str, must_weight: float, nice_weight: float):
    """
    GPT call: extract skills, calculate weighted score, generate summary with highlighted skills
    """
    prompt = f"""
You are an expert HR Recruitment AI.

Candidate Name: {candidate_name}
Candidate Email: {candidate_email}

Job Description:
{jd_text}

Resume:
{resume_text}

Instructions:
1. Extract must-have and nice-to-have skills from JD and candidate.
2. Calculate a matching score (0-100) using these adaptive weights:
   Must-have skills weight: {must_weight}%
   Nice-to-have skills weight: {nice_weight}%
3. Provide a recommendation: "Shortlist" or "Reject"
4. Generate a summary that visually highlights skills:
   - Matched skills: green
   - Missing skills: red
5. Return strictly JSON with fields:
   Score, SkillsetMatch, MissingSkills, Summary, Recommendation, Name, Email
6. Use HTML or Markdown formatting in Summary for color highlighting.
Do not output any text outside JSON.
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
    common_skills = ["Python", "Java", "C++", "Machine Learning", "AI", "FastAPI", "DevOps", "Generative AI", "Agentic AI"]
    return [skill for skill in common_skills if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]

# -------------------------------
# POST endpoint
# -------------------------------
@app.post("/")
def analyze_resume(data: ResumeRequest):
    if not OPENAI_API_KEY or not client:
        return {"error": "OPENAI_API_KEY is not set. Please configure it in your environment variables."}

    jd_text = data.jd[:1000]
    resume_text = data.resume[:1000]

    # -------------------------------
    # Adaptive weights
    # -------------------------------
    weights = dynamic_weight(jd_text)
    must_weight = weights["must_have_weight"]
    nice_weight = weights["nice_to_have_weight"]

    # -------------------------------
    # Single GPT call
    # -------------------------------
    raw_output = cached_gpt_evaluate(jd_text, resume_text, data.candidate_name, data.candidate_email, must_weight, nice_weight)
    if not raw_output:
        jd_skills = fallback_skills(jd_text)
        candidate_skills = fallback_skills(resume_text)
        matching = [s for s in candidate_skills if s in jd_skills]
        missing = [s for s in jd_skills if s not in candidate_skills]
        score = round((len(matching)/max(1,len(jd_skills)))*100, 2)
        recommendation = decide_recommendation(score)
        # fallback visual highlighting in summary
        summary = "Matched skills: " + ", ".join([f"<span style='color:green'>{s}</span>" for s in matching])
        summary += "<br>Missing skills: " + ", ".join([f"<span style='color:red'>{s}</span>" for s in missing])
        return {
            "Score": score,
            "SkillsetMatch": ", ".join(matching),
            "MissingSkills": ", ".join(missing),
            "Summary": summary,
            "Recommendation": recommendation,
            "Name": data.candidate_name,
            "Email": data.candidate_email
        }

    # -------------------------------
    # Parse JSON from GPT
    # -------------------------------
    start = raw_output.find("{")
    end = raw_output.rfind("}") + 1
    if start == -1 or end == -1:
        return default_response()

    raw_json = raw_output[start:end]
    try:
        result = json.loads(raw_json)
    except Exception as e:
        print("Failed to parse GPT output:", e)
        return default_response()

    # -------------------------------
    # Ensure all fields exist
    # -------------------------------
    for key in ["Score", "SkillsetMatch", "MissingSkills", "Summary", "Recommendation", "Name", "Email"]:
        if key not in result:
            result[key] = "" if key != "Score" else 0

    # Validate Score & Recommendation
    try:
        result["Score"] = max(0, min(100, round(float(result.get("Score",0)))))
    except:
        result["Score"] = 0
    if result.get("Recommendation") not in ["Shortlist", "Reject"]:
        result["Recommendation"] = decide_recommendation(result["Score"])
    result["Name"] = data.candidate_name or result.get("Name","")
    result["Email"] = data.candidate_email or result.get("Email","")

    return result
