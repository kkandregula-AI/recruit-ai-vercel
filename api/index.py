from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json

# -------------------------------
# FastAPI setup (DO NOT CHANGE)
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
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
        "Role": "",
        "SkillsetMatch": "",
        "MissingSkills": "",
        "Summary": "",
        "Recommendation": "Reject",
        "Name": "",
        "Email": ""
    }

# -------------------------------
# POST endpoint (STABLE)
# -------------------------------
@app.post("/")
def analyze_resume(data: ResumeRequest):

    if not OPENAI_API_KEY or not client:
        return {
            "error": "OPENAI_API_KEY is not set."
        }

    # Limit input size for serverless stability
    jd = data.jd[:1200]
    resume = data.resume[:1200]

    prompt = f"""
You are an expert HR Recruitment AI.

Evaluate the Resume against the Job Description using professional hiring standards.

SCORING FRAMEWORK:
- 90–100: Excellent alignment, all critical skills match → Shortlist
- 75–89: Strong alignment, minor skill gaps → Shortlist
- 60–74: Moderate alignment, some important gaps → Use judgment
- Below 60: Poor alignment or missing core requirements → Reject

SCREENING CRITERIA:
1. Compare required vs actual skills
2. Identify critical missing skills
3. Evaluate years and relevance of experience
4. Check role alignment
5. Decide if gaps are trainable or fundamental

Return STRICTLY valid JSON only in this format:

{{
  "Score": 85,
  "Role": "AI Engineer",
  "MatchedSkills": "Python, FastAPI, Machine Learning",
  "MissingSkills": "Agentic AI, DevOps",
  "Summary": "Explain clearly why shortlisted or rejected based on skills, experience, and fit.",
  "Recommendation": "Shortlist",
  "Name": "John Doe",
  "Email": "john.doe@example.com"
}}

JOB DESCRIPTION:
{jd}

RESUME:
{resume}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON-only HR screening assistant. Never output text outside JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw_content = response.choices[0].message.content.strip()
        print("GPT raw output:", raw_content)

        # Extract JSON safely
        start = raw_content.find("{")
        end = raw_content.rfind("}") + 1

        if start == -1 or end == -1:
            return default_response()

        raw_json = raw_content[start:end]

        try:
            result = json.loads(raw_json)
        except:
            return default_response()

        # -------------------------------
        # Score Validation
        # -------------------------------
        try:
            score = max(0, min(100, round(float(result.get("Score", 0)))))
        except:
            score = 0

        # -------------------------------
        # Recommendation Logic
        # -------------------------------
        gpt_recommendation = result.get("Recommendation", "Reject")

        if score >= 75:
            final_recommendation = "Shortlist"
        elif score < 60:
            final_recommendation = "Reject"
        else:
            # 60–74 zone → trust GPT if valid
            if gpt_recommendation in ["Shortlist", "Reject"]:
                final_recommendation = gpt_recommendation
            else:
                final_recommendation = "Reject"

        # -------------------------------
        # Safe Field Extraction
        # -------------------------------
        role = result.get("Role", "")
        matched = result.get("MatchedSkills", "")
        missing = result.get("MissingSkills", "")
        summary = result.get("Summary", "")
        name = result.get("Name", "")
        email = result.get("Email", "")

        role = role if isinstance(role, str) else ""
        matched = matched if isinstance(matched, str) else ""
        missing = missing if isinstance(missing, str) else ""
        summary = summary if isinstance(summary, str) else ""
        name = name if isinstance(name, str) else ""
        email = email if isinstance(email, str) else ""

        # -------------------------------
        # Final Flat JSON Response
        # -------------------------------
        return {
            "Score": score,
            "Role": role,
            "SkillsetMatch": matched,
            "MissingSkills": missing,
            "Summary": summary,
            "Recommendation": final_recommendation,
            "Name": name,
            "Email": email
        }

    except Exception as e:
        print("OpenAI API error:", e)
        return default_response()
