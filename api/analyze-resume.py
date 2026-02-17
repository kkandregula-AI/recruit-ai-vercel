from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)

class ResumeRequest(BaseModel):
    jd: str
    resume: str

@app.post("/")
async def analyze_resume(data: ResumeRequest):

    prompt = f"""
You are an expert HR Recruitment AI.

Your task is to evaluate a candidate’s Resume against a Job Description (JD) and provide an objective, structured assessment.

### Evaluation Guidelines:

1. Score (0–100):
   - 90–100: Excellent match
   - 75–89: Strong match
   - 60–74: Partial match
   - Below 60: Weak match

2. SkillsetMatch:
   - List only key skills appearing in BOTH JD and Resume.
   - Do not invent skills.

3. Summary:
   - Provide a concise 2-sentence professional assessment.

4. Recommendation:
   - "Shortlist" if Score >= 75
   - "Reject" if Score < 75
   - Must be exactly "Shortlist" or "Reject"

JOB DESCRIPTION:
{data.jd}

RESUME:
{data.resume}

You must output valid JSON only.
No markdown formatting.

Required JSON structure:

{{
  "Score": number (0-100),
  "SkillsetMatch": string,
  "Summary": string,
  "Recommendation": string
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Return strictly valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    # Enforce strict structure
    result["Score"] = max(0, min(100, int(result.get("Score", 0))))

    if result.get("Recommendation") not in ["Shortlist", "Reject"]:
        result["Recommendation"] = "Reject"

    return result


# Required for Vercel
handler = app
