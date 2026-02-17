from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

class ResumeRequest(BaseModel):
    jd: str
    resume: str

@app.post("/")
async def analyze_resume(data: ResumeRequest):

    prompt = f"""
You are an expert HR Recruitment AI.

Evaluate the Resume against the Job Description.

JOB DESCRIPTION:
{data.jd}

RESUME:
{data.resume}

Return strictly valid JSON only:

{{
  "Score": number (0-100),
  "SkillsetMatch": string,
  "Summary": string,
  "Recommendation": "Shortlist or Reject"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    result["Score"] = max(0, min(100, int(result.get("Score", 0))))
    if result.get("Recommendation") not in ["Shortlist", "Reject"]:
        result["Recommendation"] = "Reject"

    return result
