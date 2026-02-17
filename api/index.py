from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import asyncio
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

def default_response():
    return {
        "Score": 0,
        "SkillsetMatch": "",
        "Summary": "",
        "Recommendation": "Reject",
        "Name": "",
        "Email": ""
    }

@app.post("/")
async def analyze_resume(data: ResumeRequest):
    jd = data.jd[:2000]      # truncate to avoid serverless timeout
    resume = data.resume[:2000]

    prompt = f"""
You are an expert HR Recruitment AI.

Evaluate the following Resume against the Job Description.

JOB DESCRIPTION:
{jd}

RESUME:
{resume}

Return strictly valid JSON only, following this exact format:

{{
  "Score": 85,
  "SkillsetMatch": "Python, FastAPI, Machine Learning",
  "Summary": "Candidate is highly suitable for the role.",
  "Recommendation": "Shortlist",
  "Name": "John Doe",
  "Email": "john.doe@example.com"
}}
"""

    try:
        # Async GPT call with timeout
        response = await asyncio.wait_for(
            client.chat.completions.acreate(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return strictly valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                # Do NOT rely on json_object; treat as text
            ),
            timeout=8
        )

        raw_content = response.choices[0].message.content
        print("GPT raw output:", raw_content)

        try:
            result = json.loads(raw_content)
        except Exception as e:
            print("Failed to parse GPT output:", e)
            return default_response()

        # Validate Score
        try:
            result["Score"] = max(0, min(100, round(float(result.get("Score", 0)))))
        except:
            result["Score"] = 0

        # Validate Recommendation
        if result.get("Recommendation") not in ["Shortlist", "Reject"]:
            result["Recommendation"] = "Reject"

        # Ensure other fields exist
        for key in ["SkillsetMatch", "Summary", "Name", "Email"]:
            if key not in result or not isinstance(result[key], str):
                result[key] = ""

        return result

    except Exception as e:
        print("OpenAI API error:", e)
        return default_response()
