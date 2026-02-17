from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend domain in production
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
        "Summary": "",
        "Recommendation": "Reject",
        "Name": "",
        "Email": ""
    }

# -------------------------------
# POST endpoint
# -------------------------------
@app.post("/")
def analyze_resume(data: ResumeRequest):
    # Safety check: API key must be set
    if not OPENAI_API_KEY or not client:
        return {
            "error": "OPENAI_API_KEY is not set. Please configure it in your environment variables."
        }

    # Truncate inputs to avoid serverless timeout / large token issues
    jd = data.jd[:1000]
    resume = data.resume[:1000]

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
        # Synchronous GPT call (reliable in serverless)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON-only HR assistant. NEVER output text outside JSON."
                },
                {"role": "user", "content": prompt}
            ],
        )

        # Raw GPT output
        raw_content = response.choices[0].message.content.strip()
        print("GPT raw output:", raw_content)

        # Extract first JSON block safely
        start = raw_content.find("{")
        end = raw_content.rfind("}") + 1
        if start == -1 or end == -1:
            return default_response()

        raw_json = raw_content[start:end]

        # Parse JSON
        try:
            result = json.loads(raw_json)
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

        # Ensure all other fields exist
        for key in ["SkillsetMatch", "Summary", "Name", "Email"]:
            if key not in result or not isinstance(result[key], str):
                result[key] = ""

        return result

    except Exception as e:
        print("OpenAI API error:", e)
        return default_response()
