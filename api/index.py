from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

app = FastAPI()  # <-- Vercel auto-detects this

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
async def analyze_resume(
    jd: str = Form(...),
    resume: str = Form(...)
):
    prompt = f"""
You are an expert HR Recruitment AI.

Evaluate the following Resume against the Job Description.

JOB DESCRIPTION:
{jd}

RESUME:
{resume}

Return strictly valid JSON only, following this format:

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
        response = await client.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return strictly valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content

        try:
            result["Score"] = max(0, min(100, round(float(result.get("Score", 0)))))
        except:
            result["Score"] = 0

        if result.get("Recommendation") not in ["Shortlist", "Reject"]:
            result["Recommendation"] = "Reject"

        for key in ["SkillsetMatch", "Summary", "Name", "Email"]:
            if key not in result or not isinstance(result[key], str):
                result[key] = ""

        return result

    except Exception as e:
        print("OpenAI API error:", e)
        return default_response()
