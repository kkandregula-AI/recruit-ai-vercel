from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# Request model
class ResumeRequest(BaseModel):
    jd: str
    resume: str

# Default response template
def default_response():
    return {
        "Score": 0,
        "SkillsetMatch": "",
        "Summary": "",
        "Recommendation": "Reject",
        "Name": "",
        "Email": ""
    }

# POST endpoint
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
  "Recommendation": "Shortlist or Reject",
  "Name": string,
  "Email": string
}}
"""

    try:
        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # Extract result
        result = response.choices[0].message.content  # Should already be dict

        # Ensure Score is 0-100 integer
        try:
            result["Score"] = max(0, min(100, round(float(result.get("Score", 0)))))
        except (ValueError, TypeError):
            result["Score"] = 0

        # Validate Recommendation
        if result.get("Recommendation") not in ["Shortlist", "Reject"]:
            result["Recommendation"] = "Reject"

        # Ensure all other fields exist as strings
        for key in ["SkillsetMatch", "Summary", "Name", "Email"]:
            if key not in result or not isinstance(result[key], str):
                result[key] = ""

        return result

    except Exception as e:
        # Log the error if you want (print or proper logger)
        print(f"OpenAI API error: {e}")
        # Return default JSON response
        return default_response()

