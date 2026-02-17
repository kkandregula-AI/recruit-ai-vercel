from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# Default response if AI fails
def default_response():
    return {
        "Score": 0,
        "SkillsetMatch": "",
        "Summary": "",
        "Recommendation": "Reject",
        "Name": "",
        "Email": ""
    }

@app.post("/analyze")
async def analyze_resume(
    jd: str = Form(...),
    resume: str = Form(...)
):
    """
    Accepts form-data (jd, resume) and returns JSON evaluation.
    """
    prompt = f"""
You are an expert HR Recruitment AI.

Evaluate the following Resume against the Job Description.

JOB DESCRIPTION:
{jd}

RESUME:
{resume}

Return strictly valid JSON only, following this exact format and replace the example values appropriately:

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
        # Async call to OpenAI
        response = await client.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content  # Already a dict
        print("MODEL OUTPUT:", result)  # Debugging

        # Validate Score
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
        print(f"OpenAI API error: {e}")
        return default_response()
