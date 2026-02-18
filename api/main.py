import os
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise Exception("OPENAI_API_KEY not configured")

client = OpenAI(api_key=api_key)


def extract_name(resume_text):
    lines = resume_text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if len(line.split()) <= 4 and "@" not in line:
            return line
    return "Unknown Candidate"


def extract_email(resume_text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)
    return match.group(0) if match else ""


def decide_recommendation(score):
    if score >= 75:
        return "Shortlist"
    elif score >= 50:
        return "Consider"
    else:
        return "Reject"


@app.post("/api/evaluate")
async def evaluate(request: Request):
    try:
        body = await request.json()

        jd = body.get("jd", "")
        resume = body.get("resume", "")

        if not jd or not resume:
            return JSONResponse(
                status_code=400,
                content={"error": "Both JD and Resume are required"}
            )

        name = extract_name(resume)
        email = extract_email(resume)

        prompt = f"""
Return STRICT JSON:
{{
  "Score": integer,
  "Role": string,
  "MatchedSkills": string,
  "MissingSkills": string,
  "Summary": string
}}

Job Description:
{jd}

Resume:
{resume}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        ai_output = response.choices[0].message.content
        ai_output = ai_output.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(ai_output)

        score = int(parsed.get("Score", 0))

        return {
            "Score": score,
            "Role": parsed.get("Role", ""),
            "SkillsetMatch": parsed.get("MatchedSkills", ""),
            "MissingSkills": parsed.get("MissingSkills", ""),
            "Summary": parsed.get("Summary", ""),
            "Recommendation": decide_recommendation(score),
            "Name": name,
            "Email": email
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/")
def health():
    return {"status": "Recruit AI Running 🚀"}
