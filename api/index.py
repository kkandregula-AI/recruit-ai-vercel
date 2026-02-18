import os
import re
import json
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# -----------------------------
# Helper: Extract Name
# -----------------------------
def extract_name(resume_text):
    lines = resume_text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if len(line.split()) <= 4 and not "@" in line:
            return line
    return "Unknown Candidate"


# -----------------------------
# Helper: Extract Email
# -----------------------------
def extract_email(resume_text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)
    return match.group(0) if match else ""


# -----------------------------
# Recommendation Logic
# -----------------------------
def decide_recommendation(score):
    if score >= 75:
        return "Shortlist"
    elif score >= 50:
        return "Consider"
    else:
        return "Reject"


# -----------------------------
# Main Evaluation Route
# -----------------------------
@app.route("/evaluate", methods=["POST"])
def evaluate():

    try:
        data = request.get_json()

        jd = data.get("jd", "")
        resume = data.get("resume", "")

        if not jd or not resume:
            return jsonify({"error": "Both JD and Resume are required"}), 400

        # Extract name & email automatically
        name = extract_name(resume)
        email = extract_email(resume)

        # GPT Prompt
        prompt = f"""
You are an HR AI assistant.

Compare the following Job Description and Resume.

Return STRICT JSON with:
- Score (0-100 integer)
- Role (Job role inferred from JD)
- MatchedSkills (comma separated string)
- MissingSkills (comma separated string)
- Summary (professional 4-5 line HR brief about candidate background and suitability)

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

        # Clean markdown if present
        ai_output = ai_output.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(ai_output)

        score = int(parsed.get("Score", 0))
        role = parsed.get("Role", "")
        matched = parsed.get("MatchedSkills", "")
        missing = parsed.get("MissingSkills", "")
        summary_text = parsed.get("Summary", "")

        recommendation = decide_recommendation(score)

        return jsonify({
            "Score": score,
            "Role": role,
            "SkillsetMatch": matched,
            "MissingSkills": missing,
            "Summary": summary_text,
            "Recommendation": recommendation,
            "Name": name,
            "Email": email
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Health Check
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Recruit AI Running Successfully 🚀"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
