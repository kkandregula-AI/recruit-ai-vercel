🚀 AI-Powered Candidate Screening (MVP)
An AI-driven recruitment screening platform that automates resume evaluation by comparing candidate resumes against job descriptions using GPT-4o.
This project represents Phase 1 (MVP) of a larger AI Recruitment Intelligence Platform.

📌 Overview
Recruiters spend significant time manually screening resumes.
This MVP automates:
Resume intake


Resume vs Job Description comparison


AI-based scoring


Shortlist / Reject recommendation


Structured evaluation output



✅ Current Features (Phase 1 – Live)
Upload resume (PDF/DOCX/TXT or direct text)


Provide Job Description (JD)- PDF/DOCX/TXT or direct text)


AI-powered resume analysis using GPT-4o


Scorecard-based evaluation


Final recommendation:


✅ Shortlist


❌ Reject


JSON-based structured output for frontend integration



🧠 How It Works
Resume text is extracted from uploaded PDF.


Job description is provided via API request.


GPT-4o evaluates:


Skill match


Experience relevance


Role alignment


Mandatory skill presence


AI generates:


Match score (0–100)


Strengths


Gaps


Recommendation (Shortlist / Reject)


Reasoning summary



🏗 Architecture (MVP)
Frontend (Lovable / Web App)
↓
FastAPI Backend
↓
Resume Parser (PDF extraction)
↓
OpenAI GPT-4o Evaluation Engine
↓
Structured JSON Response

🛠 Tech Stack
Backend: FastAPI


AI Engine: OpenAI GPT-4o


PDF Parsing: PyPDF / pdfplumber


Deployment: Vercel / Render


Frontend: Lovable



📂 API Endpoint
POST 
/screen
Request Body (JSON):
{
  "job_description": "Full job description text here",
  "resume_text": "Extracted resume text"
}

Response:
{
  "match_score": 82,
  "skills_match_score": 85,
  "experience_match_score": 78,
  "mandatory_skills_present": true,
  "strengths": ["Python", "FastAPI", "AWS"],
  "gaps": ["No Kubernetes experience"],
  "final_recommendation": "Shortlist",
  "reasoning": "Strong technical alignment with backend role."
}

🎯 Business Impact
Reduces manual resume screening effort by up to 70–80%


Improves consistency in candidate evaluation


Enables data-driven hiring decisions



🚀 Roadmap
Phase 2 (Planned)
Interview scheduling


Interviewer feedback module


Hiring decision tracking


Offer letter generation


End-to-end recruitment lifecycle management



🔐 Environment Variables
Create a .env file:
OPENAI_API_KEY=your_openai_key

🧪 Running Locally

pip install -r requirements.txt
uvicorn index:app --reload

Deployed through Vercel 
https://recruit-ai-vercel.vercel.app/


and Webhooked to Lovable frontend
https://resume-vs-jd.lovable.app/

👤 Author
Krishnamurthy Kandregula
AI/ML & Product Engineering
