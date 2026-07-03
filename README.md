# 🧑‍💼 AI HR Recruitment Assistant

An Advanced AI-powered **HR Recruitment Assistant** built on a **4-Agent CrewAI Pipeline** that screens, ranks, interviews-preps, and evaluates candidate resumes end to end from a single job description and a batch of resumes.

The system integrates **Groq LLaMA 3.3 70B** for high-speed reasoning, **LlamaIndex + ChromaDB** for local resume retrieval (RAG), a **FastAPI** backend for orchestration, and a **Streamlit** dashboard for an interactive, recruiter-friendly experience — all agent stages run strictly **sequentially** to respect Groq's free-tier rate limits, with programmatic (non-LLM) merging of results for reliability.

---

## 🚀 Key Highlights

- 🤖 **Multi-Agent Orchestration** using **CrewAI**
  
- ⚡ **High-Speed Inference** using **Groq LLaMA 3.3 70B** (via LiteLLM)
  
- 🔍 **Local Resume Retrieval (RAG)** via **LlamaIndex + ChromaDB** (in-memory, ephemeral, no external vector DB required)
  
- 🧮 **Bias-Aware Ranking** — every score ships with an explicit fairness self-review
  
- 🎤 **Targeted Interview Questions** auto-generated for the top 3 ranked candidates
  
- ✅ **Final Hiring Verdicts** — Strong Hire / Hire / Maybe / No Hire, with strengths & risks
  
- 🚀 **FastAPI Backend** with `/analyze` and `/query` endpoints
  
- 🖥️ **Interactive Streamlit Dashboard** with candidate comparison table, profile cards, and one-click PDF export
  
- 🛡️ **Robust Pipeline Engineering** — JSON-safe parsing, retry with exponential backoff on rate limits, truncated prompts to control token growth, and a full agent execution trace

---

## ⭐ Key Features

- **Structured Resume Screening** — extracts matched/missing skills, certifications, awards, notable projects, experience and education into a validated Pydantic schema, grounded strictly in resume text
  
- **Objective Candidate Ranking** — scores every candidate 0–100 against the job description, with a written justification, a confidence level, and an explicit bias self-check for name, gender, age, and ethnicity
  
- **Role- and Candidate-Specific Interview Questions** — 5–7 mixed technical/behavioral/situational questions per top-3 candidate, tuned to probe matched strengths and identified skill gaps
  
- **Committee-Style Hiring Recommendations** — a final verdict, concise summary, key strengths, and key risks for every candidate
  
- **Local RAG Retrieval Tool** — the Screening Agent can query indexed resumes (via a HuggingFace `BAAI/bge-small-en-v1.5` embedding index) to confirm details rather than relying purely on prompt context
  
- **Rate-Limit-Aware Orchestration** — 5–10s delays between agent stages plus exponential-backoff retries, tuned for Groq's free-tier limits (30 RPM / 12,000 TPM)
  
- **Multi-Format Resume Ingestion** — PDF, DOCX, and TXT resumes are parsed server-side, with graceful skipping of unreadable files
  
- **Recruiter Dashboard** — sortable comparison table, skill-match rings, verdict pills, and a downloadable PDF report generated with ReportLab

---

## 🧠 System Architecture

```
Job Description + Resumes (PDF/DOCX/TXT)
              ↓
   FastAPI /analyze endpoint
              ↓
  Resume Text Extraction + Local Indexing (ChromaDB)
              ↓
Screening Agent      → Resume Retrieval Tool (LlamaIndex + ChromaDB RAG)
              ↓  (rate-limit delay)

Ranking Agent        → Fairness self-review + 0–100 scoring
              ↓  (rate-limit delay)

Interview Agent      → 5–7 questions for Top 3 candidates
              ↓  (rate-limit delay)

Recommendation Agent → Final verdict per candidate
              ↓
Programmatic JSON merge → FinalReport
              ↓
   Streamlit Dashboard + PDF Export
```

Each stage's output is parsed as JSON and passed forward as a **compact, programmatically-built summary** (not raw conversation history), keeping token usage bounded as the pipeline progresses.

---

## 🔹 Agent Details

### 🔍 Resume Screening Specialist
Reads every resume against the job description and produces a structured `CandidateScreen` profile — matched skills, missing skills, certifications, awards, notable projects, an experience summary, and an education summary — using the **Resume Retrieval Tool** to confirm details from the indexed resume text rather than assuming.

---

### 📊 Candidate Ranking Analyst
Scores each candidate 0–100 purely on job fit and assigns a rank (1 = best). Every score comes with a written justification, a `Low/Medium/High` confidence level, and a **fairness note** — an explicit self-review confirming the score wasn't influenced by name, gender, age, ethnicity, or any other factor unrelated to qualifications.

---

### 🎤 Interview Question Designer
Generates 5–7 tailored interview questions for each of the **top 3** ranked candidates only, mixing technical, behavioral, and situational formats — deliberately probing both the role's requirements and any skill gaps surfaced during screening.

---

### ✅ Hiring Recommendation Lead
Synthesizes the screening and ranking output into a final verdict per candidate — `Strong Hire`, `Hire`, `Maybe`, or `No Hire` — along with a concise summary, key strengths, and key risks for the hiring manager.

---

## 🏗️ Project Structure

```
AI_HR_Recruitment_Assistant/
├── main.py                  # FastAPI backend — /analyze and /query endpoints
├── agents.py                 # 4 CrewAI agents (Groq/LiteLLM-backed) + Groq compatibility patch
├── tasks.py                  # Sequential pipeline orchestration, retries, JSON merging
├── tools.py                   # LlamaIndex + ChromaDB resume indexing & retrieval tool
├── models.py                 # Pydantic schemas shared across agents, API, and frontend
├── app.py                    # Streamlit dashboard (candidate comparison, profiles, PDF export)
├── .env                       # GROQ_API_KEY
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Tech Stack

| Category | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3 70B Versatile (via LiteLLM) |
| Agent Orchestration | CrewAI |
| RAG / Retrieval | LlamaIndex + ChromaDB (ephemeral, local) |
| Embeddings | HuggingFace — BAAI/bge-small-en-v1.5 |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Resume Parsing | pypdf, python-docx |
| PDF Report Export | ReportLab |
| Data Validation | Pydantic |

---

## 🔑 API Keys Required

| Service | Link |
|---|---|
| Groq | https://console.groq.com |

This project is intentionally lightweight on external dependencies — resume indexing and retrieval run **fully locally** via ChromaDB and a HuggingFace embedding model (downloaded once and cached), so the only API key you need is Groq.

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 🧪 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-HR-Recruitment-Assistant.git
cd AI-HR-Recruitment-Assistant
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file as shown above with your `GROQ_API_KEY`.

---

## ▶️ Run the Application

### 1. Start the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### 2. Start the Streamlit frontend (in a separate terminal)

```bash
streamlit run app.py
```

---

## 💡 Example Usage

```
Job Description: "Senior Backend Engineer — 5+ years Python, distributed
systems, and AWS experience required."

Upload: resume_1.pdf, resume_2.docx, resume_3.txt

→ Screening profiles for all 3 candidates
→ Ranked 0–100 with fairness-reviewed justifications
→ Interview questions for the top 3
→ Final Strong Hire / Hire / Maybe / No Hire verdicts
→ Downloadable PDF report
```

---

## 🖥️ UI Preview

### 📝 Job Description & Resume Upload
<img src="screenshots/upload.png" width="800"/>

### 🤖 Agent Pipeline Running
<img src="screenshots/pipeline.png" width="800"/>

### 📊 Candidate Dashboard & Comparison Table
<img src="screenshots/dashboard.png" width="800"/>

### 👤 Candidate Profile Cards
<img src="screenshots/profile.png" width="800"/>

---

## 🎯 Use Cases

- Multi-Agent AI System Demonstration
- End-to-End Recruitment Automation Prototype
- Portfolio Project for AI/ML Roles
- Base for Building Production HR-Tech Assistants (ATS integration, bulk screening, etc.)

---

## 👨‍💻 Author

**Your Name**

LinkedIn: https://www.linkedin.com/in/mandarborhade

GitHub: https://github.com/mandar7-star

Live Demo : https://ai-hr-recruitment-assistant.streamlit.app/
