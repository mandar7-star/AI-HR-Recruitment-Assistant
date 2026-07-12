---
title: AI HR Recruitment Assistant API
emoji: 🧑‍💼
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---

# AI HR Recruitment Assistant — Backend API

FastAPI backend running a 4-agent CrewAI pipeline (Screening → Ranking →
Interview Questions → Hiring Recommendation) powered by Groq LLaMA 3.3 70B,
with local resume retrieval via LlamaIndex + ChromaDB.

Visit `/docs` for the interactive Swagger UI.
