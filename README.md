# 📄 PDF Question Answering Bot

An AI-powered web application that lets you upload any PDF and ask natural language questions about its contents. Built with a RAG (Retrieval-Augmented Generation) pipeline for accurate, context-grounded responses.

🔗 **[Live Demo](https://christianseals7-web-pdf-question-ai.streamlit.app)**

---

## 🚀 Features

- Upload any PDF and instantly start asking questions
- AI retrieves the most relevant sections before answering
- Persistent chat history within a session
- Source excerpt viewer shows exactly where answers came from
- Reset and re-upload a new document anytime

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web interface and deployment |
| LangChain | RAG pipeline orchestration |
| Groq API (LLaMA 3) | Language model for answering questions |
| HuggingFace Embeddings | Converting text to vectors |
| FAISS | Vector database for semantic search |
| PyPDF | PDF parsing and text extraction |

---

## 🧠 How It Works

1. User uploads a PDF
2. The document is split into chunks and converted into vector embeddings
3. When a question is asked, the most relevant chunks are retrieved using FAISS
4. The retrieved context is passed to LLaMA 3 via Groq to generate an answer
5. The answer is displayed along with the source excerpts it was drawn from

---

## 💻 Run Locally

1. Clone the repository
```bash
   git clone https://github.com/christianseals7-web/PDF-question-AI.git
   cd PDF-question-AI
```

2. Install dependencies
```bash
   pip install -r requirements.txt
```

3. Create a `.env` file and add your Groq API key
4. Run the app
```bash
   streamlit run PDF_reader_chatbot.py
```

---

## 📬 Contact

Christian Seals — [LinkedIn](https://www.linkedin.com/in/christian-seals) — christian.seals7@gmail.com
