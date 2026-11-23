# 📄 PDF Question Answering Bot

An AI-powered web application that lets you upload any PDF and ask natural language questions about its contents. Built with a RAG (Retrieval-Augmented Generation) pipeline for accurate, context-grounded responses with page citations.

🔗 **[Live Demo](https://pdf-question-ai-3psnt6gdcmjxqvhdml6qdb.streamlit.app/)**

---

## 🚀 Features

- Upload any PDF and instantly start asking questions
- Answers cite the page numbers they came from, and say so when the document doesn't contain the answer
- Conversational follow-ups: questions like "what about the second one?" are rewritten into standalone search queries
- Streaming responses, word by word
- Source excerpt viewer shows exactly where each answer came from
- Download the chat as Markdown
- Clear error messages for scanned (image-only) PDFs, bad API keys, and rate limits

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Streamlit | Web interface and deployment |
| LangChain | RAG pipeline orchestration |
| Groq API (`openai/gpt-oss-20b`) | Language model for answering questions |
| HuggingFace Embeddings (`all-MiniLM-L6-v2`) | Converting text to vectors |
| FAISS | Vector database for semantic search |
| PyPDF | PDF parsing and text extraction |

---

## 🧠 How It Works

1. User uploads a PDF
2. The document is split into overlapping chunks and converted into vector embeddings (the embedding model is cached, so it loads only once)
3. For follow-up questions, the LLM first rewrites the question using recent chat history so the search finds the right sections
4. FAISS retrieves the most relevant chunks
5. The chunks, labeled with page numbers, are passed to the LLM via Groq, which streams back a grounded, cited answer
6. The answer is displayed along with the source excerpts it was drawn from

---

## 💻 Run Locally

1. Clone the repository
```bash
   git clone https://github.com/christianseals7-web/PDF-question-AI.git
   cd PDF-question-AI
```

2. Install dependencies (Python 3.11 recommended)
```bash
   pip install -r requirements.txt
```

3. Create a `.env` file with your Groq API key (it's in `.gitignore`, so it won't be committed)
```
   GROQ_API_KEY=your_key_here
```

4. Run the app
```bash
   streamlit run PDF_reader_chatbot.py
```

On Streamlit Cloud, add `GROQ_API_KEY` under **Manage app → Settings → Secrets** instead of using a `.env` file.

---

## 📬 Contact

Christian Seals — [LinkedIn](https://www.linkedin.com/in/christian-seals) — christian.seals7@gmail.com