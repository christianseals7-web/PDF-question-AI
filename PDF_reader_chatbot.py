import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import AuthenticationError, RateLimitError
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Settings (all the knobs in one place)
# ---------------------------------------------------------------------------
LLM_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4            # how many chunks to retrieve per question
HISTORY_TURNS = 3    # how many past question/answer pairs the model sees

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Step 1: turn a follow-up like "what about the second one?" into a question
# that makes sense on its own, so the vector search can find the right chunks.
CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation below and a follow-up question, rewrite the follow-up
as a standalone question that makes sense without the conversation.
If it is already standalone, return it unchanged. Return ONLY the question.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""
)

# Step 2: answer using only the retrieved excerpts, with page citations.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions about a PDF the user uploaded.\n"
        "Use ONLY the excerpts below. If the answer is not in them, say you couldn't find it "
        "in the document instead of guessing.\n"
        "When you use information from an excerpt, cite its page like (p. 3).\n"
        "Keep answers clear and concise.\n\n"
        "Excerpts:\n{context}",
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_api_key():
    """Read the Groq key from a local .env file, or from Streamlit Cloud secrets."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:  # no secrets file exists locally
            key = None
    return key


@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load the embedding model once per server, not on every upload."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_llm(api_key):
    return ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0.2)


def page_label(doc):
    """PyPDF pages are 0-indexed; show them 1-indexed. Handle missing page numbers."""
    page = doc.metadata.get("page")
    return page + 1 if isinstance(page, int) else "?"


def build_vector_db(file_bytes, embeddings):
    """PDF bytes -> searchable FAISS index. Returns (db, page_count, chunk_count)."""
    # A unique temp file per upload, so two users on the live app can't
    # overwrite each other's PDF (the old shared temp.pdf could).
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    text_pages = [p for p in pages if p.page_content.strip()]
    if not text_pages:
        return None, len(pages), 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(text_pages)
    return FAISS.from_documents(chunks, embeddings), len(pages), len(chunks)


def recent_history(messages):
    """The last few turns, in the (role, text) format LangChain prompts accept."""
    recent = messages[-HISTORY_TURNS * 2:]
    return [("human" if m["role"] == "user" else "ai", m["content"]) for m in recent]


def condense_question(llm, question, history):
    if not history:
        return question
    history_text = "\n".join(f"{role}: {text}" for role, text in history)
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"history": history_text, "question": question}).strip() or question


def format_context(docs):
    return "\n\n".join(f"[Page {page_label(d)}]\n{d.page_content}" for d in docs)


def show_sources(sources):
    with st.expander("📚 View source excerpts"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**Excerpt {i} – Page {src['page']}:**")
            text = src["text"]
            st.caption(text[:500] + ("..." if len(text) > 500 else ""))
            if i < len(sources):
                st.divider()


def chat_as_markdown(messages, file_name):
    lines = [f"# Chat about {file_name}\n"]
    for m in messages:
        who = "You" if m["role"] == "user" else "AI"
        lines.append(f"**{who}:** {m['content']}\n")
    return "\n".join(lines)


def reset_session(clear_file=False):
    st.session_state.chat_history = []
    if clear_file:
        st.session_state.vector_db = None
        st.session_state.file_hash = None
        st.session_state.file_name = None
        st.session_state.doc_stats = None
        st.session_state.uploader_key += 1  # new key = empty uploader widget


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PDF AI Scout", page_icon="🔍")
st.title("📄 PDF Question Answering Bot")
st.markdown("Upload a PDF and ask the AI anything about its contents.")

defaults = {
    "chat_history": [],
    "vector_db": None,
    "file_hash": None,
    "file_name": None,
    "doc_stats": None,
    "uploader_key": 0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

api_key = get_api_key()
if not api_key:
    st.error(
        "**Missing GROQ_API_KEY.** Locally, add it to a `.env` file. "
        "On Streamlit Cloud, add it under *Manage app → Settings → Secrets*."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload your document here", type="pdf", key=f"uploader_{st.session_state.uploader_key}"
    )
    st.info("This app uses RAG (Retrieval-Augmented Generation) to ground AI responses in your document.")

    if st.session_state.vector_db is not None:
        stats = st.session_state.doc_stats
        st.success(f"📄 Active doc: **{st.session_state.file_name}**")
        st.caption(
            f"{stats['pages']} pages · {stats['chunks']} chunks · "
            f"{len(st.session_state.chat_history)} messages"
        )

    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear chat", use_container_width=True):
        reset_session()
        st.rerun()
    if col2.button("🔄️ Reset app", use_container_width=True):
        reset_session(clear_file=True)
        st.rerun()

    if st.session_state.chat_history:
        st.download_button(
            "⬇️ Download chat",
            data=chat_as_markdown(st.session_state.chat_history, st.session_state.file_name),
            file_name="pdf_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# No file yet: show instructions
# ---------------------------------------------------------------------------
if not uploaded_file:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.markdown("""
    ### How it works
    1. **Upload** any PDF using the sidebar
    2. **Wait** a moment while the document is analyzed
    3. **Ask** any question about the content in plain English
    4. The AI retrieves the most relevant sections and generates a grounded answer, with page citations

    ### Tips
    - Ask specific questions for the best results
    - Follow-up questions work — the bot remembers the last few exchanges
    - Use **View source excerpts** under each answer to see exactly where the AI found its information
    """)
    st.stop()

# ---------------------------------------------------------------------------
# Process a new PDF (detected by content hash, so re-uploading an edited file
# with the same name still gets reprocessed)
# ---------------------------------------------------------------------------
file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.file_hash != file_hash:
    with st.spinner("Analyzing PDF... (the first upload takes longer while the model loads)"):
        try:
            db, n_pages, n_chunks = build_vector_db(file_bytes, load_embeddings())
        except Exception as e:
            st.error(f"Couldn't read that PDF. It may be corrupted or password-protected.\n\n`{e}`")
            st.stop()

    if db is None:
        st.error(
            "No text found in this PDF. It's probably a scanned image — "
            "try a PDF with selectable text."
        )
        st.stop()

    st.session_state.vector_db = db
    st.session_state.file_hash = file_hash
    st.session_state.file_name = uploaded_file.name
    st.session_state.doc_stats = {"pages": n_pages, "chunks": n_chunks}
    st.session_state.chat_history = []
    st.rerun()  # refresh the sidebar stats right away

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

query = st.chat_input("Ask a question about the PDF:")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        llm = load_llm(api_key)
        history = recent_history(st.session_state.chat_history)
        try:
            with st.spinner("Searching the document..."):
                search_query = condense_question(llm, query, history)
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": TOP_K})
                source_docs = retriever.invoke(search_query)

            chain = ANSWER_PROMPT | llm | StrOutputParser()
            # Stream the answer word-by-word instead of waiting for all of it
            answer = st.write_stream(chain.stream({
                "context": format_context(source_docs),
                "chat_history": history,
                "question": query,
            }))
        except AuthenticationError:
            st.error("Groq rejected the API key. Check that GROQ_API_KEY is correct and hasn't been revoked.")
            st.stop()
        except RateLimitError:
            st.error("Hit Groq's rate limit. Wait a minute and try again.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong while answering: `{e}`")
            st.stop()

        sources = [{"page": page_label(d), "text": d.page_content} for d in source_docs]
        show_sources(sources)

    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()  # keep the sidebar's message count and download button in sync     