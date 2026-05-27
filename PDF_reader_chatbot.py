import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Look for API key in hidden .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# UI styling
st.set_page_config(page_title="PDF AI Scout", page_icon="🔍")
st.title("📄 PDF Question Answering Bot")
st.markdown("Upload a PDF and ask the AI anything about its contents.")

# Initialize session state for chat history and vector DB
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Sidebar for uploads
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your document here", type="pdf")
    st.info("This app uses RAG (Retrieval-Augmented Generation) to ground AI responses in your data.")

    # Show documet stats once processed
    if st.session_state.vector_db is not None:
        st.success(f"📄 Active doc: **{st.session_state.last_uploaded_file}**")
        st.caption(f"💬 Messages in history: {len(st.session_state.chat_history)}")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Clear everything button
    if st.button("🔄️ Reset App"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.session_state.last_uploaded_file = None
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
        st.rerun()

# Process the PDF only when a new file is uploaded
if uploaded_file:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        # Save the uploaded file temporarily to the local disk
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Process the PDF
        with st.spinner("Analyzing PDF..."):
            # Extract text from the PDF
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()

             # Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

            # Create the Vector Store
            # Converts text into numbers so the AI can search them
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_db = Chroma.from_documents(chunks, embeddings)

            # Track which file has been processed and reset chat for new doc
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.chat_history = []


        st.success(f"✅ **{uploaded_file.name}** processed! You can now ask questions.")

    # Display existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # The chat interface
    query = st.chat_input("Ask a question about the PDF:")

    if query:
        # Display and store the user message
        with st.chat_message("user"):
            st.write(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Build and run the RAG chain, then display the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGroq(model = "llama-3.3-70b-versatile", groq_api_key=api_key) 
                prompt = ChatPromptTemplate.from_template("""
                Answer the question based on the context below.
                Context: {context}
                Question: {input}
                """)
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})
                source_docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in source_docs])
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "input": query})

            # Write the answer
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Show the source chunks the answer was drawn from
            with st.expander("📚 View source excerpts"):
                for i, doc in enumerate(source_docs):
                    page = doc.metadata.get("page", "unknown")
                    st.markdown(f"**Excerpt {i+1} - Page {page + 1}:**")
                    st.caption(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                    st.divider()
# No file uploaded yet. Prompt the user
else:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.markdown("""
    ### How it works
    1. **Upload** any PDF using the sidebar
    2. **Wait** a moment while the document is analyzed
    3. **Ask** any question about the content in plain English
    4. The AI retrieves the most relevant sections and generates a grounded answer
                
    ### Tips
    - Ask specific questions for the best results
    - Use **View source excerpts** under each answer to see exactly where the AI found its information
    - Upload a new PDF anytime and the app will reprocess automatically
    """)                   