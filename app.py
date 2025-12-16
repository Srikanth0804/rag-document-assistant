import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Domain-Specific RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Domain-Specific Document Assistant (RAG)")
st.write(
    "Ask questions from the **Employee Handbook**. "
    "Answers are generated using **RAG with Google Gemini**."
)

# -------------------------------------------------
# Cache RAG pipeline (important for performance)
# -------------------------------------------------
@st.cache_resource
def load_rag_pipeline():
    # 1. Load PDF
    loader = PyPDFLoader("data/documents.pdf")
    documents = loader.load()

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # 3. Local Embeddings (NO API QUOTA ISSUES)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. Prompt (Hallucination Control)
    prompt_template = """
    You are an AI assistant answering questions strictly from the given context.

    Rules:
    - Use ONLY the provided context
    - If the answer is not present, say "The information is not available in the provided documents"
    - Do NOT make assumptions

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 6. LLM (Google Gemini – Generation Only)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # 7. RAG QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain

qa_chain = load_rag_pipeline()

# -------------------------------------------------
# User Input
# -------------------------------------------------
query = st.text_input("Ask a question from the document:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            result = qa_chain({"query": query})
            answer = result["result"]
            sources = result["source_documents"]

        # Display Answer
        st.subheader("✅ Answer")
        st.write(answer)

        # Display Sources
        st.subheader("📚 Source Pages")
        for i, doc in enumerate(sources, 1):
            st.write(f"**Source {i}:** Page {doc.metadata.get('page')}")