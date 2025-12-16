📄 Domain-Specific RAG Document Assistant

A domain-specific Retrieval Augmented Generation (RAG) application built using LangChain and Large Language Models (LLMs) to answer queries from internal policy documents.
The system demonstrates how GenAI can be safely applied in enterprise environments by grounding LLM responses strictly in retrieved document context.

🔴 Problem Statement

Generic LLMs perform well on open-domain knowledge but fail in organization-specific scenarios such as:

Employee policies

Internal SOPs

Compliance documents

Without access to internal context, LLMs tend to hallucinate or provide inaccurate answers, which is unacceptable in enterprise settings.

✅ Solution Overview (RAG Approach)

This project implements a Retrieval Augmented Generation (RAG) pipeline that:

Ingests internal PDF documents (Employee Handbook)

Converts document content into semantic vector embeddings

Retrieves the most relevant document chunks for a user query

Uses an LLM to generate answers strictly from retrieved context

This ensures accurate, explainable, and hallucination-controlled responses.

🏗️ Architecture
PDF Document (Employee Handbook)
        ↓
Document Loader (LangChain)
        ↓
Text Chunking
        ↓
Embedding Generation (Hugging Face)
        ↓
Vector Store (FAISS)
        ↓
Semantic Retriever
        ↓
LLM (Google Gemini)
        ↓
Context-Grounded Answer

🛠️ Tech Stack

Programming Language: Python

Framework: LangChain

LLMs: Google Gemini

Embeddings: Hugging Face Sentence Transformers

Vector Database: FAISS

UI: Streamlit

Environment Management: Python Virtual Environment (venv)

🚀 Features

Domain-specific question answering over internal documents

Semantic search using vector embeddings

Hallucination control via prompt constraints

Source-aware responses (page-level traceability)

Interactive Streamlit interface

📂 Project Structure
rag_document_assistant/
│
├── data/
│   └── documents.pdf
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
