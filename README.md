# RAG Chatbot (Medical + GenAI)

This project is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, ChromaDB, Sentence Transformers, BioMistral, and Gradio.

## Features
- Reads multiple PDF documents
- Splits documents into chunks
- Creates embeddings for semantic search
- Stores vectors in ChromaDB
- Retrieves relevant context for each question
- Generates concise answers using a local LLM
- Provides a Gradio chatbot interface

## Project Structure

```text
rag-pdf-chatbot/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── diabetes.pdf
    └── genai-interview-questions.pdf
