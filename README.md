# RAG Chatbot (Medical + GenAI)

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents using a local LLM.

## Features
- Multi-document support (Medical + GenAI PDFs)
- Semantic search using embeddings
- Vector database (ChromaDB)
- Local LLM inference using LlamaCpp (BioMistral GGUF)
- Hybrid prompt (context + fallback knowledge)
- Interactive chatbot UI using Gradio

## Architecture

User Query → Retriever → Context → LLM → Answer

## Tech Stack
- LangChain
- Sentence Transformers
- ChromaDB
- LlamaCpp (GGUF model)
- Gradio

## How It Works
1. Load PDF documents
2. Split into chunks
3. Generate embeddings
4. Store in vector database
5. Retrieve relevant chunks
6. Pass context + query to LLM
7. Generate concise answer

## Setup

```bash
pip install -r requirements.txt
python app.py
