import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr


# =========================================
# 1. LOAD ONLY REQUIRED PDF DOCUMENTS
# =========================================
pdf_files = [
    "data/genai-interview-questions.pdf",
    "data/diabetes.pdf"
]

docs = []

for file in pdf_files:
    loader = PyPDFLoader(file)
    docs.extend(loader.load())

print(f"Documents loaded: {len(docs)}")


# =========================================
# 2. SPLIT DOCUMENTS INTO CHUNKS
# =========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)

print(f"Chunks created: {len(chunks)}")


# =========================================
# 3. CREATE EMBEDDINGS
# =========================================
embeddings = SentenceTransformerEmbeddings(
    model_name="neuml/pubmedbert-base-embeddings"
)


# =========================================
# 4. CREATE VECTOR STORE
# =========================================
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


# =========================================
# 5. DOWNLOAD MODEL
# =========================================
model_path = hf_hub_download(
    repo_id="BioMistral/BioMistral-7B-GGUF",
    filename="ggml-model-Q4_K_M.gguf"
)

print(f"Model path: {model_path}")


# =========================================
# 6. LOAD LLM
# =========================================
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.2,
    max_tokens=150,
    n_ctx=2048,
    verbose=False
)


# =========================================
# 7. FORMAT DOCUMENTS
# =========================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =========================================
# 8. PROMPT TEMPLATE
# =========================================
prompt = PromptTemplate.from_template(
    """<s>[INST]
You are a helpful assistant with expertise in medical and AI topics.

Answer in 2 to 3 lines only.
Be concise and avoid unnecessary details.

Use the context below when relevant.
If the answer is not clearly in the context, you may use general knowledge.
When using general knowledge, start with:
"This answer is based on general knowledge, not from the provided documents."

Context:
{context}

Question:
{question}
[/INST]"""
)


# =========================================
# 9. BUILD RAG PIPELINE
# =========================================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# =========================================
# 10. CREATE CHAT FUNCTION FOR GRADIO
# =========================================
def chat_with_pdf(message, history):
    try:
        answer = rag_chain.invoke(message)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"


# =========================================
# 11. BUILD GRADIO CHATBOT UI
# =========================================
demo = gr.ChatInterface(
    fn=chat_with_pdf,
    title="RAG Chatbot (Medical + GenAI)",
    description="Ask questions from diabetes.pdf and genai-interview-questions.pdf",
    textbox=gr.Textbox(
        placeholder="Type your question here...",
        container=False,
        scale=7
    ),
    theme="soft"
)


# =========================================
# 12. LAUNCH THE CHATBOT
# =========================================
if __name__ == "__main__":
    demo.launch()
