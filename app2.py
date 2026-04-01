from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import streamlit as st
import os

load_dotenv()

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="ChatPDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827, #1e293b);
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    .info-card {
        background: #111827;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.92rem;
    }

    .feature-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem 0.35rem 0.25rem 0;
        border-radius: 999px;
        background: rgba(59,130,246,0.15);
        color: #bfdbfe;
        font-size: 0.85rem;
        border: 1px solid rgba(59,130,246,0.25);
    }

    .ocr-note {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(59,130,246,0.12));
        border: 1px solid rgba(16,185,129,0.25);
        padding: 1rem 1.1rem;
        border-radius: 14px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .chat-tip {
        background: rgba(255,255,255,0.03);
        padding: 0.85rem 1rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        padding: 1rem;
        border-radius: 14px;
        border: 1px dashed rgba(255,255,255,0.15);
    }

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
    }

    .footer-note {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Header -------------------
st.markdown("""
<div class="hero-card">
    <h1 style="margin-bottom:0.4rem;">📄 ChatPDF</h1>
    <p class="small-muted" style="margin-bottom:0.8rem;">
        Upload a PDF and chat with it using Retrieval-Augmented Generation (RAG).
    </p>
    <div>
        <span class="feature-pill">PDF Q&A</span>
        <span class="feature-pill">Semantic Search</span>
        <span class="feature-pill">FAISS Vector Store</span>
        <span class="feature-pill">Hugging Face LLM</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
with st.sidebar:
    st.markdown("## ⚙️ App Panel")

    st.markdown("""
    <div class="info-card">
        <h4 style="margin-bottom:0.5rem;">How it works</h4>
        <p class="small-muted">
            1. Upload a PDF<br>
            2. The file is split into chunks<br>
            3. Embeddings are created<br>
            4. Ask questions from the document
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ocr-note">
        <h4 style="margin-bottom:0.4rem;">🚀 Coming Next Update</h4>
        <p class="small-muted" style="margin-bottom:0;">
            <strong>Scanned PDF support</strong> will be added soon using <strong>OCR</strong>,
            so you'll be able to chat with image-based / scanned documents too.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4 style="margin-bottom:0.5rem;">Tips</h4>
        <p class="small-muted">
            • Ask specific questions for better answers<br>
            • Works best on text-based PDFs right now<br>
            • If the answer isn't in the file, the bot won't hallucinate
        </p>
    </div>
    """, unsafe_allow_html=True)

# For parsing String output
parser = StrOutputParser()

# ------------------- LLM -------------------
llm = HuggingFaceEndpoint(
    repo_id='moonshotai/Kimi-K2-Thinking',
    provider='auto',
    task='text-generation',
)

model = ChatHuggingFace(llm=llm)

# ------------------- Embedding model to use -------------------
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
)

# ------------------- Session State -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ------------------- Upload Section -------------------
st.markdown("""
<div class="info-card">
    <h3 style="margin-bottom:0.4rem;">📤 Upload your PDF</h3>
    <p class="small-muted" style="margin-bottom:0;">
        Upload a text-based PDF to start chatting with its content.
    </p>
</div>
""", unsafe_allow_html=True)

pdf_input = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

if pdf_input is not None:
    os.makedirs("data/uploads", exist_ok=True)
    file_path = os.path.join("data/uploads", pdf_input.name)

    with open(file_path, "wb") as f:
        f.write(pdf_input.getbuffer())

    if st.session_state["current_pdf"] != pdf_input.name:
        st.info("🔄 Processing new PDF...")

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        extracted_docs = splitter.split_documents(docs)

        st.write(f"📚 **Total chunks created:** {len(extracted_docs)}")

        vector_store = FAISS.from_documents(extracted_docs, embedding_model)

        st.session_state['current_pdf'] = pdf_input.name
        st.session_state['extracted_docs'] = extracted_docs
        st.session_state['vector_store'] = vector_store

        st.success("✅ PDF processed successfully!")
    else:
        st.info("📄 Using already processed PDF.")

# ------------------- Retriever -------------------
retriever = None
vector_store = st.session_state.get("vector_store")

if vector_store is not None:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

# ------------------- Prompt -------------------
prompt = PromptTemplate(
    template="""
You are a helpful AI Assistant.

Answer the user's question from the context provided below ONLY.
If you don't know the answer, just say:
"I couldn't find answer in the document"
Always mention page numbers where the answer(s) are found, like this:

According to Page 1 or As mentioned on Page(2,5)

Always try giving citations for every answer.
Don't make up answers on your own.

Question: {question}

Context: {context}
""",
    input_variables=["context", "question"]
)

# ------------------- Helper Function to store metadata and chunks for further use -------------------
def format_docs(retrieved_docs):
    formatted_chunks = []

    for doc in retrieved_docs:
        page    = doc.metadata.get("page","Unknown")
        content = doc.page_content.strip()

        formatted_chunks.append(f"Page{page}\n\n{content}")
    return '\n\n'.join(formatted_chunks)
            
    

# ------------------- Chains -------------------
final_chain = None

if retriever is not None:
    rag_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    final_chain = rag_chain | prompt | model | parser

# ------------------- Chat Section -------------------
# st.markdown("""
# <div class="info-card">
#     <h3 style="margin-bottom:0.4rem;">💬 Chat with your document</h3>
#     <p class="small-muted" style="margin-bottom:0;">
#         Ask anything about the uploaded PDF.
#     </p>
# </div>
# """, unsafe_allow_html=True)

# if retriever is None:
#     st.markdown("""
#     <div class="chat-tip">
#         <strong>Tip:</strong> Upload a PDF first, then ask questions like:
#         <br>• "Summarize this document"
#         <br>• "What are the main points?"
#         <br>• "Explain page 3"
#     </div>
#     """, unsafe_allow_html=True)

# ------------------- Display Old Chat -------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------- User Input -------------------
user_question = st.chat_input("Ask anything about the uploaded PDF...")

if user_question:
    if retriever is None or final_chain is None:
        st.warning("⚠️ Please upload a PDF first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Show assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response = final_chain.invoke(user_question)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# ------------------- Reset Section -------------------
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🗑️ Reset Chat", use_container_width=True):
        # Delete uploaded PDF file from disk (optional but recommended)
        if st.session_state.get("current_pdf"):
            file_path = os.path.join("data/uploads", st.session_state["current_pdf"])
            if os.path.exists(file_path):
                os.remove(file_path)

        # Clear all related session state
        st.session_state.messages = []
        st.session_state.current_pdf = None
        st.session_state.vector_store = None
        st.session_state.extracted_docs = None

        st.success("✅ Chat and uploaded PDF cleared!")
        st.rerun()

with col2:
    st.markdown("""
    <p class="small-muted" style="padding-top:0.45rem;">
        Reset removes the current uploaded PDF and clears the conversation.
    </p>
    """, unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown("""
<div class="footer-note">
    Built using Streamlit, LangChain, Hugging Face, and FAISS
</div>
""", unsafe_allow_html=True)