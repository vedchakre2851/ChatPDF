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

st.set_page_config(page_title="ChatPDF", page_icon="📄", layout="wide")
st.title("ChatPDF")


# For parsing String outuput
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

# ------------------- PDF Upload -------------------
pdf_input = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_input is not None:
    os.makedirs("data/uploads", exist_ok=True)
    file_path = os.path.join("data/uploads", pdf_input.name)

    with open(file_path, "wb") as f:
        f.write(pdf_input.getbuffer())

    if st.session_state["current_pdf"] != pdf_input.name:
        st.info("Processing new PDF...")

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        extracted_docs = splitter.split_documents(docs)

        st.write(f"Total chunks created: {len(extracted_docs)}")

        vector_store = FAISS.from_documents(extracted_docs, embedding_model)

        st.session_state['current_pdf'] = pdf_input.name
        st.session_state['extracted_docs'] = extracted_docs
        st.session_state['vector_store'] = vector_store

        st.success("PDF processed successfully!")
    else:
        st.info("Using already processed PDF.")

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

Don't make up answers on your own.

Question: {question}

Context: {context}
""",
    input_variables=["context", "question"]
)

# ------------------- Helper Function -------------------
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# ------------------- Chains -------------------
final_chain = None

if retriever is not None:
    rag_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    final_chain = rag_chain | prompt | model | parser

# ------------------- Display Old Chat -------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------- User Input -------------------
user_question = st.chat_input("Ask anything about the uploaded PDF!")

if user_question:
    if retriever is None or final_chain is None:
        st.warning("Please upload a PDF first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = final_chain.invoke(user_question)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# ------------------- Clear Chat -------------------
if st.button("🗑️ Reset Chat"):
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

    st.success("Chat and uploaded PDF cleared!")
    st.rerun()