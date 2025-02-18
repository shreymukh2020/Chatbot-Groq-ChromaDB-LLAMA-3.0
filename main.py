import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import uuid  # To generate unique IDs for documents
from langchain.schema import Document  # Correct Document class import

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Set up the vector store
def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

# Set up the conversation chain
def chat_chain(vectorstore):
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    return chain

# Process the PDF and extract text
def process_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Page config for Streamlit
st.set_page_config(
    page_title="LLAMA Multidoc Chatbot",
    page_icon="📚",
    layout="centered",
)

# Main Title with Icon
st.title("📚 LLAMA Multidoc Chatbot 🤖")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

if "documents" not in st.session_state:
    st.session_state.documents = {}

# Sidebar customization
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #1E3A5F; /* Dark Navy Blue */
        }

        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] div {
            color: white; /* Ensure all text in the sidebar is white */
        }

        /* Style the Clear Chat History Button */
        .stButton>button {
            background-color: #f44336; /* Red background for visibility */
            color: white; /* White text */
            font-size: 16px; /* Set font size */
            padding: 10px 20px; /* Set padding for better appearance */
            border-radius: 5px; /* Rounded corners */
            border: none; /* Remove border */
            cursor: pointer; /* Pointer cursor on hover */
        }

        .stButton>button:hover {
            background-color: #d32f2f; /* Darker red when hovering */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with components and cool icons
with st.sidebar:
    st.title("📚 LLAMA Multidoc Chatbot 🤖")
    st.markdown("### Components:")
    st.markdown("🤖 **LLAMA 3.1**")
    st.markdown("💾 **Chroma DB**")
    st.markdown("🔗 **GROQ**")

    # Button to clear chat history with improved styling
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared! Start fresh!")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

# Process uploaded PDFs and add to the vector store
if uploaded_files:
    for uploaded_file in uploaded_files:
        text = process_pdf(uploaded_file)
        
        # Add documents to the vector store
        if text:
            st.session_state.documents[uploaded_file.name] = text
            embeddings = HuggingFaceEmbeddings()
            vectorstore = st.session_state.vectorstore
            # Convert the text into embeddings and add to vector store as a list of documents
            document_id = str(uuid.uuid4())  # Generate a unique ID for each document
            doc = Document(page_content=text, metadata={"id": document_id})  # Correct Document format
            vectorstore.add_documents([doc])  # Now passing as a list of Document objects
            st.success(f"Added {uploaded_file.name} to the vector store.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and chatbot response logic
user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from the chatbot
    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
