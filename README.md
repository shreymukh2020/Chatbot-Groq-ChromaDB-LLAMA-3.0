# LLAMA Multidoc Chatbot with Chroma and GROQ

## Overview

This project implements an advanced, multi-document chatbot powered by **LLAMA 3.0** using **Chroma DB** for document storage and **GROQ** for advanced natural language processing. 
It allows users to upload multiple PDF documents, which are processed, indexed, and then used to answer questions interactively. The project is implemented with **Streamlit** for a user-friendly web interface.

## Table of Contents

1. [Configuration](#configuration)
2. [Running the Application](#running-the-application)
3. [Features in the Sidebar](#features-in-the-sidebar)
4. [Application Flow](#application-flow)
5. [Code Walkthrough](#code-walkthrough)
6. [Customization](#customization)

---

## Configuration

Before running the application, ensure that your environment is properly set up. The application requires:

- Python 3.8 or later
- The following Python libraries:
  - `streamlit`
  - `langchain`
  - `huggingface-hub`
  - `PyPDF2`
  - `uuid`
  - `chromadb`
  - `groq`

### Steps to Set Up:

1. Create a virtual environment:
 ```bash
   python -m venv venv
 ```

2. Activate the virtual environment
```bash
.\venv\Scripts\activate

```

3. Install the required libraries using the following command:

    ```bash
   pip install -r requirements.txt
    ```

4. Configure the `config.json` file with the following content:

    ```json
    {
      "GROQ_API_KEY": "your-groq-api-key"
    }
    ```
   Replace `your-groq-api-key` with your actual **GROQ API Key** from [GROQ Console](https://console.groq.com).

---

## Running the Application

1. Clone or download the repository:

   ```bash
    git clone "https://github.com/shreymukh2020/Chatbot-Groq-ChromaDB-LLAMA3.1.git"
    
    cd your-repository-name
    ```

2. Start the document vectorizer file:
    ```bash
    python vectorize_documents.py

    ```

3. Start the application using Streamlit:
    ```bash
    streamlit run main.py
    ```

4. Open the Streamlit app in your web browser at `http://localhost:8501`.

---

## Features in the Sidebar

- **Chat History**: A feature to clear the chat history and start fresh.
- **LLAMA 3.1**: The core AI model used for document-based conversation.
- **Chroma DB**: Handles document storage and retrieval based on embeddings.
- **GROQ Integration**: The LLM model is powered by the GROQ API.
- **File Upload**: Users can upload multiple PDF files which will be indexed and stored in the Chroma database for future reference.

---

## Application Flow

1. **Upload PDFs**: Users upload multiple PDF files through the file uploader.
2. **Process and Add to Vector Store**: The application processes the uploaded PDFs, extracts text, and stores the documents in a Chroma vector database.
3. **Conversational Interactions**: Users can ask questions, and the chatbot uses the stored documents to provide relevant answers.
4. **Chat History**: The chat history is maintained for the session and can be cleared anytime through the sidebar.

---

## Code Walkthrough

### Setup

The application initializes with the necessary configurations, such as API keys, vector stores, and model initialization. Here's an overview of the components:

1. **Vector Store Setup**:
    - The `Chroma` vector store is initialized to store documents as embeddings.
    - It uses `HuggingFaceEmbeddings` for embedding generation.

2. **Conversation Chain**:
    - The `ChatGroq` model is loaded with `llama3-70b-8192`.
    - It interacts with the vector store to retrieve relevant documents and provide responses.

3. **File Upload & Processing**:
    - The uploaded PDFs are processed using the `PyPDF2` library, and the text is extracted and stored in the Chroma vector store.

4. **Streamlit Interface**:
    - Streamlit provides a simple interface to upload PDFs, view chat history, and interact with the chatbot.

---

## Customization

### Changing the Model

To switch to a different LLM model, change the `model` parameter in the `ChatGroq` initialization:

```python
llm = ChatGroq(model="your-desired-model-name", temperature=0)
