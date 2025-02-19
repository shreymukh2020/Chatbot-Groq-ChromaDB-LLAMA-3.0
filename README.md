# 📚 LLAMA 3.0 Multidoc Chatbot with ChromaDB and GROQ 🤖

## 💬 Overview 

🚀 Welcome to the **LLAMA Multidoc Chatbot** project! This advanced chatbot is powered by Meta's **LLAMA 3.0** model, leveraging **Chroma DB Vector Store** for document storage and **GROQ** for powerful natural language processing. 

🚀 Groq enables Large Language Models to interact with external resources like APIs, databases, and the web to fetch real-time data and perform actions beyond basic text generation. This functionality enhances the LLM's capabilities to provide dynamic, contextually relevant responses.

🚀 The chatbot allows users to upload multiple PDF documents 📄, which are processed, indexed, and stored. Once the documents are indexed, the chatbot can interactively answer any questions based on the content of the documents. Whether you're looking to search through a large document collection or engage in interactive conversations, this tool makes document-based AI conversations easier than ever! 💬

Built with **Streamlit** 🖥️, the chatbot provides an intuitive and easy-to-use web interface, offering a seamless experience for querying and interacting with your uploaded documents.

🤖 Technical Stack:

LLAMA 3.0 Model - Meta's advanced natural language processing model.

Huggingface Embeddings - For high-quality text embeddings.

Chroma DB Vector Store - For efficient document storage and retrieval.

GROQ - For enabling dynamic, real-time interactions with external resources.

Streamlit - For building an intuitive web interface.


![image(https://github.com/shreymukh2020/Chatbot-Groq-ChromaDB-LLAMA-3.0/blob/main/App_screenshot1.jpeg)

![image](https://github.com/shreymukh2020/Chatbot-Groq-ChromaDB-LLAMA-3.0/blob/main/App_screenshot2.jpeg)

![image](https://github.com/shreymukh2020/Chatbot-Groq-ChromaDB-LLAMA-3.0/blob/main/App_screenshot3.jpeg)

---

## 📝 Table of Contents

1. [Configuration](#configuration) 
2. [Running the Application](#running-the-application) 
3. [Features in the Sidebar](#features-in-the-sidebar) 
4. [Application Flow](#application-flow) 
5. [Code Walkthrough](#code-walkthrough) 
6. [Customization](#customization) 

---

## 🛠️ Configuration

Before running the application, ensure that your environment is properly set up. This application requires the following prerequisites:

- Python 3.11 or later 
- The following Python libraries (you can install them via `pip install -r requirements.txt`):
  - `streamlit` 
  - `langchain` 
  - `huggingface-hub` 
  - `PyPDF2` 
  - `chromadb` 
  - `groq` 

### Steps to Set Up 🛠️:

1. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```

2. **Activate the Virtual Environment**:
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3. **Install Required Libraries**:
    Install all dependencies with:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure the `config.json` File**:
    Create a `config.json` file with the following content:
    ```json
    {
      "GROQ_API_KEY": "your-groq-api-key"
    }
    ```
    Replace `your-groq-api-key` with your actual **GROQ API Key** from the [GROQ Console](https://console.groq.com).

---

## 🚀 Running the Application

Ready to run the chatbot? Follow these steps:

1. **Clone or Download the Repository**:
    ```bash
    git clone "https://github.com/shreymukh2020/Chatbot-Groq-ChromaDB-LLAMA3.1.git"
    cd your-repository-name
    ```

2. **Start the Document Vectorizer File** (index your documents):
    ```bash
    python vectorize_documents.py
    ```

3. **Start the Application with Streamlit**:
    ```bash
    streamlit run main.py
    ```

4. **Open the Streamlit App** in your web browser at:
    [http://localhost:8501](http://localhost:8501) 🌐

---

## 🔧 Features in the Sidebar

The **Sidebar** of the application provides multiple features that enhance the user experience. Here’s what you’ll find:

- **🗨️ Chat History**: Clear the chat history and start fresh with a clean slate.
- **🤖 LLAMA 3.0**: The core AI model that powers document-based conversations.
- **📚 Chroma DB**: Handles document storage and retrieval based on embeddings for efficient searching.
- **🧠 GROQ Integration**: The LLM model is powered by the **GROQ API** for efficient natural language processing.
- **📂 File Upload**: Upload multiple PDF files 📄. These files will be indexed and stored in the Chroma vector database for future retrieval.

---

## 🔄 Application Flow

Here’s a step-by-step guide on how the application works:

1. **📤 Upload PDFs**: Users can upload one or more PDF files directly through the file uploader.
2. **⚙️ Process and Add to Vector Store**: The application processes the uploaded PDFs, extracts text, and adds the documents to the **Chroma** vector database as embeddings.
3. **💬 Conversational Interactions**: Once documents are indexed, users can ask questions, and the chatbot will retrieve relevant information based on the content of the documents.
4. **🧹 Chat History**: Maintain the chat history for the session, and clear it at any time through the sidebar to start fresh.

---

## 💻 Code Walkthrough

The application consists of several key components that work together to bring the chatbot to life:

### 1. **Vector Store Setup** 
   - The **Chroma** vector store is initialized to store documents as embeddings.
   - **HuggingFaceEmbeddings** is used for generating document embeddings.

### 2. **Conversation Chain** 
   - The **ChatGroq** model is loaded with **LLAMA 3.0** (specifically `llama3-70b-8192`).
   - The model interacts with the vector store to fetch relevant documents and answer questions.

### 3. **File Upload & Processing** 
   - The uploaded PDFs are processed using the **PyPDF2** library to extract text from them.
   - The extracted text is stored in the **Chroma** vector store, ready for interaction.

### 4. **Streamlit Interface** 
   - Streamlit provides the front-end interface, allowing users to upload PDFs, interact with the chatbot, and view chat history.

---

## ✨ Customization

You can customize this project based on your specific needs. Here are a few ways to make this chatbot your own:

### Changing the Model 

To switch to a different **LLM model**, modify the `model` parameter in the `ChatGroq` initialization:

```python
llm = ChatGroq(model="your-desired-model-name", temperature=0)
