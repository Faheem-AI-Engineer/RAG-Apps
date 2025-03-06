# AskPDF - PDF Question Answering System

AskPDF is a Retrieval-Augmented Generation (RAG)-based application that enables users to ask questions about PDF documents using hybrid search (dense + sparse embeddings). It leverages Pinecone for vector search and Mistral-7B for intelligent responses.

## 🚀 Features

- 📄 **Upload & Chat**: Upload PDF documents and interact with their content.
- 🔍 **Hybrid Search**: Combines semantic (dense) and keyword-based (sparse) retrieval for accurate information retrieval.
- 🧠 **LLM-Powered**: Uses Mistral-7B LLM for generating intelligent answers.
- 🎨 **User-Friendly UI**: Built with Streamlit for an intuitive experience.
- 📈 **Efficient Processing**: Automatic text chunking with semantic splitting for optimal search performance.

## 🛠️ Tech Stack

- **Framework**: LangChain
- **Vector Database**: Pinecone
- **LLM**: Mistral-7B-Instruct-v0.2 (via Hugging Face Serverless API)
- **Embeddings Model**: `BAAI/bge-base-en-v1.5`
- **User Interface**: Streamlit
- **Text Processing**: NLTK, Semantic Chunking

## 📦 Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Faheem-AI-Engineer/RAG-Apps.git
cd Hybrid_Search_RAG_App
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory and add your API keys:
```
PINECONE_ONE_TIME_KEY=your_pinecone_api_key
HF_TOKEN=your_huggingface_token
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 🚀 How to Use
1. Upload a PDF document via the web interface.
2. Ask questions about its content.
3. The first upload will take longer due to processing and indexing, but subsequent queries will be much faster.

## 📂 Project Structure
```
├── app.py                # Main Streamlit application
├── config.py             # Environment configuration
├── retriever.py          # Pinecone hybrid search setup
├── document_processor.py # PDF processing utilities
├── llm_chain.py          # LLM and RAG chain configuration
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 📝 License
This project is open-source and available under the MIT License.

---

💡 **Contributions are welcome!** If you have any suggestions or improvements, feel free to submit a pull request.

