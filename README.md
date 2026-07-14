# 📄 AI PDF Chunk Converter

An AI-powered utility that converts PDF documents into meaningful text chunks, making them ready for AI applications such as Retrieval-Augmented Generation (RAG), semantic search, embeddings, and LLM-based chatbots.

## 🚀 Features

- 📥 Upload PDF files
- 📖 Extract text from PDF documents
- ✂️ Split text into intelligent chunks
- 🤖 AI-friendly chunking for LLM applications
- ⚡ Fast and efficient processing
- 🔄 Ready for Vector Databases and Embedding Models

## 🛠️ Tech Stack

- Python
- FastAPI
- PyPDF2 / PyMuPDF
- LangChain (optional)
- Recursive Character Text Splitter
- Uvicorn

## 📂 Project Structure

```
AI-PDF-Chunk-Converter/
│
├── app/
│   ├── main.py
│   ├── services/
│   ├── utils/
│   └── schemas/
│
├── uploads/
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AI-PDF-Chunk-Converter.git
```

Navigate to the project directory:

```bash
cd AI-PDF-Chunk-Converter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn app.main:app --reload
```

## 📌 How It Works

1. Upload a PDF document.
2. The application extracts all readable text.
3. The extracted text is cleaned.
4. The text is divided into optimized chunks using an AI-friendly text splitter.
5. The generated chunks can be directly used for:
   - RAG Applications
   - Vector Databases
   - Embedding Models
   - AI Chatbots
   - Semantic Search

## 🎯 Use Cases

- AI Chatbots
- Retrieval-Augmented Generation (RAG)
- Knowledge Base Systems
- Document Search
- LLM Applications
- Semantic Search

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

## 📜 License

This project is licensed under the MIT License.

---

⭐ If you found this project helpful, don't forget to star the repository!
