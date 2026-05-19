# 🎬 RAG YouTube Chatbot

A full-stack AI application that lets you chat with any YouTube video. Paste a video URL, get an instant summary, and ask follow-up questions — all powered by Retrieval-Augmented Generation (RAG).

---

## How It Works

The app fetches the transcript of a YouTube video using the YouTube Transcript API, splits it into chunks, embeds those chunks using Google Generative AI embeddings, and stores them in a FAISS vector store. When you ask a question, the most relevant transcript chunks are retrieved and passed to a Gemini LLM to generate a grounded answer.

```
YouTube URL
    │
    ▼
YouTube Transcript API
    │
    ▼
Text Chunking → Google Gemini Embeddings → FAISS Vector Store
                                                    │
                          User Question ────────────┘
                                                    │
                                            RAG Retrieval
                                                    │
                                          Gemini LLM Answer
                                                    │
                                          Streamlit UI Response
```

---

## Tech Stack

| Layer     | Technology                              |
|-----------|-----------------------------------------|
| Frontend  | Streamlit                               |
| Backend   | FastAPI + Uvicorn                       |
| AI / RAG  | LangChain, Google Gemini, FAISS         |
| Transcript| youtube-transcript-api                  |
| Config    | python-dotenv, Pydantic                 |

---

## Project Structure

```
rag-youtube-chatbot/
├── youtube_rag/
│   ├── main.py          # FastAPI app — endpoints for transcript & Q&A
│   ├── rag.py           # RAG pipeline — chunking, embedding, retrieval
│   ├── frontend.py      # Streamlit UI
│   └── ...
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/porasnehra/rag-youtube-chatbot.git
cd rag-youtube-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/).

### 5. Run the backend (FastAPI)

```bash
uvicorn youtube_rag.main:app --reload --port 8000
```

### 6. Run the frontend (Streamlit)

Open a new terminal:

```bash
streamlit run youtube_rag/frontend.py
```

The app will be available at `http://localhost:8501`.

---

## Features

- Paste any YouTube URL with a transcript/subtitles
- Auto-generates a concise video summary on load
- Ask any question about the video content
- Answers are grounded in the actual transcript (no hallucinations)
- Clean, minimal chat interface via Streamlit
- Fast retrieval using FAISS vector search

---

## API Endpoints

| Method | Endpoint      | Description                        |
|--------|---------------|------------------------------------|
| POST   | `/process`    | Process a YouTube URL and build the vector store |
| POST   | `/ask`        | Ask a question about the loaded video |

---

## Requirements

- Python 3.9+
- A Google Gemini API key (free tier available)
- Internet access (for fetching YouTube transcripts)

---

## Dependencies

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
streamlit==1.39.0
youtube-transcript-api==0.6.2
langchain==0.3.4
langchain-google-genai==2.0.1
faiss-cpu>=1.8.0
python-dotenv==1.0.1
pydantic==2.9.2
httpx==0.27.2
google-genai==0.2.2
langchain-community
```

---

## Author

**Poras Nehra**  
B.Tech Computer Science | Maharishi Markandeshwar University  
[GitHub](https://github.com/porasnehra) · [LinkedIn](https://linkedin.com/in/poras-nehra-142170367)

---

## License

This project is open source and available under the [MIT License](LICENSE).
