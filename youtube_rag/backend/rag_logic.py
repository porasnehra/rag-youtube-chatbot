import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load .env from D:\langchane\.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "vector_stores")

def extract_video_id(url: str) -> str:
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def ingest_youtube_video(url: str) -> dict:
    video_id = extract_video_id(url)
    if not video_id:
        return {"status": "error", "message": "Invalid YouTube URL"}
    
    store_dir = os.path.join(VECTOR_STORE_PATH, video_id)
    if os.path.exists(store_dir):
        return {"status": "success", "message": "Video already ingested.", "video_id": video_id}
        
    try:
        from langchain_community.document_loaders import YoutubeLoader
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["en", "en-US", "en-GB", "en-CA", "en-IN"])
        docs = loader.load()
    except Exception as e:
        return {"status": "error", "message": f"Error fetching transcript: {str(e)}"}

    if not docs:
        return {"status": "error", "message": "No captions available or failed to load video."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_documents(docs)

    if len(chunks) > 80:
        chunks = chunks[:80]
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        os.makedirs(store_dir, exist_ok=True)
        vector_store.save_local(store_dir)
    except Exception as e:
        return {"status": "error", "message": f"Error generating embeddings or storing: {str(e)}"}
    
    return {"status": "success", "message": "Video ingested successfully.", "video_id": video_id}

def ask_question(query: str, video_id: str) -> dict:
    store_dir = os.path.join(VECTOR_STORE_PATH, video_id)
    if not os.path.exists(store_dir):
        return {"status": "error", "message": "Video not ingested yet."}
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": query})
        
        return {"status": "success", "answer": response["answer"]}
    except Exception as e:
        return {"status": "error", "message": f"Error during question answering: {str(e)}"}
