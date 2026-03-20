import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="YouTube RAG Platform", layout="centered", page_icon="📺")

st.title("📺 YouTube Video Chat RAG")

if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for ingestion
with st.sidebar:
    st.header("Ingest Video")
    url_input = st.text_input("YouTube URL")
    
    if st.button("Ingest"):
        if url_input:
            with st.spinner("Processing video..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/ingest", json={"url": url_input})
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.video_id = data["video_id"]
                        
                        # Reset chat on new video
                        st.session_state.messages = []
                        
                        st.success(data["message"])
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.warning("Please enter a URL first.")
            
if st.session_state.video_id:
    st.markdown(f"**Current Video ID:** `{st.session_state.video_id}`")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the video"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/ask", 
                        json={"query": prompt, "video_id": st.session_state.video_id}
                    )
                    if res.status_code == 200:
                        answer = res.json().get("answer", "No answer found.")
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        err = res.json().get("detail", "Error fetching answer")
                        st.error(err)
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
else:
    st.info("👈 Please ingest a YouTube video from the sidebar to start chatting.")
