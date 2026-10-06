import streamlit as st
from data_loader_runnable import (
    create_rag_chain,
    get_context_for_question,
    get_video_id,
    ingest_video,
)

if "chain" not in st.session_state:
    st.session_state.chain = None



if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.title("🎥 YouTube RAG")

st.write("Ask questions about YouTube videos.")

youtube_url = st.text_input(
    "Enter YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

question = st.text_input(
    "Ask a question about the video",
    placeholder="What is the video about?",
)

if st.button("Get Answer"):
    if youtube_url and question:
        try:
           st.write("Processing the video and building the context. Please wait...")
           video_id = get_video_id(youtube_url)
           vector_store = ingest_video(youtube_url)
           chain = create_rag_chain(vector_store, video_id)
           context = get_context_for_question(vector_store, question, video_id)
           
                       
           answer = chain.invoke(question)
           
           st.write("Context:", context)
           st.write("Answer:", answer) 
                
        except Exception as e:
            st.error(f"Error: {e}")
