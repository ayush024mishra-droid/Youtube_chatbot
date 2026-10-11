
import streamlit as st

from data_loader_runnable import (
    get_video_id,
    ingest_video,
    get_context_for_question,
    rewrite_question,
    create_chat_rag_chain,
)


st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
    layout="wide",
)

# --------------------------------------------------
# 1. Initialize session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = None

if "active_url" not in st.session_state:
    st.session_state.active_url = None


# --------------------------------------------------
# 2. Page header and video loading
# --------------------------------------------------

st.title("🎥 YouTube Chatbot")
st.caption("Ask questions and have a conversation about any supported YouTube video.")

with st.form("video_form"):
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    load_video = st.form_submit_button("Load video")

if load_video:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL.")
    else:
        try:
            new_video_id = get_video_id(youtube_url.strip())

            with st.spinner("Loading video and checking its transcript index..."):
                vector_store = ingest_video(youtube_url.strip())

            # Reset conversation only when switching videos.
            if new_video_id != st.session_state.active_video_id:
                st.session_state.messages = []

            st.session_state.vector_store = vector_store
            st.session_state.active_video_id = new_video_id
            st.session_state.active_url = youtube_url.strip()

            st.success("Video loaded successfully!")

        except Exception as e:
            st.error(f"Could not load video: {e}")


# --------------------------------------------------
# 3. Chat interface
# --------------------------------------------------

if st.session_state.active_video_id is None:
    st.info("Load a YouTube video to start chatting.")
    st.stop()

st.caption(
    f"Active video ID: {st.session_state.active_video_id}"
)

# Display existing messages after each Streamlit rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about the video...")

if question:
    # Display the user's new message immediately.
    with st.chat_message("user"):
        st.markdown(question)

    # Add it to the in-session conversation.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    # Use previous messages, excluding the current question.
    previous_messages = st.session_state.messages[:-1]

    # Keep the recent conversation for the MVP.
    history_messages = previous_messages[-10:]

    chat_history = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: "
        f"{msg['content']}"
        for msg in history_messages
    )

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Step 1: Understand follow-up questions.
                retrieval_question = rewrite_question(
                    question,
                    chat_history,
                )

                # Step 2: Retrieve chunks from this video.
                llm_context, display_context = (
                    get_context_for_question(
                        st.session_state.vector_store,
                        retrieval_question,
                        st.session_state.active_video_id,
                    )
                )

                # Step 3: Generate a contextual answer.
                chat_chain = create_chat_rag_chain()

                answer = chat_chain.invoke({
                    "context": llm_context,
                    "chat_history": chat_history,
                    "question": question,
                })

                st.markdown(answer)

        # Store the assistant's response for future turns.
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Keep retrieval debugging available.
        with st.expander("View retrieved transcript chunks"):
            st.caption(
                f"Retrieval query: {retrieval_question}"
            )
            st.text(display_context)

    except Exception as e:
        st.error(f"Could not generate an answer: {e}")
