from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
from pathlib import Path
import streamlit as st

RETRIEVAL_TOP_K = 20
LLM_CONTEXT_TOP_K = 12
UI_DISPLAY_TOP_K = 4
RETRIEVAL_LOG_PATH = Path(__file__).with_name("retrieved_chunks.json")

load_dotenv()
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]  # Load from .env file
import re

def clean_transcript(transcript: str) -> str:
    transcript = re.sub(r"\[music\]", " ", transcript, flags=re.IGNORECASE)
    transcript = transcript.replace(">>", " ")
    return re.sub(r"\s+", " ", transcript).strip()

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def define_chunks(text, chunk_size=600, sentence_overlap=2):
    chunks = []
    current_chunk = []
    sentences = split_into_sentences(text)

    for sentence in sentences:
        current_sentence = "".join(current_chunk + [sentence])
        if len(current_sentence) <= chunk_size:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-sentence_overlap:] + [sentence]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        " ",
        ""
    ]
)

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

def create_vector_store(embedding_model, documents, persist_directory: str, collection_name: str):
    """
    Creates a vector store using the provided embedding model and documents.

    Args:
        embedding_model: The embedding model to use for creating embeddings.
        documents: The documents to be stored in the vector store.
        persist_directory (str): The directory where the vector store will be persisted.
        collection_name (str): The name of the collection in the vector store.
    Returns:
        Chroma: The created vector store.
    """

    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents = documents,
        embedding = embedding_model,
        collection_name = collection_name,
        persist_directory = persist_directory
    )
    return vector_store

def get_vector_store(embedding, persist_directory: str, collection_name: str):
    """
    Retrieves an existing vector store from the specified directory.

    Args:
        embedding: The embedding model to use for retrieving embeddings.
        persist_directory (str): The directory where the vector store is persisted.
        collection_name (str): The name of the collection in the vector store.
    Returns:
        Chroma: The retrieved vector store.
    """

    vector_store = Chroma(
        embedding_function = embedding,
        persist_directory = persist_directory,
        collection_name = collection_name
    )

    return vector_store

def get_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.

    Args:
        url (str): The YouTube video URL.

    Returns:
        str: The extracted video ID.
    """
    parsed_url = urlparse(url)

    # Normal YouTube URL: youtube.com/watch?v=VIDEO_ID
    if parsed_url.path == "/watch":
        return parse_qs(parsed_url.query)["v"][0]

    # Short URL: youtu.be/VIDEO_ID
    if parsed_url.netloc == "youtu.be":
        return parsed_url.path.strip("/")

    raise ValueError("Invalid or unsupported YouTube URL")

def get_transcript(video_id: str) -> str:
    """
    Fetches the transcript for a given YouTube video ID.

    Args:
        video_id (str): The YouTube video ID.
    Returns:
        str: The transcript text.
    """
    video_id = str(video_id).strip()

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        return " ".join([entry.text for entry in transcript])
    except Exception as e:
        try:
            transcript = YouTubeTranscriptApi().fetch(video_id)
            return " ".join([entry.text for entry in transcript])
        except Exception as inner_error:
            print(f"Error fetching transcript for video ID {video_id}: {inner_error}")
            return ""

def _video_already_indexed(vector_store, video_id: str, url: str) -> bool:
    """Return True if this video has already been indexed in the Chroma collection."""
    try:
        collection = getattr(vector_store, "_collection", None)
        if collection is None:
            return False

        for key, value in (("video_id", video_id), ("url", url)):
            try:
                result = collection.get(where={key: value}, include=["metadatas"])
                if result and result.get("ids"):
                    return True
            except Exception:
                pass
        return False
    except Exception:
        return False

from collections import Counter

def report_duplicate_chunks(vector_store, video_id: str):
    result = vector_store._collection.get(
        where={"video_id": video_id},
        include=["documents", "metadatas"],
    )

    counts = Counter(result["documents"])
    duplicates = [(text, count) for text, count in counts.items() if count > 1]

    print(f"Records for video {video_id}: {len(result['ids'])}")
    print(f"Distinct chunk texts: {len(counts)}")
    print(f"Chunk texts stored more than once: {len(duplicates)}")

    for text, count in duplicates:
        print(f"\nStored {count} times: {text[:200]!r}")

def ingest_video(url):
    video_id = get_video_id(url)
    persist_directory = r"C:\Users\Lenovo\Desktop\Rag\youtube_chat_bot\chroma_db"
    collection_name = "youtube_transcripts"

    vector_store = get_vector_store(
        embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    if _video_already_indexed(vector_store, video_id, url):
        print(f"Video {video_id} already indexed. Skipping ingestion.")
        report_duplicate_chunks(vector_store, video_id)
        return vector_store

    transcript = clean_transcript(get_transcript(video_id))
    if not transcript:
        raise ValueError(f"No transcript found for video ID: {video_id}")

    documents = [Document(page_content=chunk) for chunk in define_chunks(transcript)]
    for doc in documents:
        doc.metadata["video_id"] = video_id
        doc.metadata["url"] = url

    vector_store = create_vector_store(
        embedding_model,
        documents,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    report_duplicate_chunks(vector_store, video_id)

    return vector_store



def create_question_rewriter_chain():
    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
You rewrite follow-up questions into standalone questions
for retrieving information from a YouTube transcript.

Use the conversation history only to resolve references
such as "it", "that", "the second point", or "why".

Rules:
- Preserve the user's actual intent.
- Do not answer the question.
- Do not invent details.
- If the question is already standalone, preserve its meaning.
- Return only the rewritten question.

Conversation history:
{chat_history}

Latest user question:
{question}

Standalone retrieval question:
"""
    )

    return prompt | llm | StrOutputParser()


def rewrite_question(question, chat_history):
    if not chat_history.strip():
        return question

    chain = create_question_rewriter_chain()

    rewritten = chain.invoke({
        "chat_history": chat_history,
        "question": question,
    })

    return rewritten.strip() or question



def create_chat_rag_chain():
    prompt = PromptTemplate(
        input_variables=[
            "context",
            "chat_history",
            "question",
        ],
        template="""
You are a helpful assistant that answers questions
about a YouTube video.

Use the retrieved transcript context as your primary
source of factual information.

Rules:
- Answer the user's actual question.
- Use conversation history to understand follow-ups.
- Ground factual claims in the provided transcript context.
- Do not invent facts or claim the speaker said something
  that is not supported by the context.
- If the transcript context is insufficient, say so.
- You may use conversation history to resolve references,
  but do not treat earlier assistant answers as proof
  of facts not supported by the transcript.
- Be natural, concise, and conversational.
- Do not repeat the entire previous answer unnecessarily.

Conversation history:
{chat_history}

Retrieved transcript context:
{context}

Latest user question:
{question}

Answer:
"""
    )

    return prompt | llm | StrOutputParser()



# vid = get_video_id("https://www.youtube.com/watch?v=RgV57kDzcng&t=1s")

# transcript = get_transcript(vid)

# print("Transcript:", transcript[:500])  # Print the first 500 characters of the transcript

# transcript_chunks = text_splitter.create_documents([transcript])



def build_context(res):
    """
    Builds a context string from the retrieved documents.

    Args:
        res: The retrieved documents.
    Returns:
        str: The concatenated context string.
    """
    return "\n\n".join([doc.page_content for doc in res])

def create_rag_chain():
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Use the following context to answer the question.Do not go outside the context specified and do not invent new facts. Always stay grounded to the provided context. If you are making any inference not in the context, state it explicitely.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_context_for_question(vector_store, question: str, video_id: str):
    scored_docs = vector_store.similarity_search_with_score(
        question,
        k=RETRIEVAL_TOP_K,
        filter={"video_id": video_id},
    )

    log_entry = {
        "question": question,
        "chunklist": [
            {"chunk": doc.page_content, "score": float(score)}
            for doc, score in scored_docs
        ],
    }
    if RETRIEVAL_LOG_PATH.exists():
        with RETRIEVAL_LOG_PATH.open("r", encoding="utf-8") as log_file:
            retrieval_log = json.load(log_file)
        if not isinstance(retrieval_log, list):
            raise ValueError(f"Expected a JSON list in {RETRIEVAL_LOG_PATH}")
    else:
        retrieval_log = []
    retrieval_log.append(log_entry)
    temporary_log_path = RETRIEVAL_LOG_PATH.with_suffix(".json.tmp")
    with temporary_log_path.open("w", encoding="utf-8") as log_file:
        json.dump(retrieval_log, log_file, ensure_ascii=False, indent=2)
    temporary_log_path.replace(RETRIEVAL_LOG_PATH)

    for index, (doc, score) in enumerate(scored_docs, start=1):
        print(f"Retrieved chunk {index} (distance: {score:.6f}): {doc.page_content}")

    display_docs = [doc for doc, _ in scored_docs[:UI_DISPLAY_TOP_K]]
    llm_docs = [doc for doc, _ in scored_docs[:LLM_CONTEXT_TOP_K]]
    return build_context(llm_docs), build_context(display_docs)