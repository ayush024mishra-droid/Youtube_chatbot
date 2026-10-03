from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Load from .env file

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

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


vid = get_video_id("https://www.youtube.com/watch?v=RgV57kDzcng&t=1s")

transcript = get_transcript(vid)

print("Transcript:", transcript[:500])  # Print the first 500 characters of the transcript

# transcript_chunks = text_splitter.create_documents([transcript])

vector_store = get_vector_store(embedding_model, persist_directory="chroma_db", collection_name="youtube_transcripts")

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="You are a helpful assistant. Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

def build_context(res):
    """
    Builds a context string from the retrieved documents.

    Args:
        res: The retrieved documents.
    Returns:
        str: The concatenated context string.
    """
    return "\n\n".join([doc.page_content for doc in res])

chain = {
    "context": retriever | RunnableLambda(build_context),
    "question": RunnablePassthrough(),
} | prompt | llm | StrOutputParser()

res = chain.invoke("What backlash is AI facing ?")

print("Answer:", res)