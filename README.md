# YouTube Chat Bot

A Streamlit app that lets you ask questions about a YouTube video. It fetches
the video's transcript, splits it into text chunks, stores embeddings in Chroma,
and uses a Google Gemini model to answer questions from the retrieved context.

## Features

- Accepts standard YouTube watch URLs and `youtu.be` links.
- Retrieves the video's transcript and indexes it for semantic search.
- Retrieves relevant transcript chunks for each question.
- Generates an answer with Gemini and shows the retrieved context.
- Stores the vector index locally in Chroma.

## Requirements

- Python 3.10 or later
- A Google AI API key with access to the Gemini embedding and chat models used
  by this project
- A YouTube video with an available transcript

## Setup

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt langchain-chroma langchain-google-genai langchain-text-splitters python-dotenv
```

Create a `.env` file in the project directory and add your API key:

```dotenv
GOOGLE_API_KEY=your_google_ai_api_key
```

Do not commit `.env` or share its contents. If an API key has ever been
committed or exposed, revoke it with Google and create a replacement.

## Run the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. Enter a YouTube URL and a question,
then select **Get Answer**.

## How it works

The Streamlit interface in `app.py` calls the ingestion and retrieval helpers
in `data_loader_runnable.py`. The app fetches the transcript, splits it into
chunks, stores the chunks and their embeddings in Chroma, and retrieves up to
four relevant chunks for the question. The retrieved context is passed to the
chat model to produce an answer.

The current code configures `gemini-embedding-2` for embeddings and
`gemini-3.8-flash` for chat responses. Ensure these model names are available
to your Google AI API key.

## Project files

| File or directory | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface and request handling |
| `data_loader_runnable.py` | Transcript ingestion, Chroma storage, retrieval, and RAG chain |
| `data_loader.py` | Earlier standalone transcript and retrieval implementation |
| `requirements.txt` | Base Python dependencies |
| `chroma_db/` | Local persistent Chroma data |

## Notes

- `data_loader_runnable.py` currently sets the Chroma persistence directory to
  an absolute Windows path. If you run the project from a different location
  or operating system, update `persist_directory` in that file to a path on
  your machine (for example, a path relative to the project directory).
- A video's transcript must be available to the `youtube-transcript-api`
  library. Videos without an accessible transcript cannot be indexed.
- The first question for a video may take longer because the transcript must
  be fetched and embedded. The local Chroma index is reused on later requests
  for an already-indexed video.
