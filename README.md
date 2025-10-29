
# YT_ChatBot

An interactive RAG pipeline that lets you ask questions, get summaries, and find information in any YouTube video instantly, powered by Langchain and Gemini 2.5 Flash.

## ✨ Features
 - Chat with Videos: Paste any YouTube link to start a conversation.

- Get Summaries: Ask for a quick summary of the entire video.

- Ask Specific Questions: Find specific details, definitions, or topics discussed in the video.

- Simple UI: Clean and easy-to-use interface built with Streamlit.

## ⚙️ How It Works (Architecture)
This project is a complete RAG pipeline. Here's the step-by-step data flow:

1. Transcript Fetching: The app takes a YouTube URL and uses the youtube-transcript-api to extract the full video transcript.

2. Chunking: The transcript is broken down by  Recursive- CharacterTextSplitter into smaller, manageable text chunks.

3. Embedding: Each chunk is converted into a numerical vector representation (an embedding) using the "gemini-embedding-001" model.

4. Vector Store: These embeddings are stored and indexed in a local vector store (using FAISS) for efficient, high-speed searching.

5. Retrieval: When you ask a question, your query is also embedded. The app then performs a similarity search (using cosine similarity) in the vector store to find the most relevant text chunks from the video.

6. Generation: These relevant chunks are passed as context, along with your original question, to the Gemini 2.5 Flash LLM. The LLM processes all the tokens and generates a comprehensive, context-aware answer.

## 🛠️ Tech Stack
- Frontend: Streamlit

- Orchestration: Langchain

- LLM: Google Gemini 2.5 Flash (gemini-2.5-flash)

- Embedding Model: Google Gemini Embedding 001 (gemini-embedding-001)

- Vector Store: FAISS (Facebook AI Similarity Search)

- Key Libraries: google-generativeai, langchain-google-genai, youtube-transcript-api, python-dotenv

## 🚀 Getting Started
Follow these instructions to get a copy of the project running on your local machine.

### Prerequisites
- Python 3.9 or later

- A Google Gemini API Key. You can get one from Google AI Studio.

### Installation & Setup
1. Clone the repository:

```bash
git clone https://github.com/Rakshit-dev64/YT-Chatbot.git
cd YT-Chatbot
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```
3. Install the required dependencies: (It's recommended to create a requirements.txt file with the libraries below)
```bash
pip install streamlit langchain langchain-google-genai google-generativeai faiss-cpu youtube-transcript-api python-dotenv
```
4. Set up your environment variables: Create a file named .env in the root of your project directory and add your API key:
```bash
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```
Your Python script (e.g., app.py) should use python-dotenv to load this key:
```bash
from dotenv import load_dotenv
load_dotenv()
```

### Usage
1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Paste a YouTube video URL into the first text box and click "Process Video".

Once it's processed, ask your questions in the second text box and click "Get Answer".
