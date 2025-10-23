import streamlit as st
import os

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Helper Functions

def load_api_key():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in .env file. Please add it.")
        st.stop()
    return api_key

def get_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    st.error("Invalid YouTube URL. Please use a 'watch?v=' or 'youtu.be/' URL.")
    return None

def get_transcript_text(video_id):
    try:
        transcript_snippets = YouTubeTranscriptApi().fetch(video_id, languages=['hi'])
        text = " ".join(snippet.text for snippet in transcript_snippets)
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Could not retrieve transcript: {e}")
        return None

def get_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=100)
    chunks = splitter.create_documents([raw_text])
    return chunks

def get_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_rag_chain(vector_store, api_key):
    """
    Creates the full RAG (Retrieval-Augmented Generation) chain.
    """
    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)

    # 2. Create the Retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3. Create the Prompt
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Answer only from provided transcript context. If the context is insufficient, just say you dont know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
    )

    # 4. format retrieved documents
    def format_documents(retrieved_docs):
        return "\n".join(doc.page_content for doc in retrieved_docs)

    # 5. Build the parallel chain
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_documents),
        "question": RunnablePassthrough()
    })

    # 6. Create the final chain
    parser = StrOutputParser()
    final_chain = parallel_chain | prompt | llm | parser
    
    return final_chain

