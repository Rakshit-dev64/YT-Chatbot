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

# Main Streamlit App

def main():
    api_key = load_api_key()
    
    st.set_page_config(page_title="YT Video Chatbot", layout="wide")
    st.title("Ask Questions to a YouTube Video 💬")
    
    # Get user input for the video
    youtube_url = st.text_input("Enter the YouTube Video URL:", placeholder="https.youtube.com/watch?v=...")

    if st.button("Process Video"):
        if not youtube_url:
            st.warning("Please enter a YouTube URL.")
        else:
            video_id = get_video_id(youtube_url)
            
            if video_id:
                with st.spinner("Processing video... This may take a moment."):
                    try:
                        # Clearing any old chain from memory
                        if "rag_chain" in st.session_state:
                            del st.session_state.rag_chain

                        # 1. Get Transcript
                        raw_text = get_transcript_text(video_id)
                        
                        if raw_text:
                            # 2. Split into Chunks
                            text_chunks = get_text_chunks(raw_text)
                            
                            # 3. Create Vector Store
                            vector_store = get_vector_store(text_chunks, api_key)
                            
                            # 4. Create the RAG chain and save to session state
                            st.session_state.rag_chain = create_rag_chain(vector_store, api_key)
                            
                            st.success("Video processed! You can now ask questions below.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    
    st.divider() # Adds a horizontal line

    # --- This is the new "below" section ---
    
    # 2. Get user's question
    user_question = st.text_input("Ask a question about the video:")

    if st.button("Get Answer"):
        if "rag_chain" not in st.session_state:
            st.warning("You must process a video first.")
        elif not user_question:
            st.warning("Please enter a question.")
        else:
            # 3. Use the chain to get an answer
            with st.spinner("Thinking..."):
                try:
                    # Retrieve the chain from memory
                    chain = st.session_state.rag_chain
                    
                    # Invoke the chain with the user's question
                    result = chain.invoke(user_question)
                    
                    # 4. Display the answer
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")

if __name__ == "__main__":
    main()