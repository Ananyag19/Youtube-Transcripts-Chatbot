import os
import traceback # Import traceback for detailed error printing
from flask import Flask, request, jsonify, render_template # Import render_template
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from flask_cors import CORS 

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, essential for web page interaction

# Global variables to store the QA chain and vector store
qa_chain_global = None
current_video_id = None
vector_store_global = None

#  Getting transcript
def fetch_transcript(video_id):
    """
    Fetches the transcript for a given YouTube video ID, prioritizing Hindi, then English.
    Handles cases where transcripts are disabled or not available in specified languages.
    """
    try:
        print(f"Attempting to fetch transcript for video ID: {video_id}")
        # Try fetching Hindi transcript first, then English
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi", "en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print("Transcript fetched successfully.")
        return transcript
    except TranscriptsDisabled:
        print(f"Error: Transcripts are disabled for video ID: {video_id}. No captions available in Hindi or English.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while fetching transcript: {e}")
        return ""

#  Chunk the transcript
def split_text(text):
    """
    Splits the given text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    if not text:
        print("No text provided for splitting. Returning empty list.")
        return []
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

# embeddings with HuggingFace
def create_vector_store(chunks):
    """
    Creates a FAISS vector store from text chunks using HuggingFace embeddings.
    """
    if not chunks:
        print("No chunks provided for vector store creation. Returning None.")
        return None
    print("Creating embeddings and building vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully.")
    return vector_store

# Retrieval and QA setup
def create_qa_chain(vector_store):
    """
    Sets up the RetrievalQA chain using the provided vector store and ChatGoogleGenerativeAI LLM.
    """
    if vector_store is None:
        print("Vector store is None. Cannot create QA chain.")
        return None

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("WARNING: GOOGLE_API_KEY environment variable is NOT set.")
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set it in your .env file or as an environment variable."
        )
    else:
        print("GOOGLE_API_KEY environment variable IS set.")

    print("Setting up QA chain...")
    prompt = PromptTemplate(
        template="""
        Use the context below to answer the question. If the context is insufficient, say you don't know.

        Context:
        {context}

        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_output_tokens=512,
            google_api_key=google_api_key # API key
        )
        print("ChatGoogleGenerativeAI LLM initialized successfully with gemini-1.5-flash.")
    except Exception as e:
        print(f"Error initializing ChatGoogleGenerativeAI: {e}")
        print(f"Detailed error (repr): {repr(e)}")
        print("Please ensure your GOOGLE_API_KEY is correct and has access to the 'gemini-1.5-flash' model.")
        print("You can check available models using the Google AI Studio ListModels API or by visiting the documentation.")
        return None

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    print("RetrievalQA chain created.")
    return chain

@app.route('/')
def index():
    """
    Serves the main HTML page for the chatbot.
    """
    #using render_template to implement index.html from the 'templates' folder
    return render_template('index.html')

@app.route('/load_video', methods=['POST'])
def load_video():
    """
    API endpoint to load a YouTube video transcript and initialize the QA chain.
    """
    global qa_chain_global, current_video_id, vector_store_global
    data = request.json
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400

    if video_id == current_video_id and qa_chain_global is not None:
        return jsonify({"message": f"Video {video_id} already loaded."}), 200

    print(f"Loading new video: {video_id}")
    text = fetch_transcript(video_id)
    if not text:
        return jsonify({"error": "Could not retrieve transcript for this video."}), 500

    chunks = split_text(text)
    if not chunks:
        return jsonify({"error": "Could not chunk transcript."}), 500

    vector_store = create_vector_store(chunks)
    if vector_store is None:
        return jsonify({"error": "Could not create vector store."}), 500

    qa_chain = create_qa_chain(vector_store)
    if qa_chain is None:
        return jsonify({"error": "Could not create QA chain."}), 500

    qa_chain_global = qa_chain
    current_video_id = video_id
    vector_store_global = vector_store # Store the vector store globally if needed for re-use or inspection
    print(f"Video {video_id} loaded and QA chain initialized.")
    return jsonify({"message": f"Video '{video_id}' transcript loaded and ready for questions."}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    API endpoint to ask a question about the loaded video.
    """
    global qa_chain_global, current_video_id
    data = request.json
    video_id = data.get('video_id')
    question = data.get('question')

    if not video_id or not question:
        return jsonify({"error": "Video ID and question are required."}), 400

    if video_id != current_video_id or qa_chain_global is None:
        # Attempt to re-load if video ID doesn't match or chain isn't initialized
        load_response, status_code = load_video()
        if status_code != 200:
            return load_response, status_code

    if qa_chain_global is None:
        return jsonify({"error": "Chatbot is not initialized. Please load a video first."}), 500

    print(f"Received question for video {video_id}: '{question}'")
    try:
        answer_dict = qa_chain_global.invoke({"query": question})
        answer = answer_dict.get("result", "No answer found.")
        print(f"Answer: {answer}")
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"An error occurred during QA chain invocation: {e}")
        print(f"Detailed error (repr): {repr(e)}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred while processing your question."}), 500

if __name__ == '__main__':

    app.run(debug=True) # debug=True allows for auto-reloading and better error messages
