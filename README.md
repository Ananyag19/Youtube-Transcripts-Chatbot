# 🤖 YouTube Chatbot using RAG (Retrieval-Augmented Generation)

This project is a YouTube-powered chatbot that allows users to ask questions based on the transcript of any YouTube video. It uses RAG (Retrieval-Augmented Generation) to combine the power of LangChain,
FAISS, Hugging Face embeddings, and Google's Gemini (via LangChain) to generate intelligent, context-aware answers.

## 🚀 Features

- 🔗 Input any YouTube video link (with captions enabled)
- 📄 Automatically fetches and processes the transcript
- 🧠 Breaks text into chunks and embeds them using `sentence-transformers`
- 🔍 Creates a FAISS vector store for fast retrieval
- 🤖 Uses `gemini-1.5-flash` via `langchain-google-genai` for answering queries
- 🗂️ HTML frontend support (via `templates/index.html`)

## 🧰 Tech Stack

| Area        | Tools & Libraries |
|-------------|-------------------|
| Language    | Python |
| Embeddings  | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM         | Google Gemini via `langchain-google-genai` |
| Retrieval   | FAISS vector store |
| Pipeline    | LangChain |
| Transcript  | YouTubeTranscriptAPI |
| UI (optional) | HTML (Flask-ready) |

## 📁 Project Structure

Youtube-Transcripts-Chatbot/
├── .venv/ # Virtual environment
├── templates/
│ └── index.html # UI file (Flask or other backend)
├── .env # Contains your API key 
├── .gitignore # Ignores .venv
├── app.py # Main Python script
├── requirements.txt # Python dependencies

## 🔐 Environment Setup

Create a `.env` file in the root directory and add your Google API key:

env
# .env
# Get your API key here: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

⚙️ Installation & Usage
Clone the repository:
git clone https://github.com/Ananyag19/Youtube-Transcripts-Chatbot.git
cd Youtube-Transcripts-Chatbot
Install dependencies:

pip install -r requirements.txt
Set your API key in .env

Run the chatbot:
python app.py

Ask a question!
The script will:
Fetch the transcript
Create vector embeddings
Generate a response based on your query

💬 Example
<img width="1919" height="869" alt="Image" src="https://github.com/user-attachments/assets/856fa302-7396-41a6-9d90-4f7ac23c2e76" />

<img width="1870" height="843" alt="Image" src="https://github.com/user-attachments/assets/c0d68a3c-f854-4d48-a7a9-7105134983e6" />

