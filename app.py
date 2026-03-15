import streamlit as st
import os
import sys

# --- STEP 1: SQLITE FIX FOR STREAMLIT CLOUD ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import chromadb
from chromadb.utils import embedding_functions
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 2. INITIAL SETUP ---
st.set_page_config(page_title="Haikal AI", page_icon="🕌", layout="wide")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("Please add 'GEMINI_API_KEY' to Streamlit Secrets.")
    st.stop()

# Initialize Client with the 2026 SDK
client_gemini = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = {"pdfs": [], "vids": []}

# --- 3. DATABASE & BATCH INGESTION ---
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

@st.cache_resource
def load_and_ingest_db():
    client_db = chromadb.PersistentClient(path="./my_db")
    collection = client_db.get_or_create_collection(
        name="religious_knowledge", 
        embedding_function=embedding_func
    )

    if collection.count() == 0 and os.path.exists("./knowledge"):
        st.info("🚀 Processing 1GB Library... (First-time setup)")
        loader = PyPDFDirectoryLoader("./knowledge", recursive=True)
        docs = loader.load()
        
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Batching to prevent memory errors
            batch_size = 100
            progress_bar = st.progress(0)
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                collection.add(
                    ids=[f"id_{j}" for j in range(i, i + len(batch))],
                    documents=[chunk.page_content for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch]
                )
                progress_bar.progress(min(100, int((i + batch_size) / len(chunks) * 100)))
            st.success("✅ Database Built!")
    return collection

collection = load_and_ingest_db()

# --- 4. CORE UTILITIES ---

def fix_arabic(text):
    if not text: return ""
    return get_display(arabic_reshaper.reshape(text))

def get_data(query, search_mode):
    context, pdf_sources, yt_sources = "", [], []
    
    if search_mode in ["Search Local Books Only", "Hybrid (Both)"]:
        try:
            # Limit results to 3 to stay within API token limits
            res = collection.query(query_texts=[query], n_results=3)
            for d, m, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
                conf = round((1 - dist) * 100)
                s_name = os.path.basename(m.get('source', 'Reference'))
                pdf_sources.append(f"{s_name} ({conf}%)")
                context += f"\n[SOURCE: {s_name}]\n{d}\n"
        except: pass

    if search_mode in ["Ask Mostafa Al-Adawi", "Hybrid (Both)"]:
        try:
            search = VideosSearch(f"{query} @ftawamostafaaladwy", limit=1)
            for v in search.result().get('result', []):
                try:
                    t = YouTubeTranscriptApi.get_transcript(v['id'], languages=['ar', 'en'])
                    yt_sources.append({"title": v['title'], "link": v['link']})
                    context += f"\n[VIDEO: {v['title']}]\n{' '.join([x['text'] for x in t])[:1000]}\n"
                except: continue
        except: pass
                
    return context, pdf_sources, yt_sources

# --- 5. CHAT UI ---
st.title("🕌 Haikal AI - Scholarly Assistant")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Search Mode:", ["Search Local Books Only", "Ask Mostafa Al-Adawi", "Hybrid (Both)"], index=2)
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting library..."):
            context, pdfs, vids = get_data(prompt, mode)
            st.session_state.current_sources = {"pdfs": pdfs, "vids": vids}
            
            # --- UPDATED GENERATION LOGIC WITH FALLBACK ---
            try:
                # Primary model (Fast and Smart)
                target_model = "gemini-3-flash-preview"
                response = client_gemini.models.generate_content(
                    model=target_model,
                    contents=f"Context:\n{context}\n\nQuestion: {prompt}"
                )
            except Exception as e:
                # Fallback model (Reliable)
                st.warning("Switching to secondary engine...")
                target_model = "gemini-2.5-flash"
                response = client_gemini.models.generate_content(
                    model=target_model,
                    contents=f"Context:\n{context}\n\nQuestion: {prompt}"
                )
            
            answer = response.text
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

# Update Sidebar
with st.sidebar:
    st.divider()
    st.subheader("📚 Sources:")
    for p in set(st.session_state.current_sources["pdfs"]): st.write(f"📖 {p}")
    for v in st.session_state.current_sources["vids"]: st.markdown(f"🎥 [{v['title']}]({v['link']})")
