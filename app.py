import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import os
import zipfile
import gdown

# --- 1. INITIAL SETUP & SECURITY ---
st.set_page_config(page_title="Sharee'a شريعة AI", page_icon="🕌", layout="wide")

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API Key not found! Please add GEMINI_API_KEY to your Streamlit secrets.")
    st.stop()

client_gemini = genai.Client(api_key=API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_pdfs" not in st.session_state:
    st.session_state.current_pdfs = []
if "current_vids" not in st.session_state:
    st.session_state.current_vids = []

# --- 2. DATABASE DOWNLOAD & SETUP ---
@st.cache_resource
def setup_database():
    db_path = "./my_db"
    zip_path = "my_db.zip"
    
    if not os.path.exists(db_path):
        st.info("📥 Downloading knowledge base from Google Drive... (This happens only once)")
        file_id = "11T6mhgRjwd7EFvs1VvkE7wUS3XJGUL0M"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
            st.success("✅ Database loaded successfully!")
        except Exception as e:
            st.error(f"Failed to download or extract database: {e}")
            st.stop()

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client_db = chromadb.PersistentClient(path=db_path)
    return client_db.get_collection(name="religious_knowledge", embedding_function=embedding_func)

collection = setup_database()

# --- 3. CORE FUNCTIONS ---
def fix_arabic_for_pdf(text):
    if not text: return ""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def create_pdf(question, answer):
    pdf = FPDF()
    pdf.add_page()
    if os.path.exists("arial.ttf"):
        pdf.add_font("ArialAR", "", "arial.ttf")
        pdf.set_font("ArialAR", size=12)
    else:
        pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, txt=fix_arabic_for_pdf(f"السؤال: {question}"), align='R')
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=fix_arabic_for_pdf(f"الإجابة:\n{answer}"), align='R')
    return pdf.output()

def get_data(query, search_mode):
    pdf_context, pdf_sources = "", []
    yt_context, yt_sources = "", [] 
    CHANNELS = ["@ftawamostafaaladwy", "@fatawa_eladawy"]

    if search_mode in ["Search Hadith and Tafsir Books حديث و تفسير Only", "Hybrid (Both)"]:
        try:
            # INCREASED to 5 results to give the AI more context to work with
            results = collection.query(query_texts=[query], n_results=5)
            if results['documents'] and results['documents']:
                for d, m in zip(results['documents'], results['metadatas']):
                    s_name = m.get('source', 'Unknown Book')
                    s_page = m.get('page', 'Unknown Page') 
                    source_label = f"{s_name} (Page: {s_page})"
                    pdf_sources.append(source_label)
                    pdf_context += f"\n[BOOK SOURCE: {s_name} | PAGE: {s_page}]\n{d}\n"
        except Exception as e: 
            st.sidebar.error(f"DB Error: {e}") # Added visible error catching

    if search_mode in ["Ask Mostafa Al-Adawi", "Hybrid (Both)"]:
        for handle in CHANNELS:
            try:
                search = VideosSearch(f"{query} {handle}", limit=2)
                res = search.result().get('result', [])
                for v in res:
                    try:
                        t = YouTubeTranscriptApi.get_transcript(v['id'], languages=['ar', 'en'])
                        v_title = str(v['title'])
                        v_link = str(v['link'])
                        yt_sources.append({"title": v_title, "link": v_link})
                        yt_context += f"\n[VIDEO SOURCE: {v_title} | LINK: {v_link}]\nTranscript: {' '.join([x['text'] for x in t])[:2000]}\n"
                    except: continue
            except Exception: continue
                
    return pdf_context, yt_context, pdf_sources, yt_sources

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Room")
    mode = st.radio("Search Mode:", ["Search Hadith and Tafsir Books حديث و تفسير Only", "Ask Mostafa Al-Adawi", "Hybrid (Both)"], index=2)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.current_pdfs = []
        st.session_state.current_vids = []
        st.rerun()

    st.divider()
    st.subheader("📍 Sources Consulted:")
    
    if st.session_state.current_pdfs:
        for p in set(st.session_state.current_pdfs):
            st.write(f"📖 {p}")
            
    if st.session_state.current_vids:
        # Make sure we only show unique videos in the sidebar
        unique_vids = {v['link']: v for v in st.session_state.current_vids}.values()
        for v in unique_vids:
            st.markdown(f"🎥 [{v['title']}]({v['link']})")

# --- 5. MAIN CHAT INTERFACE ---
st.title("🕌 Sharee'a شريعة AI")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching sources..."):
            pdf_ctx, yt_ctx, pdfs, vids = get_data(prompt, mode)
            
            st.session_state.current_pdfs = pdfs
            st.session_state.current_vids = vids
            
            combined_context = pdf_ctx + yt_ctx
            
            # --- DEBUG WINDOW (Click to expand) ---
            with st.expander("🔍 Debug: View Raw Search Results"):
                st.write("**Books Found:**", pdf_ctx if pdf_ctx else "None")
                st.write("**Videos Found:**", yt_ctx if yt_ctx else "None")
            
            # Slightly relaxed instruction to prevent "I don't know" loops
            instruction = (
                "You are an expert assistant for Sheikh Mostafa Al-Adawi. "
                "CRITICAL RULES:\n"
                "1. Base your answer on the CONTEXT below. If the exact answer isn't there, provide the closest relevant information from the context.\n"
                "2. Provide a Confidence Score (0-100%).\n"
                "3. You MUST list the exact sources used at the end of your response:\n"
                "   - For Books: List the Book Name and Page Number.\n"
                "   - For Videos: List the Video Title and include the clickable Video Link.\n"
                "4. Respond naturally in the user's language."
            )
            
            response = client_gemini.models.generate_content(
                model="gemini-3-flash-preview",
                contents=f"{instruction}\n\nCONTEXT:\n{combined_context}\n\nQ: {prompt}"
            )
            
            answer_text = response.text
            st.markdown(answer_text)
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
            
            try:
                pdf_bytes = create_pdf(prompt, answer_text)
                st.download_button(label="📥 Save PDF", data=pdf_bytes, file_name="Report.pdf", mime="application/pdf")
            except: pass
            
            st.rerun()
