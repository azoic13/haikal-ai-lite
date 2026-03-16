# --- STEP 0: THE SQLITE FIX (MUST BE AT THE VERY TOP) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import streamlit as st
import streamlit_analytics2 as streamlit_analytics
import chromadb
from chromadb.utils import embedding_functions
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai import types
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import os

# --- 1. INITIAL SETUP & ANALYTICS WRAPPER ---
with streamlit_analytics.track(save_to_json="./analytics.json", unsafe_password="haikal2026"):

    st.set_page_config(page_title="Sharee'a AI (by Haikal)", page_icon="🕌", layout="wide")

    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        st.error("Please add GEMINI_API_KEY to your Streamlit Secrets.")
        st.stop()

    client_gemini = genai.Client(api_key=API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_pdfs" not in st.session_state:
        st.session_state.current_pdfs = []
    if "current_vids" not in st.session_state:
        st.session_state.current_vids = []

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    db_path = "./my_db"
    client_db = chromadb.PersistentClient(path=db_path)
    collection = client_db.get_or_create_collection(name="religious_knowledge", embedding_function=embedding_func)

    # --- 2. CORE FUNCTIONS ---

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

        # 1. Search Local Books First
        if search_mode in ["Search Books Only", "Hybrid (Both)"]:
            try:
                # Increased n_results to 5 to find more possible viewpoints
                results = collection.query(query_texts=[query], n_results=5)
                for d, m in zip(results['documents'][0], results['metadatas'][0]):
                    s_name = m.get('source', 'Unknown Book')
                    pdf_sources.append(s_name)
                    pdf_context += f"SOURCE (BOOK): {s_name}\nTEXT: {d}\n---\n"
            except Exception: pass

        # 2. Search YouTube Path Second
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
                            yt_context += f"SOURCE (VIDEO): {v_title}\nLINK: {v_link}\nTRANSCRIPT: {' '.join([x['text'] for x in t])[:1500]}\n---\n"
                        except: continue
                except Exception: continue
                    
        return pdf_context, yt_context, pdf_sources, yt_sources

    # --- 3. SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Control Room")
        mode = st.radio("Search Mode:", ["Search Books Only", "Ask Mostafa Al-Adawi", "Hybrid (Both)"], index=2)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.current_pdfs = []
            st.session_state.current_vids = []
            st.rerun()

    # --- 4. MAIN CHAT INTERFACE ---
    st.title("🕌 Sharee'a AI (by Haikal)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Prioritizing local books, then searching videos..."):
                pdf_context, yt_context, pdf_s, vid_s = get_data(prompt, mode)
                st.session_state.current_pdfs = pdf_s
                st.session_state.current_vids = vid_s
                
                # REFINED SYSTEM INSTRUCTION
                system_instr = """
                You are a scholarly research assistant.
                REQUIRED STRUCTURE:
                1. Start with 'Answers from Books:' and list all different perspectives found in the provided BOOK context. Mention the book name for each.
                2. Then follow with 'Answers from Sheikh Mostafa Al-Adawi (Videos):' and provide his views from the VIDEO context.
                3. If there are multiple different answers for the same question, list them all clearly.
                4. Maintain a 'Confidence Score' at the end.
                5. Match the user's language (Arabic or English).
                """

                try:
                    full_context = f"BOOK CONTEXT:\n{pdf_context}\n\nVIDEO CONTEXT:\n{yt_context}"
                    response = client_gemini.models.generate_content(
                        model="gemini-3.1-flash-lite-preview",
                        contents=f"USER QUESTION: {prompt}\n\n{full_context}",
                        config=types.GenerateContentConfig(
                            system_instruction=system_instr,
                            temperature=0.2
                        )
                    )
                    answer_text = response.text
                except Exception as e:
                    answer_text = f"I encountered an error: {str(e)}"
                
                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
                try:
                    pdf_bytes = create_pdf(prompt, answer_text)
                    st.download_button(label="📥 Save PDF", data=pdf_bytes, file_name="Fatwa_Report.pdf", mime="application/pdf")
                except: pass
                
                st.rerun()
