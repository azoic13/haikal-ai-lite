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

# --- CONSTANTS ---
MODE_BOOKS   = "Search Hadith Books (كتب الاحاديث و التفسير) Only"
MODE_YOUTUBE = "Mostafa Al-Adawi Youtube Channel"
MODE_HYBRID  = "Hybrid (Both)"

# --- 1. INITIAL SETUP & ANALYTICS WRAPPER ---
with streamlit_analytics.track(save_to_json="./analytics.json", unsafe_password="haikal2026"):

    st.set_page_config(page_title="Sharee'a (شريعة) AI", page_icon="🕌", layout="wide")

    # SECURITY: Use secrets instead of hardcoding
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        st.error("Please add GEMINI_API_KEY to your Streamlit Secrets.")
        st.stop()

    client_gemini = genai.Client(api_key=API_KEY)

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_pdfs" not in st.session_state:
        st.session_state.current_pdfs = []
    if "current_vids" not in st.session_state:
        st.session_state.current_vids = []

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # PERSISTENCE
    db_path = "./my_db"
    client_db = chromadb.PersistentClient(path=db_path)
    collection = client_db.get_or_create_collection(
        name="religious_knowledge",
        embedding_function=embedding_func
    )

    # --- 2. CORE FUNCTIONS ---

    def fix_arabic_for_pdf(text):
        if not text:
            return ""
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
        """
        Fetches context from ChromaDB (books) and/or YouTube transcripts
        depending on the selected search mode.
        """
        pdf_context, pdf_sources = "", []
        yt_context,  yt_sources  = "", []
        CHANNELS = ["@ftawamostafaaladwy", "@fatawa_eladawy"]

        # ── Book search ──────────────────────────────────────────────────────
        if search_mode in [MODE_BOOKS, MODE_HYBRID]:
            try:
                results = collection.query(query_texts=[query], n_results=3)
                if results and results.get("documents"):
                    for d, m in zip(results["documents"][0], results["metadatas"][0]):
                        s_name = m.get("source", "Unknown Book")
                        pdf_sources.append(s_name)
                        pdf_context += f"\n[BOOK SOURCE: {s_name}]\n{d}\n"
                if not pdf_context:
                    st.warning("⚠️ No results found in the book database. Make sure books have been ingested.")
            except Exception as e:
                st.warning(f"⚠️ Book search error: {e}")

        # ── YouTube search ───────────────────────────────────────────────────
        # FIX: mode string now matches the radio button label exactly
        if search_mode in [MODE_YOUTUBE, MODE_HYBRID]:
            found_any = False
            for handle in CHANNELS:
                try:
                    search = VideosSearch(f"{query} {handle}", limit=2)
                    res = search.result().get("result", [])
                    for v in res:
                        try:
                            transcript = YouTubeTranscriptApi.get_transcript(
                                v["id"], languages=["ar", "en"]
                            )
                            v_title = str(v["title"])
                            v_link  = str(v["link"])
                            yt_sources.append({"title": v_title, "link": v_link})
                            transcript_text = " ".join([x["text"] for x in transcript])[:2000]
                            yt_context += (
                                f"\n[VIDEO SOURCE: {v_title}]\n"
                                f"Link: {v_link}\n"
                                f"Transcript: {transcript_text}\n"
                            )
                            found_any = True
                        except Exception as e:
                            st.warning(f"⚠️ Could not fetch transcript for '{v.get('title', 'unknown')}': {e}")
                            continue
                except Exception as e:
                    st.warning(f"⚠️ YouTube search error for channel {handle}: {e}")
                    continue

            if not found_any:
                st.warning("⚠️ No YouTube transcripts could be retrieved.")

        return pdf_context + yt_context, pdf_sources, yt_sources

    # --- 3. SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Control Room")

        # All three strings here are now the single source of truth (use constants)
        mode = st.radio(
            "Search Mode:",
            [MODE_BOOKS, MODE_YOUTUBE, MODE_HYBRID],
            index=2
        )

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages    = []
            st.session_state.current_pdfs = []
            st.session_state.current_vids = []
            st.rerun()

        st.divider()
        st.subheader("📍 Sources Consulted:")
        if st.session_state.current_pdfs:
            for p in set(st.session_state.current_pdfs):
                st.write(f"📖 {p}")
        if st.session_state.current_vids:
            for v in st.session_state.current_vids:
                st.markdown(f"🎥 [{v['title']}]({v['link']})")

        st.divider()
        st.caption("Admin: Add ?analytics=on to URL. Pass: haikal2026")

    # --- 4. MAIN CHAT INTERFACE ---
    st.title("🕌 Sharee'a AI (شريعة)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching sources..."):
                context, pdfs, vids = get_data(prompt, mode)
                st.session_state.current_pdfs = pdfs
                st.session_state.current_vids = vids

                try:
                    response = client_gemini.models.generate_content(
                        # FIX: use a real, available Gemini model name
                        model="gemini-2.0-flash",
                        contents=f"CONTEXT:\n{context}\n\nQ: {prompt}",
                        config=types.GenerateContentConfig(
                            system_instruction=(
                                "You are an expert Islamic knowledge assistant specialising in "
                                "the teachings of Sheikh Mostafa Al-Adawi. "
                                "Always: 1) Provide a confidence score (e.g. Confidence: 85%). "
                                "2) Cite the source titles you used. "
                                "3) Reply in the same language the user used."
                            ),
                            temperature=0.3
                        )
                    )
                    answer_text = response.text
                except Exception as e:
                    answer_text = f"❌ Gemini API error: {str(e)}"

                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})

                # PDF download — fail silently but log to console
                try:
                    pdf_bytes = create_pdf(prompt, answer_text)
                    st.download_button(
                        label="📥 Save as PDF",
                        data=pdf_bytes,
                        file_name="Sharee'a_Report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    print(f"PDF generation error: {e}")

                # FIX: removed st.rerun() — Streamlit reruns automatically after
                # chat input; calling it manually here caused UI flicker and
                # could interrupt the assistant message render.
