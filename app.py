# ============================================
# FULL UPDATED STREAMLIT APP (FINAL FIXED VERSION)
# ============================================

# --- SQLITE FIX (MUST STAY FIRST) ---
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
from groq import Groq
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import zipfile
import pathlib
import re
import shutil
import gdown

# ---------------- PATHS ----------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()
GDRIVE_FILE_ID = "11T6mhgRjwd7EFvs1VvkE7wUS3XJGUL0M"
DB_FOLDER = pathlib.Path("/tmp/my_db")
DB_ZIP_PATH = pathlib.Path("/tmp/my_db.zip")

# ---------------- FIRST RUN DB DOWNLOAD ----------------
if not DB_FOLDER.exists():
    with st.spinner("📥 Downloading book database for the first time…"):
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                str(DB_ZIP_PATH),
                quiet=False
            )

            with zipfile.ZipFile(DB_ZIP_PATH, "r") as zf:
                zf.extractall("/tmp")

            nested = DB_FOLDER / "my_db"
            if nested.exists() and not (DB_FOLDER / "chroma.sqlite3").exists():
                shutil.copytree(str(nested), str(DB_FOLDER), dirs_exist_ok=True)

            DB_ZIP_PATH.unlink(missing_ok=True)

        except Exception as e:
            st.error(f"❌ Failed to download DB: {e}")
            DB_FOLDER.mkdir(parents=True, exist_ok=True)

# ---------------- CONSTANTS ----------------
MODE_BOOKS = "Search Hadith Books Only"

SYSTEM_PROMPT = (
    "You are an expert Islamic knowledge assistant. "
    "Reply ONLY from the provided CONTEXT. "
    "If not found, say clearly you could not find it."
)

# ---------------- HELPERS ----------------
def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def fix_arabic_for_pdf(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text)) if text else ""


# ✅ FINAL SAFE PDF FUNCTION (NO MULTILINE F-STRING)
def create_pdf(question: str, answer: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    font_path = BASE_DIR / "arial.ttf"

    if font_path.exists():
        pdf.add_font("ArialAR", "", str(font_path))
        pdf.set_font("ArialAR", size=12)
    else:
        pdf.set_font("Helvetica", size=12)

    question_text = fix_arabic_for_pdf("السؤال: " + question)
    answer_text = fix_arabic_for_pdf("الإجابة:\n" + answer)

    pdf.multi_cell(0, 10, txt=question_text, align="R")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=answer_text, align="R")

    return pdf.output(dest="S").encode("latin-1")


# ---------------- SEARCH ----------------
def search_books_only(query: str, collection, max_results: int = 10):
    """
    Safe deep search that keeps the app startup stable.
    Uses wider recall + exact keyword fallback, but stays library-only.
    """
    normalized_query = normalize_arabic(query)
    query_words = [w for w in normalized_query.split() if len(w) >= 3][:6]

    all_docs = []
    seen = set()

    # 1) wider semantic recall
    try:
        results = collection.query(
            query_texts=[normalized_query],
            n_results=max_results * 4,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            key = doc[:150]
            if key in seen:
                continue
            seen.add(key)
            score = 0
            if dist <= 1.5:
                score += 5
            for w in query_words:
                if w in normalize_arabic(doc):
                    score += 3
            all_docs.append((score, doc, meta.get("source", "Unknown Book")))
    except Exception:
        pass

    # 2) exact keyword fallback from same library
    for w in query_words:
        try:
            res = collection.get(
                where_document={"$contains": w},
                limit=8,
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
                key = doc[:150]
                if key in seen:
                    continue
                seen.add(key)
                score = 10
                all_docs.append((score, doc, meta.get("source", "Unknown Book")))
        except Exception:
            continue

    # 3) rerank + build context
    all_docs.sort(key=lambda x: x[0], reverse=True)

    context_parts = []
    sources = []

    for _, doc, source in all_docs[:30]:
        sources.append(source)
        context_parts.append(f"\n[BOOK SOURCE: {source}]\n{doc}\n")

    return "\n".join(context_parts), list(dict.fromkeys(sources))


# ---------------- LLM ----------------
def call_llm(context: str, prompt: str):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQ: {prompt}"}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================
# MAIN APP
# ============================================
with streamlit_analytics.track(
    save_to_json=str(BASE_DIR / "analytics.json"),
    unsafe_password="haikal2026"
):
    st.set_page_config(page_title="Sharee'a AI", page_icon="🕌")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    client_db = chromadb.PersistentClient(path=str(DB_FOLDER))
    collection = client_db.get_or_create_collection(
        name="religious_knowledge",
        embedding_function=embedding_func
    )

    st.title("🕌 Sharee'a AI")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            context, _ = search_books_only(prompt, collection)

            if not context:
                answer = "لم أجد إجابة لهذا السؤال في المصادر المتاحة"
            else:
                answer = call_llm(context, prompt)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            pdf_bytes = create_pdf(prompt, answer)
            st.download_button(
                label="📥 Save as PDF",
                data=pdf_bytes,
                file_name="Shareea_Report.pdf",
                mime="application/pdf",
            )
