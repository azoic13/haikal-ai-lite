# =============================================================================
# STEP 0 — SQLITE FIX (must be the very first thing that runs)
# =============================================================================
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
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from google import genai
from google.genai import types
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import zipfile
import pathlib
import shutil
import time
import re
import gdown

# =============================================================================
# STEP 1 — PATH RESOLUTION
# All paths are relative to this file so the app works anywhere.
# =============================================================================
BASE_DIR  = pathlib.Path(__file__).parent.resolve()
FONT_PATH = BASE_DIR / "arial.ttf"

# =============================================================================
# STEP 2 — DOWNLOAD & EXTRACT KNOWLEDGE BASE FROM GOOGLE DRIVE
#
# my_db.zip (110 MB) lives on Google Drive — too large for GitHub.
# gdown handles Google's virus-scan confirmation page automatically.
# Extracts into /tmp/my_db (always writable on Streamlit Cloud).
# On every subsequent boot DB_FOLDER already exists → skipped instantly.
# =============================================================================
GDRIVE_FILE_ID = "11T6mhgRjwd7EFvs1VvkE7wUS3XJGUL0M"
DB_FOLDER      = pathlib.Path("/tmp/my_db")
DB_ZIP_PATH    = pathlib.Path("/tmp/my_db.zip")

if not DB_FOLDER.exists():
    with st.spinner("📥 Downloading knowledge base for the first time… (~30s)"):
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                str(DB_ZIP_PATH),
                quiet=False
            )

            # Inspect zip contents before extracting
            with zipfile.ZipFile(DB_ZIP_PATH, "r") as zf:
                names = zf.namelist()
                zf.extractall("/tmp")

            # Auto-detect extraction structure
            # Case 1: my_db/chroma.sqlite3  ← correct
            # Case 2: my_db/my_db/chroma.sqlite3  ← nested, needs fix
            # Case 3: chroma.sqlite3 at root ← extracted flat into /tmp
            sqlite_direct = DB_FOLDER / "chroma.sqlite3"
            nested        = DB_FOLDER / "my_db"
            flat_sqlite   = pathlib.Path("/tmp/chroma.sqlite3")

            if sqlite_direct.exists():
                pass  # perfect structure
            elif nested.exists():
                shutil.copytree(str(nested), str(DB_FOLDER), dirs_exist_ok=True)
            elif flat_sqlite.exists():
                DB_FOLDER.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(flat_sqlite), str(sqlite_direct))

            DB_ZIP_PATH.unlink(missing_ok=True)

            # Verify extraction worked
            if (DB_FOLDER / "chroma.sqlite3").exists():
                st.success("✅ Knowledge base downloaded and ready.")
            else:
                st.warning(
                    f"⚠️ DB folder exists but chroma.sqlite3 not found. "
                    f"Zip top-level entries: {names[:5]}"
                )

        except Exception as e:
            st.error(f"❌ Failed to download knowledge base: {e}")
            DB_FOLDER.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================
MODE_BOOKS   = "Search Hadith Books (كتب الاحاديث و التفسير) Only"
MODE_YOUTUBE = "Mostafa Al-Adawi Youtube Channel"
MODE_HYBRID  = "Hybrid (Both)"

# Both official Mostafa Al-Adawi channels
ADAWI_CHANNELS = ["@mustafaaladawi", "@ftawamostafaaladwy"]

SYSTEM_PROMPT = """You are an expert Islamic knowledge assistant specialising in \
the teachings of Sheikh Mostafa Al-Adawi.

LANGUAGE RULE (CRITICAL):
Reply in the EXACT same language as the user's question.
Arabic question → full Arabic response.
English question → full English response.
Never mix languages. Never default to English.

SOURCES RULE (CRITICAL — MOST IMPORTANT):
You MUST answer ONLY using the information inside the CONTEXT block provided.
NEVER use your own training knowledge to add hadiths, book names, or opinions.
NEVER invent or hallucinate sources, page numbers, or scholar quotes.
If the answer is NOT found in the provided context, reply ONLY with:
  Arabic: "لم أجد إجابة لهذا السؤال في المصادر المتاحة."
  English: "I could not find an answer to this question in the available sources."
Then STOP. Do not supplement with outside knowledge under any circumstances.

OUTPUT FORMAT (CRITICAL):
Structure every response that has context exactly as follows:

1. النص / The Text
   Quote the EXACT wording from the context (hadith, Quran verse, or scholar \
statement). Never quote text not explicitly in the context.

2. الشرح / Explanation
   Explain the meaning based solely on what the context says.

3. المصادر / Sources
   List ONLY sources that appear in the context, with:
   - Book name, volume, and page number (if available)
   - For videos: include the full YouTube link exactly as given in the context

4. درجة الثقة / Confidence
   Give a percentage and a brief reason. Use low confidence if context is thin."""

# =============================================================================
# DEEP SEARCH CONFIGURATION
# Tunable parameters — increase these to cast a wider net.
# =============================================================================
BOOK_RESULTS_PER_VARIANT   = 40    # ChromaDB results per query variant
MAX_BOOK_CONTEXT_CHARS     = 24000 # larger book context for broader recall
YOUTUBE_VIDEOS_PER_CHANNEL = 15    # videos fetched per channel search
TRANSCRIPT_CHAR_LIMIT      = 8000  # characters kept per transcript

# =============================================================================
# MAIN APP
# =============================================================================
with streamlit_analytics.track(
    save_to_json=str(BASE_DIR / "analytics.json"),
    unsafe_password="haikal2026"
):
    st.set_page_config(
        page_title="Sharee'a (شريعة) AI",
        page_icon="🕌",
        layout="wide"
    )

    # ── API key validation ────────────────────────────────────────────────────
    has_groq   = "GROQ_API_KEY"      in st.secrets
    has_gemini = "GEMINI_API_KEY_1"  in st.secrets
    if not has_groq and not has_gemini:
        st.error(
            "No API keys found. Add at least one to Streamlit Secrets:\n"
            "- `GEMINI_API_KEY_1` — from aistudio.google.com (recommended)\n"
            "- `GROQ_API_KEY` — free at console.groq.com (fallback)"
        )
        st.stop()

    # ── Session state ─────────────────────────────────────────────────────────
    for key, default in [("messages", []), ("current_pdfs", []), ("current_vids", [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client_db  = chromadb.PersistentClient(path=str(DB_FOLDER))
    collection = client_db.get_or_create_collection(
        name="religious_knowledge",
        embedding_function=embedding_func
    )

    # ── YouTube transcript API instance (>= 0.6.0 instance-based) ────────────
    ytt_api = YouTubeTranscriptApi()

    # =========================================================================
    # FUNCTIONS
    # =========================================================================

    def fix_arabic_for_pdf(text: str) -> str:
        if not text:
            return ""
        return get_display(arabic_reshaper.reshape(text))

    def create_pdf(question: str, answer: str) -> bytes:
        pdf = FPDF()
        pdf.add_page()
        if FONT_PATH.exists():
            pdf.add_font("ArialAR", "", str(FONT_PATH))
            pdf.set_font("ArialAR", size=12)
        else:
            pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, txt=fix_arabic_for_pdf(f"السؤال: {question}"), align="R")
        pdf.ln(5)
        pdf.multi_cell(0, 10, txt=fix_arabic_for_pdf(f"الإجابة:\n{answer}"), align="R")
        return pdf.output()

    # ── Book search ───────────────────────────────────────────────────────────

    def normalize_arabic(text: str) -> str:
        """
        Normalize Arabic text before searching so that spelling
        variations, diacritics, and different letter forms all match.
        """
        if not text:
            return text
        # Remove tashkeel (diacritics / harakat)
        text = re.sub(r'[ً-ٰٟ]', '', text)
        # Normalize alef variants → bare alef
        text = re.sub(r'[أإآٱ]', 'ا', text)
        # Normalize teh marbuta → heh
        text = text.replace('ة', 'ه')
        # Normalize alef maqsura → ya
        text = text.replace('ى', 'ي')
        # Normalize waw with hamza
        text = text.replace('ؤ', 'و')
        # Normalize ya with hamza
        text = text.replace('ئ', 'ي')
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text).strip()
        return text

    def build_query_variants(query: str) -> list:
        """
        Build multiple search variants from the original query to maximise recall:
        1. Original query as-is
        2. Normalized Arabic (removes diacritics, unifies letter forms)
        3. Core keywords (strips common question words in Arabic/English)
        4. Normalized keywords (normalize the stripped version too)
        5. Individual significant keywords (≥3 chars) for granular matching
        6. Bi-gram pairs of consecutive keywords for phrase-level matching
        """
        normalized = normalize_arabic(query)
        # Strip leading question words
        strip_pat = (
            r'^(ما|ما هو|ما هي|هل|كيف|متى|من|ما صحة|ما حكم|ما حكم|أين|لماذا|'
            r'what is|what are|is |how|when|who|where|why|does|do|can)\s+'
        )
        keywords   = re.sub(strip_pat, '', query,      flags=re.IGNORECASE).strip()
        kw_norm    = re.sub(strip_pat, '', normalized,  flags=re.IGNORECASE).strip()

        # Deduplicate while preserving order
        seen, variants = set(), []
        for v in [query, normalized, keywords, kw_norm]:
            if v and v not in seen:
                seen.add(v)
                variants.append(v)

        # ── NEW: individual keyword variants ──────────────────────────────────
        # Split on spaces, keep only tokens ≥ 3 chars to skip noise words
        stop_words = {
            'في', 'من', 'عن', 'على', 'إلى', 'هل', 'ما', 'هو', 'هي',
            'the', 'a', 'an', 'is', 'of', 'in', 'to', 'and', 'or',
        }
        tokens = [
            t for t in kw_norm.split()
            if len(t) >= 3 and t not in stop_words
        ]
        for token in tokens:
            if token not in seen:
                seen.add(token)
                variants.append(token)

        # ── NEW: bi-gram pairs for phrase matching ────────────────────────────
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram not in seen:
                seen.add(bigram)
                variants.append(bigram)

        return variants

    def search_books(query: str) -> tuple[str, list]:
        """
        Deep multi-variant Arabic-normalized ChromaDB search.
        Runs multiple query variants and merges results, giving broad coverage
        even when the user's spelling differs from the indexed text.
        Results are ranked by best (lowest) distance across all variants.
        """
        sources = []
        try:
            count = collection.count()
            if count == 0:
                st.warning("⚠️ Book database is empty.")
                return "", []

            variants = build_query_variants(query)
            # Store (doc, metadata, best_distance) for dedup + ranking
            doc_map = {}  # key → (doc, meta, distance)

            for variant in variants:
                try:
                    results = collection.query(
                        query_texts=[variant],
                        n_results=min(BOOK_RESULTS_PER_VARIANT, count),
                        include=["documents", "metadatas", "distances"],
                    )
                    if not results.get("documents") or not results["documents"][0]:
                        continue

                    docs      = results["documents"][0]
                    metas     = results["metadatas"][0]
                    distances = results.get("distances", [[]])[0]

                    for idx, (doc, meta) in enumerate(zip(docs, metas)):
                        key = doc[:120]
                        dist = distances[idx] if idx < len(distances) else 999
                        if key not in doc_map or dist < doc_map[key][2]:
                            doc_map[key] = (doc, meta, dist)
                except Exception:
                    continue

            if not doc_map:
                st.info("ℹ️ No matching passages found in the book database for this query.")
                return "", []

            # Sort by distance (best matches first) and build context
            ranked = sorted(doc_map.values(), key=lambda x: x[2])
            context = ""
            for doc, meta, dist in ranked:
                source = meta.get("source", "Unknown Book")
                entry  = f"\n[BOOK SOURCE: {source}]\n{doc}\n"
                if len(context) + len(entry) > MAX_BOOK_CONTEXT_CHARS:
                    break
                sources.append(source)
                context += entry

        except Exception as e:
            st.warning(f"⚠️ Book search error: {e}")
            return "", []

        return context, sources

    # ── YouTube search ────────────────────────────────────────────────────────

    def search_channel_videos(query: str, channel_handle: str, limit: int = YOUTUBE_VIDEOS_PER_CHANNEL) -> list:
        """
        Search directly within an Al-Adawi channel page.
        Using the channel search URL guarantees results are from that channel only.
        """
        url = f"https://www.youtube.com/{channel_handle}/search?query={query}"
        ydl_opts = {
            "quiet":          True,
            "no_warnings":    True,
            "extract_flat":   True,
            "playlist_items": f"1-{limit}",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get("entries", []) or []
        except Exception:
            return []

    def get_transcript(video_id: str) -> list:
        """
        Method 1: youtube-transcript-api (manual + known auto captions).
        Method 2: yt-dlp (catches auto-generated Arabic captions method 1 misses).
        """
        # Method 1
        try:
            fetched = ytt_api.fetch(video_id, languages=["ar", "en"])
            result  = [{"text": s.text} for s in fetched]
            if result:
                return result
        except Exception:
            pass

        # Method 2: yt-dlp auto-subtitle extraction
        ydl_opts = {
            "quiet":           True,
            "no_warnings":     True,
            "writeautosub":    True,
            "subtitleslangs":  ["ar", "en"],
            "skip_download":   True,
            "subtitlesformat": "vtt",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False
                )
                for caption_dict in [
                    info.get("automatic_captions", {}),
                    info.get("subtitles", {})
                ]:
                    for lang in ["ar", "en"]:
                        for track in caption_dict.get(lang, []):
                            raw = track.get("data", "")
                            if raw:
                                lines = [
                                    l.strip() for l in raw.splitlines()
                                    if l.strip()
                                    and "-->" not in l
                                    and not l.strip().startswith("WEBVTT")
                                    and not l.strip().isdigit()
                                ]
                                if lines:
                                    return [{"text": " ".join(lines)}]
        except Exception:
            pass

        raise Exception("No transcript available.")

    def search_youtube(query: str) -> tuple[str, list]:
        """
        Deep YouTube search: queries both Al-Adawi channels with the original
        query AND a normalized/keyword variant, fetches transcripts, deduplicates
        by video ID, and keeps more transcript text per video.
        """
        context  = ""
        sources  = []
        seen_ids = set()
        found_any = False

        # Build search variants for YouTube too (original + normalized keywords)
        strip_pat = (
            r'^(ما|ما هو|ما هي|هل|كيف|متى|من|ما صحة|ما حكم|أين|لماذا|'
            r'what is|what are|is |how|when|who|where|why|does|do|can)\s+'
        )
        yt_queries = list(dict.fromkeys([
            query,
            normalize_arabic(query),
            re.sub(strip_pat, '', normalize_arabic(query), flags=re.IGNORECASE).strip(),
        ]))

        for yt_query in yt_queries:
            if not yt_query:
                continue
            for handle in ADAWI_CHANNELS:
                videos = search_channel_videos(yt_query, handle, limit=YOUTUBE_VIDEOS_PER_CHANNEL)
                for v in videos:
                    video_id = v.get("id") or v.get("url", "").split("=")[-1]
                    if not video_id or video_id in seen_ids:
                        continue
                    seen_ids.add(video_id)

                    title = str(v.get("title", "Untitled"))
                    link  = f"https://www.youtube.com/watch?v={video_id}"

                    try:
                        transcript      = get_transcript(video_id)
                        transcript_text = " ".join(x["text"] for x in transcript)[:TRANSCRIPT_CHAR_LIMIT]
                        sources.append({"title": title, "link": link})
                        context += (
                            f"\n[VIDEO SOURCE: {title}]\n"
                            f"YouTube Link (must include in sources): {link}\n"
                            f"Transcript: {transcript_text}\n"
                        )
                        found_any = True
                    except Exception:
                        pass  # skip videos with no captions silently

        if not found_any:
            st.info("ℹ️ No video transcripts found for this query — answering from books only.")

        return context, sources

    # ── LLM calls ────────────────────────────────────────────────────────────

    def call_groq(context: str, prompt: str) -> str:
        client   = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQ: {prompt}"}
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    def call_gemini(api_key: str, context: str, prompt: str) -> str:
        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=f"CONTEXT:\n{context}\n\nQ: {prompt}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        return response.text

    def call_llm(context: str, prompt: str) -> str:
        """
        Priority: Gemini key rotation (default) → Groq (fallback).
        Gemini keys are read from secrets as GEMINI_API_KEY_1, GEMINI_API_KEY_2, …
        """
        # 1. Gemini key rotation (default LLM)
        gemini_keys = [
            v for k, v in sorted(st.secrets.items())
            if k.startswith("GEMINI_API_KEY")
        ]
        for i, key in enumerate(gemini_keys, 1):
            try:
                return call_gemini(key, context, prompt)
            except Exception as e:
                err = str(e)
                if "PerDay" in err or re.search(r'"limit"\s*:\s*0', err):
                    st.warning(f"⚠️ Gemini key {i} daily quota exhausted, trying next…")
                elif "429" in err:
                    st.warning(f"⚠️ Gemini key {i} rate limited, trying next…")
                    time.sleep(5)
                else:
                    st.warning(f"⚠️ Gemini key {i} error ({str(e)[:80]}…), trying next…")

        # 2. Groq (fallback)
        if "GROQ_API_KEY" in st.secrets:
            try:
                st.info("ℹ️ Gemini unavailable, falling back to Groq…")
                return call_groq(context, prompt)
            except Exception as e:
                st.warning(f"⚠️ Groq also unavailable ({str(e)[:80]}…)")

        return (
            "❌ All API keys exhausted.\n\n"
            "- Add `GEMINI_API_KEY_1` from aistudio.google.com (recommended)\n"
            "- Or add a free `GROQ_API_KEY` from console.groq.com\n"
            "- Or wait until tomorrow for Gemini quota to reset"
        )

    # ── Main data fetch ───────────────────────────────────────────────────────

    def get_data(query: str, search_mode: str) -> tuple[str, list, list]:
        book_context, book_sources = "", []
        yt_context,   yt_sources   = "", []

        if search_mode in [MODE_BOOKS, MODE_HYBRID]:
            book_context, book_sources = search_books(query)

        if search_mode in [MODE_YOUTUBE, MODE_HYBRID]:
            yt_context, yt_sources = search_youtube(query)

        return book_context + yt_context, book_sources, yt_sources

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.title("⚙️ Control Room")

        mode = st.radio(
            "Search Mode:",
            [MODE_BOOKS, MODE_YOUTUBE, MODE_HYBRID],
            index=2
        )

        # Live DB stats
        try:
            doc_count = collection.count()
            if doc_count > 0:
                st.success(f"📚 {doc_count:,} passages indexed")
            else:
                st.warning("📚 Book DB is empty")
        except Exception:
            pass

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages     = []
            st.session_state.current_pdfs = []
            st.session_state.current_vids = []
            st.rerun()

        st.divider()
        st.subheader("📍 Sources Consulted:")
        for p in set(st.session_state.current_pdfs):
            st.write(f"📖 {p}")
        for v in st.session_state.current_vids:
            st.markdown(f"🎥 [{v['title']}]({v['link']})")

        st.divider()
        st.caption("Admin: Add ?analytics=on to URL. Pass: haikal2026")

    # =========================================================================
    # MAIN CHAT INTERFACE
    # =========================================================================
    st.title("🕌 Sharee'a AI (شريعة)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question... / اسأل سؤالاً..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Deep searching sources..."):
                context, pdfs, vids = get_data(prompt, mode)
                st.session_state.current_pdfs = pdfs
                st.session_state.current_vids = vids

            with st.spinner("💬 Generating answer..."):
                answer_text = call_llm(context, prompt)

            st.markdown(answer_text)
            st.session_state.messages.append({"role": "assistant", "content": answer_text})

            try:
                pdf_bytes = create_pdf(prompt, answer_text)
                st.download_button(
                    label="📥 Save as PDF",
                    data=pdf_bytes,
                    file_name="Sharee'a_Report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                print(f"PDF error: {e}")
