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
            "- `GROQ_API_KEY` — free at console.groq.com (recommended)\n"
            "- `GEMINI_API_KEY_1` — fallback"
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
    client_db = chromadb.PersistentClient(path=str(DB_FOLDER))

    # Diagnostic: show all collections in the DB
    existing_collections = [c.name for c in client_db.list_collections()]

    # Use the first available collection if "religious_knowledge" not found
    COLLECTION_NAME = "religious_knowledge"
    if existing_collections and COLLECTION_NAME not in existing_collections:
        COLLECTION_NAME = existing_collections[0]
        st.info(f"ℹ️ Using collection: '{COLLECTION_NAME}' (found: {existing_collections})")

    collection = client_db.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    # Show DB status in sidebar caption for debugging
    _db_count = collection.count()
    _db_cols   = existing_collections

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

    # Transliteration map: English/Latin Islamic terms → Arabic
    TRANSLIT_MAP = {
        "zakat al fitr":"زكاه الفطر","zakat el fitr":"زكاه الفطر",
        "zakat alfitr":"زكاه الفطر","zakatul fitr":"زكاه الفطر",
        "zakat al-fitr":"زكاه الفطر","zakat":"زكاه","zakah":"زكاه",
        "fitr":"فطر","fitir":"فطر","salah":"صلاه","salat":"صلاه",
        "prayer":"صلاه","sawm":"صوم","siyam":"صيام","fasting":"صوم",
        "ramadan":"رمضان","ramadhan":"رمضان","hajj":"حج","haj":"حج",
        "hadith":"حديث","hadeeth":"حديث","sahih":"صحيح","saheeh":"صحيح",
        "sunnah":"سنه","sunna":"سنه","quran":"قرآن","koran":"قرآن",
        "tafsir":"تفسير","fiqh":"فقه","fatwa":"فتوى","fatwah":"فتوى",
        "halal":"حلال","haram":"حرام","sadaqah":"صدقه","sadaqa":"صدقه",
        "wudu":"وضوء","wudhu":"وضوء","ablution":"وضوء","jihad":"جهاد",
        "nikah":"نكاح","marriage":"زواج","divorce":"طلاق","talaq":"طلاق",
        "aqeedah":"عقيده","aqidah":"عقيده","creed":"عقيده",
        "tawhid":"توحيد","tawheed":"توحيد","shirk":"شرك",
        "bidah":"بدعه","bid'ah":"بدعه","tawbah":"توبه","repentance":"توبه",
        "dua":"دعاء","du'a":"دعاء","supplication":"دعاء",
        "dhikr":"ذكر","zikr":"ذكر","jannah":"جنه","paradise":"جنه",
        "jahannam":"جهنم","hell":"جهنم","iman":"إيمان","faith":"إيمان",
        "prophet":"نبي","nabi":"نبي","messenger":"رسول","rasool":"رسول",
        "companion":"صحابي","sahabi":"صحابي","sahaba":"صحابه",
        "ibn":"ابن","abu":"أبو","bint":"بنت",
        "sa":"صاع","sa'":"صاع","mudd":"مد","shahada":"شهاده",
    }

    def transliterate_to_arabic(text: str) -> str:
        lower = text.lower().strip()
        if lower in TRANSLIT_MAP:
            return TRANSLIT_MAP[lower]
        words, result, i = lower.split(), [], 0
        while i < len(words):
            if i + 1 < len(words):
                two = words[i] + " " + words[i+1]
                if two in TRANSLIT_MAP:
                    result.append(TRANSLIT_MAP[two])
                    i += 2
                    continue
            result.append(TRANSLIT_MAP.get(words[i], words[i]))
            i += 1
        return " ".join(result)

    def is_transliteration(text: str) -> bool:
        latin  = sum(1 for c in text if c.isascii() and c.isalpha())
        arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
        return latin > arabic

    def normalize_arabic(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"[ً-ٰٟ]", "", text)
        text = re.sub(r"[أإآٱ]", "ا", text)
        text = text.replace("ة","ه").replace("ى","ي").replace("ؤ","و").replace("ئ","ي")
        return re.sub(r" +", " ", text).strip()

    def build_query_variants(query: str) -> list:
        """
        Build up to 12 search variants covering:
        - Transliteration → Arabic conversion
        - Normalization (diacritics, letter forms)
        - Question word stripping
        - Individual keywords
        - Root approximations
        """
        strip_pat = (
            r"^(ما هو|ما هي|ما صحة|ما حكم|ما معنى|هل يجوز|هل صح|هل ورد|"
            r"ما|هل|كيف|متى|من|حكم|صحة|معنى|تفسير|شرح|"
            r"what is the ruling on|what is|is it|how|when|who|"
            r"ruling on|meaning of|tell me about|explain)\s+"
        )

        translit = transliterate_to_arabic(query) if is_transliteration(query) else ""
        norm     = normalize_arabic(query)
        stripped = re.sub(strip_pat, "", query,    flags=re.IGNORECASE).strip()
        str_norm = normalize_arabic(stripped)
        tr_strip = re.sub(strip_pat, "", translit, flags=re.IGNORECASE).strip() if translit else ""
        tr_norm  = normalize_arabic(tr_strip) if tr_strip else ""

        # Individual Arabic keywords (3+ chars)
        ar_words = [w for w in (tr_norm or str_norm).split() if len(w) >= 3]
        # Root approximation
        roots    = list({w[:-2] for w in ar_words if len(w) >= 6})[:4]

        candidates = [
            query, translit, tr_strip, tr_norm,
            norm, stripped, str_norm,
            " ".join(ar_words[:5]),
        ] + ar_words[:4] + roots

        seen, variants = set(), []
        for v in candidates:
            v = v.strip()
            if v and v not in seen and len(v) >= 2:
                seen.add(v)
                variants.append(v)
        return variants

    def search_books(query: str) -> tuple[str, list]:
        context, sources = "", []
        try:
            if collection.count() == 0:
                st.warning("⚠️ Book database is empty.")
                return "", []

            variants  = build_query_variants(query)
            seen_docs = set()
            MAX_DOCS  = 20

            for variant in variants:
                if len(seen_docs) >= MAX_DOCS:
                    break
                try:
                    results = collection.query(query_texts=[variant], n_results=10)
                    if not results.get("documents") or not results["documents"][0]:
                        continue
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                        if len(seen_docs) >= MAX_DOCS:
                            break
                        key = doc[:150]
                        if key in seen_docs:
                            continue
                        seen_docs.add(key)
                        source = meta.get("source", "Unknown Book")
                        sources.append(source)
                        context += f"\n[BOOK SOURCE: {source}]\n{doc}\n"
                except Exception:
                    continue

            if not context:
                st.info("ℹ️ No matching passages found in the book database.")
            else:
                st.caption(f"📖 {len(seen_docs)} passages from {len(set(sources))} book(s)")

        except Exception as e:
            st.warning(f"⚠️ Book search error: {e}")

        return context, sources

        # ── YouTube search ────────────────────────────────────────────────────────

    def search_channel_videos(query: str, channel_handle: str, limit: int = 5) -> list:
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
        Search both Al-Adawi channels, fetch transcripts, deduplicate by video ID.
        Searches 5 videos per channel = up to 10 candidates total.
        """
        context = ""
        sources = []
        seen_ids = set()
        found_any = False

        for handle in ADAWI_CHANNELS:
            videos = search_channel_videos(query, handle, limit=5)
            for v in videos:
                video_id = v.get("id") or v.get("url", "").split("=")[-1]
                if not video_id or video_id in seen_ids:
                    continue
                seen_ids.add(video_id)

                title = str(v.get("title", "Untitled"))
                link  = f"https://www.youtube.com/watch?v={video_id}"

                try:
                    transcript      = get_transcript(video_id)
                    transcript_text = " ".join(x["text"] for x in transcript)[:2500]
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
        Priority: Groq (free, fast, no daily cap) → Gemini key rotation (fallback).
        Gemini keys are read from secrets as GEMINI_API_KEY_1, GEMINI_API_KEY_2, …
        """
        # 1. Groq
        if "GROQ_API_KEY" in st.secrets:
            try:
                return call_groq(context, prompt)
            except Exception as e:
                st.warning(f"⚠️ Groq unavailable ({str(e)[:80]}…), trying Gemini…")

        # 2. Gemini key rotation
        gemini_keys = [
            v for k, v in sorted(st.secrets.items())
            if k.startswith("GEMINI_API_KEY")
        ]
        if not gemini_keys:
            return "❌ No LLM available. Add GROQ_API_KEY to Streamlit Secrets."

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
                    return f"❌ Gemini error: {e}"

        return (
            "❌ All API keys exhausted.\n\n"
            "- Add a free `GROQ_API_KEY` from console.groq.com\n"
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
                st.caption(f"Collections found: {_db_cols}")
                st.caption(f"DB path: {DB_FOLDER}")
                st.caption(f"chroma.sqlite3 exists: {(DB_FOLDER / 'chroma.sqlite3').exists()}")
        except Exception as e:
            st.warning(f"📚 DB error: {e}")

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
            with st.spinner("🔍 Searching sources..."):
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
