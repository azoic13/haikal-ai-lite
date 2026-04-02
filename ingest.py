"""
ingest.py — Run locally to build the ChromaDB knowledge base from your PDFs.

Handles both text-based and image-based (scanned) PDFs automatically:
- Text PDFs: extracted directly with pypdf (fast)
- Image PDFs: OCR'd with easyocr (slow but accurate, supports Arabic + English)

Usage:
    python ingest.py --books_dir ./knowledge_source

Requirements:
    pip install chromadb sentence-transformers pypdf tqdm easyocr pdf2image pillow

pdf2image also needs Poppler on Windows:
    1. Download: https://github.com/oschwartz10612/poppler-windows/releases
    2. Extract to e.g. C:\\poppler
    3. Add C:\\poppler\\Library\\bin to your system PATH

After running:
    Windows: Compress-Archive -Path my_db -DestinationPath my_db.zip -Force
    Then upload my_db.zip to Google Drive (replace existing file, keep same link)
"""

import argparse
import pathlib
import sys
import re

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from tqdm import tqdm

# ---------------------------------------------------------------------------
DB_PATH            = "./my_db"
COLLECTION         = "religious_knowledge"
CHUNK_SIZE         = 500
CHUNK_OVERLAP      = 80
EMBED_MODEL        = "paraphrase-multilingual-MiniLM-L12-v2"
# Minimum characters extracted by pypdf before we consider a page image-based
OCR_FALLBACK_CHARS = 50
# ---------------------------------------------------------------------------

# ── OCR setup ────────────────────────────────────────────────────────────────
_pdf2image_ok = None

def check_pdf2image() -> bool:
    global _pdf2image_ok
    if _pdf2image_ok is not None:
        return _pdf2image_ok
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
        try:
            convert_from_path("__test__.pdf", dpi=72, first_page=1, last_page=1)
        except PDFInfoNotInstalledError:
            print("\n  ❌ Poppler not found.")
            print("     Download: https://github.com/oschwartz10612/poppler-windows/releases")
            print("     Extract to C:\\poppler and add C:\\poppler\\Library\\bin to PATH\n")
            _pdf2image_ok = False
            return False
        except Exception:
            pass
        _pdf2image_ok = True
        return True
    except ImportError:
        print("\n  ❌ pdf2image not installed. Run: pip install pdf2image pillow\n")
        _pdf2image_ok = False
        return False

# ── Arabic normalization ────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text so spelling variations all map to the same form.
    MUST match the normalization applied at search time in app.py.
    """
    if not text:
        return text
    text = re.sub(r'[\u064b-\u065f\u0670]', '', text)   # remove tashkeel
    text = re.sub(r'[أإآٱ]', 'ا', text)                  # unify alef
    text = text.replace('ة', 'ه')                         # teh marbuta → heh
    text = text.replace('ى', 'ي')                         # alef maqsura → ya
    text = text.replace('ؤ', 'و')                         # waw with hamza
    text = text.replace('ئ', 'ي')                         # ya with hamza
    text = re.sub(r' +', ' ', text).strip()
    return text

# ── Text extraction ─────────────────────────────────────────────────────────

def extract_page_text_pypdf(page) -> str:
    """Extract text from a single pypdf page object."""
    try:
        return page.extract_text() or ""
    except Exception:
        return ""

# Explicit path to Tesseract on Windows — pytesseract needs this
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_page_image(image) -> str:
    """
    OCR a PIL image using pytesseract (fast, CPU-friendly).
    Falls back to easyocr only if tesseract genuinely fails.
    """
    # Method 1: pytesseract
    try:
        import pytesseract
        import os
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        text = pytesseract.image_to_string(image, lang="ara+eng")
        if text.strip():
            return text.strip()
        return ""   # empty page — no fallback needed
    except Exception as e:
        print(f"     ⚠️  Tesseract failed: {type(e).__name__}: {e}")
        print(f"          → falling back to easyocr")

    # Method 2: easyocr fallback
    try:
        import easyocr, numpy as np
        global _easyocr_reader
        if "_easyocr_reader" not in globals() or _easyocr_reader is None:
            print("\n  🔄 Loading easyocr model…")
            _easyocr_reader = easyocr.Reader(["ar", "en"], gpu=False)
        results = _easyocr_reader.readtext(np.array(image), detail=0, paragraph=True)
        return " ".join(results)
    except Exception as e:
        print(f"     ⚠️  OCR error: {e}")
        return ""

def extract_text_from_pdf(pdf_path: pathlib.Path) -> tuple[str, int, int]:
    """
    Extract text from all pages of a PDF.
    - Pages with enough text from pypdf → use directly (fast)
    - Pages with little/no text → convert to image and OCR (slow but complete)

    Returns: (full_text, text_page_count, ocr_page_count)
    """
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  ⚠️  Could not open {pdf_path.name}: {e}")
        return "", 0, 0

    all_text    = []
    text_pages  = 0
    ocr_pages   = 0

    total_pages = len(reader.pages)
    print(f"     → {total_pages} pages — processing...", flush=True)

    for page_num, page in enumerate(reader.pages, 1):
        # Print progress every 20 pages
        if page_num % 20 == 0 or page_num == 1:
            print(f"     → page {page_num}/{total_pages} "
                  f"(text:{text_pages} ocr:{ocr_pages})", flush=True)

        page_text = extract_page_text_pypdf(page)

        if len(page_text.strip()) >= OCR_FALLBACK_CHARS:
            all_text.append(page_text)
            text_pages += 1
        else:
            if not check_pdf2image():
                continue
            try:
                from pdf2image import convert_from_path
                # Use lower DPI (150 vs 200) for speed — still readable for OCR
                images = convert_from_path(
                    str(pdf_path),
                    first_page=page_num,
                    last_page=page_num,
                    dpi=150,
                    thread_count=2
                )
                if images:
                    ocr_text = ocr_page_image(images[0])
                    if ocr_text.strip():
                        all_text.append(ocr_text)
                        ocr_pages += 1
            except Exception as e:
                print(f"     ⚠️  Could not OCR page {page_num}: {e}", flush=True)

    return "\n".join(all_text), text_pages, ocr_pages

# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list:
    chunks = []
    start  = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 60]

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest Islamic PDFs (text + image) into ChromaDB"
    )
    parser.add_argument("--books_dir",  default="./knowledge_source",
                        help="Folder with PDFs (searched recursively)")
    parser.add_argument("--db_path",    default=DB_PATH)
    parser.add_argument("--no-ocr",     action="store_true",
                        help="Skip OCR for image-based pages (faster)")
    args = parser.parse_args()

    books_dir = pathlib.Path(args.books_dir)
    db_path   = pathlib.Path(args.db_path)

    if not books_dir.exists():
        print(f"❌ Folder not found: {books_dir}")
        sys.exit(1)

    pdf_files = list(books_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in {books_dir}")
        sys.exit(1)

    print(f"📚 Found {len(pdf_files)} PDF(s) in {books_dir}")
    print(f"💾 Output folder: {db_path}")
    print(f"🔤 Arabic normalization: ON")
    print(f"🔍 OCR for image pages: {'OFF (--no-ocr)' if args.no_ocr else 'ON'}")

    # ── Tesseract diagnostic ─────────────────────────────────────────────────
    if not args.no_ocr:
        import os
        print()
        print("🔍 Tesseract diagnostic:")
        print(f"   Exe path set to : {TESSERACT_PATH}")
        print(f"   Exe exists      : {os.path.exists(TESSERACT_PATH)}")
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            ver   = pytesseract.get_tesseract_version()
            langs = pytesseract.get_languages()
            print(f"   Version         : {ver}")
            print(f"   Languages       : {langs}")
            if "ara" not in langs:
                print("   ⚠️  Arabic pack missing!")
                print("       Download ara.traineddata from:")
                print("       https://github.com/tesseract-ocr/tessdata/blob/main/ara.traineddata")
                print("       Copy to: C:\\Program Files\\Tesseract-OCR\\tessdata\\")
        except Exception as e:
            print(f"   ❌ pytesseract error: {e}")
            print("       Make sure pytesseract is installed: pip install pytesseract")
        print()
    # ────────────────────────────────────────────────────────────────────────

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client     = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_func
    )

    existing = collection.count()
    if existing > 0:
        print(f"ℹ️  Collection already has {existing:,} passages.")
        ans = input("   Clear and rebuild from scratch? (y/N): ").strip().lower()
        if ans == "y":
            client.delete_collection(COLLECTION)
            collection = client.get_or_create_collection(
                name=COLLECTION, embedding_function=embedding_func
            )
            print("   ✅ Collection cleared.\n")
        else:
            print("   ➕ Adding on top of existing.\n")

    total_chunks     = 0
    total_text_pages = 0
    total_ocr_pages  = 0
    skipped          = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        source = pdf_path.stem
        print(f"\n  📖 {pdf_path.name}")

        if args.no_ocr:
            # Fast path: pypdf only
            try:
                reader = PdfReader(str(pdf_path))
                raw    = "\n".join(p.extract_text() or "" for p in reader.pages)
                text_p, ocr_p = len(reader.pages), 0
            except Exception as e:
                print(f"     ⚠️  Skipping: {e}")
                skipped += 1
                continue
        else:
            raw, text_p, ocr_p = extract_text_from_pdf(pdf_path)

        if not raw.strip():
            print("     ⚠️  No text extracted — skipping.")
            skipped += 1
            continue

        print(f"     → text pages: {text_p}  |  OCR pages: {ocr_p}")

        # Normalize then chunk
        normalized = normalize_arabic(raw)
        chunks     = chunk_text(normalized)
        print(f"     → {len(chunks)} chunks")

        # Store in batches of 100
        for i in range(0, len(chunks), 100):
            batch     = chunks[i:i+100]
            ids       = [f"{source}_{i+j}" for j in range(len(batch))]
            metadatas = [{"source": source} for _ in batch]
            collection.add(documents=batch, ids=ids, metadatas=metadatas)

        total_chunks     += len(chunks)
        total_text_pages += text_p
        total_ocr_pages  += ocr_p

    print(f"\n{'='*55}")
    print(f"✅ Ingestion complete!")
    print(f"   PDFs processed : {len(pdf_files) - skipped} / {len(pdf_files)}")
    print(f"   Text pages     : {total_text_pages:,}")
    print(f"   OCR pages      : {total_ocr_pages:,}")
    print(f"   Chunks stored  : {total_chunks:,}")
    print(f"   Total in DB    : {collection.count():,}")
    print(f"{'='*55}")
    print()
    print("Next steps:")
    print("  Windows : Compress-Archive -Path my_db -DestinationPath my_db.zip -Force")
    print("  Mac/Linux: zip -r my_db.zip my_db/")
    print("  Then upload my_db.zip to Google Drive (replace existing file).")

if __name__ == "__main__":
    main()