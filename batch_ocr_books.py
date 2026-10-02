import os
import sys
import time
import json
import ssl
import base64
import argparse
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pymupdf as fitz
except ImportError:
    import fitz

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKS_DIR = os.path.join(BASE_DIR, "books")
IMAGES_DIR = os.path.join(BOOKS_DIR, "images")
TEXTS_DIR = os.path.join(BOOKS_DIR, "texts")

PRIMARY_MODEL = "models/gemini-3.5-flash-lite"
FALLBACK_MODEL = "models/gemini-3.1-flash-lite"

SYSTEM_PROMPT = (
    "You are an expert OCR engine for Ethiopic Fidel (ፊደል) and Harari language texts.\n"
    "Transcribe all text from this page image exactly as printed:\n"
    "1. Preserve original columns, paragraphs, and line breaks.\n"
    "2. If two columns exist, transcribe the Left Column first, then the Right Column.\n"
    "3. Preserve all Ethiopic Fidel characters and Latin phonetic diacritics exactly.\n"
    "4. Do NOT omit any headers, footnotes, or vocabulary entries.\n"
    "CRITICAL REQUIREMENT: Wrap your entire transcribed text strictly between <TRANSCRIPTION> and </TRANSCRIPTION> tags. "
    "Do NOT output internal thoughts, analysis, or conversational text inside the tags."
)

# SSL context for macOS certifi bypass
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def clean_book_slug(filename: str) -> str:
    """Create a clean folder name from a messy filename."""
    name = filename.split(":")[-1]  # remove https::: prefixes
    if name.endswith(".pdf"):
        name = name[:-4]
    name = name.replace(" ", "_").replace(".", "_")
    return name


def render_page_to_jpeg(page, output_jpg_path: str, dpi: int = 150) -> bytes:
    """Render a fitz Page to a high-quality compact JPEG."""
    if os.path.exists(output_jpg_path) and os.path.getsize(output_jpg_path) > 1000:
        with open(output_jpg_path, "rb") as f:
            return f.read()
            
    os.makedirs(os.path.dirname(output_jpg_path), exist_ok=True)
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes(output="jpg", jpg_quality=85)
    with open(output_jpg_path, "wb") as f:
        f.write(img_bytes)
    return img_bytes


# Rate limiter for Gemini 3.1 Flash Lite (15 RPM limit -> ~4.2s per request)
RATE_LOCK = threading.Lock()
LAST_REQUEST_TIME = 0.0
MIN_INTERVAL = 4.2


def rate_limit_wait():
    """Ensure at least MIN_INTERVAL seconds between API calls across all threads."""
    global LAST_REQUEST_TIME
    with RATE_LOCK:
        now = time.time()
        elapsed = now - LAST_REQUEST_TIME
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        LAST_REQUEST_TIME = time.time()


def call_ocr_api(img_bytes: bytes, model_name: str, max_retries: int = 4) -> str:
    """Call Google Generative Language API with retries and exponential backoff."""
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"text": SYSTEM_PROMPT},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192
        }
    }
    
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={API_KEY}"
    data_bytes = json.dumps(payload).encode("utf-8")
    
    for attempt in range(max_retries):
        try:
            rate_limit_wait()
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                candidates = res_json.get("candidates", [])
                if not candidates:
                    return ""
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join([p.get("text", "") for p in parts]).strip()
                
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            # Rate limit or quota
            if e.code == 429:
                if attempt == max_retries - 1 and model_name != FALLBACK_MODEL:
                    print(f"  [*] 429 Quota exhausted for {model_name}. Switching to fallback: {FALLBACK_MODEL}")
                    return call_ocr_api(img_bytes, FALLBACK_MODEL, max_retries=2)
                wait_time = (2 ** attempt) * 5 + 5
                print(f"  [429 Rate Limit] Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            print(f"  [HTTP {e.code}] Error: {err_body[:150]}")
            if attempt == max_retries - 1 and model_name != FALLBACK_MODEL:
                print(f"  [*] Switching to fallback model: {FALLBACK_MODEL}")
                return call_ocr_api(img_bytes, FALLBACK_MODEL, max_retries=2)
            time.sleep(3)
        except Exception as e:
            print(f"  [Network Error]: {e}")
            time.sleep(3)
            
    return ""


def clean_extracted_text(text: str) -> str:
    """Extract strictly between <TRANSCRIPTION> tags, stripping markdown wrappers."""
    import re
    text = text.strip()
    
    # 1. First priority: Extract content between <TRANSCRIPTION> tags
    if "<TRANSCRIPTION>" in text and "</TRANSCRIPTION>" in text:
        text = text.split("<TRANSCRIPTION>")[1].split("</TRANSCRIPTION>")[0].strip()
    elif "<TRANSCRIPTION>" in text:
        text = text.split("<TRANSCRIPTION>")[1].strip()
    else:
        # Fallback if tags were omitted
        parts = re.split(r'\*\*(?:Detailed\s+)?Transcription:?\*\*:?', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[-1].strip()
            
    # Strip markdown code blocks if wrapped
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        
    return text.strip()


def process_single_page(args_tuple):
    """Worker function for parallel processing."""
    pdf_path, page_idx, book_slug, total_pages = args_tuple
    
    img_filename = f"page_{page_idx + 1:04d}.jpg"
    txt_filename = f"page_{page_idx + 1:04d}.txt"
    
    img_path = os.path.join(IMAGES_DIR, book_slug, img_filename)
    txt_path = os.path.join(TEXTS_DIR, book_slug, txt_filename)
    
    # Check if already processed
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 20:
        return page_idx + 1, "SKIPPED", len(open(txt_path, 'r', encoding='utf-8').read())
    
    # Open doc and render page with proper cleanup
    with fitz.open(pdf_path) as doc:
        page = doc[page_idx]
        img_bytes = render_page_to_jpeg(page, img_path)
    
    # Run OCR
    raw_text = call_ocr_api(img_bytes, PRIMARY_MODEL)
    text = clean_extracted_text(raw_text)
    
    if text:
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return page_idx + 1, "OK", len(text)
    else:
        return page_idx + 1, "FAILED", 0


def process_book(pdf_filename: str, concurrency: int = 2, limit_pages: int = None):
    """Process all pages of a single PDF book."""
    pdf_path = os.path.join(BOOKS_DIR, pdf_filename)
    book_slug = clean_book_slug(pdf_filename)
    
    print("\n" + "=" * 70, flush=True)
    print(f"STARTING BOOK: {pdf_filename}", flush=True)
    print(f"SLUG: {book_slug}", flush=True)
    print("=" * 70, flush=True)
    
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    print(f"Total pages: {total_pages}", flush=True)
    
    page_range = range(min(total_pages, limit_pages)) if limit_pages else range(total_pages)
    tasks = [(pdf_path, i, book_slug, total_pages) for i in page_range]
    
    completed = 0
    skipped = 0
    success = 0
    failed = 0
    
    # Use ThreadPoolExecutor for concurrent batch execution
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_single_page, task): task for task in tasks}
        
        for future in as_completed(futures):
            pno, status, char_count = future.result()
            completed += 1
            if status == "SKIPPED":
                skipped += 1
                status_str = f"SKIP (already done, {char_count:,} chars)"
            elif status == "OK":
                success += 1
                status_str = f"DONE ({char_count:,} chars)"
            else:
                failed += 1
                status_str = "FAILED"
                
            percent = (completed / total_pages) * 100
            print(f"[{completed:4d}/{total_pages:4d}] ({percent:5.1f}%) | Page {pno:4d}: {status_str}", flush=True)
            
    print(f"\n[+] Finished {book_slug}: {success} OCR'd, {skipped} skipped, {failed} failed.", flush=True)
    
    # Consolidate book into a single unified text file
    consolidated_path = os.path.join(TEXTS_DIR, f"{book_slug}_FULL.txt")
    book_text_dir = os.path.join(TEXTS_DIR, book_slug)
    
    if os.path.exists(book_text_dir):
        all_pages_text = []
        page_files = sorted([f for f in os.listdir(book_text_dir) if f.endswith(".txt")])
        for pf in page_files:
            p_path = os.path.join(book_text_dir, pf)
            p_text = open(p_path, "r", encoding="utf-8").read().strip()
            all_pages_text.append(f"--- PAGE {pf} ---\n{p_text}\n")
            
        with open(consolidated_path, "w", encoding="utf-8") as cf:
            cf.write("\n".join(all_pages_text))
        print(f"[+] Consolidated full text written to: {consolidated_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch OCR all Harari books using Gemma/Gemini")
    parser.add_argument("--book", type=str, help="Process a specific book by filename or substring", default=None)
    parser.add_argument("--concurrency", type=int, help="Parallel worker threads (default: 2)", default=2)
    parser.add_argument("--limit-pages", type=int, help="Optional limit on pages per book for testing", default=None)
    args = parser.parse_args()
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(TEXTS_DIR, exist_ok=True)
    
    all_pdfs = sorted([f for f in os.listdir(BOOKS_DIR) if f.endswith(".pdf")])
    if not all_pdfs:
        print(f"[-] No PDFs found in {BOOKS_DIR}")
        sys.exit(1)
        
    print(f"Found {len(all_pdfs)} books in {BOOKS_DIR}")
    print(f"Images will be saved to: {IMAGES_DIR}")
    print(f"Texts will be saved to:  {TEXTS_DIR}")
    print(f"Model: {PRIMARY_MODEL} (fallback: {FALLBACK_MODEL})")
    
    target_pdfs = all_pdfs
    if args.book:
        target_pdfs = [f for f in all_pdfs if args.book.lower() in f.lower()]
        if not target_pdfs:
            print(f"[-] No book matching '{args.book}' found.")
            sys.exit(1)
            
    print(f"\nProcessing queue ({len(target_pdfs)} books):")
    for i, b in enumerate(target_pdfs, 1):
        print(f"  {i}. {b}")
        
    for b in target_pdfs:
        process_book(b, concurrency=args.concurrency, limit_pages=args.limit_pages)
        
    print("\n" + "=" * 70)
    print("ALL BOOKS PROCESSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
