import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def get_web_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, timeout=8, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
        raw_text = soup.get_text(separator=' ')
        cleaned = clean_text(raw_text)
        word_count = len(cleaned.split())
        print(f"  [DEBUG] Retrieved {word_count} words from page")
        if word_count < 50:
            print(f"  [DEBUG] WARNING: Very little text — site likely blocked the request")
        return cleaned
    except Exception as e:
        print(f"  [DEBUG] Failed to fetch page: {e}")
        return ""


def check_plagiarism(input_text, threshold=0.2, num_results=5):
    input_text = clean_text(input_text)
    input_words = input_text.split()
    print(f"\n[DEBUG] Input text: {len(input_words)} words")
    print(f"[DEBUG] First 15 words: {' '.join(input_words[:15])}\n")
    print("=" * 60)

    query = ' '.join(input_words[:10])
    print(f'Searching for: "{query}"\n')

    try:
        # FIX: Using DuckDuckGo instead of googlesearch
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            urls = [r['href'] for r in results]
        print(f"[DEBUG] Found {len(urls)} URLs to check\n")
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return

    match_found = False

    for i, url in enumerate(urls):
        print(f"--- URL {i+1}: {url}")
        source_text = get_web_content(url)

        if not source_text:
            print("  [SKIP] Empty content\n")
            continue

        source_words = source_text.split()
        chunk_size = 200
        step = 100
        chunks = [
            ' '.join(source_words[j:j+chunk_size])
            for j in range(0, max(1, len(source_words) - chunk_size + 1), step)
        ]

        print(f"  [DEBUG] Comparing against {len(chunks)} chunks")

        best_score = 0.0
        best_chunk_preview = ""

        for chunk in chunks:
            try:
                docs = [input_text, chunk]
                vectorizer = TfidfVectorizer().fit_transform(docs)
                vectors = vectorizer.toarray()
                score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
                if score > best_score:
                    best_score = score
                    best_chunk_preview = ' '.join(chunk.split()[:20])
            except Exception:
                continue

        print(f"  [DEBUG] Best similarity score: {best_score * 100:.2f}%")
        print(f'  [DEBUG] Best chunk starts: "{best_chunk_preview}..."')

        if best_score >= threshold:
            print(f"  PLAGIARISM DETECTED! Score: {best_score * 100:.2f}%")
            print(f"  Source: {url}")
            match_found = True
        else:
            print(f"  Below threshold ({threshold*100:.0f}%) — not flagged")
        print()

    print("=" * 60)
    if match_found:
        print("RESULT: Plagiarism detected.")
    else:
        print("RESULT: No plagiarism detected.")
    print("=" * 60)


# --- Test ---
my_text = """
created a public image of a celibate man without a domestic life, dedicated entirely to his political mission and the governance of Nazi Germany. His relationship with Eva Braun, which lasted nearly 14 years, was hidden from the public and all but his inner circle. Braun biographer Heike Görtemaker notes that the couple enjoyed a normal sex life. Hitler and Braun married in late April 1945, less than 40 hours before committing suicide together. 
"""

check_plagiarism(my_text)