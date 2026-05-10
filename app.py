import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import fitz  # pymupdf
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# ─── Core plagiarism logic ───────────────────────────────────────────────────

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
        return clean_text(soup.get_text(separator=' '))
    except:
        return ""


def check_plagiarism(input_text, num_results=5, threshold=0.2, log=None):
    input_text = clean_text(input_text)
    if not input_text:
        return []

    query = ' '.join(input_text.split()[:10])
    if log:
        log(f"Searching for: \"{query}\"\n")

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            urls = [r['href'] for r in results]
    except Exception as e:
        if log:
            log(f"Search failed: {e}\n")
        return []

    if log:
        log(f"Found {len(urls)} URLs to check...\n")

    findings = []
    for i, url in enumerate(urls):
        if log:
            log(f"Checking URL {i+1}/{len(urls)}: {url}\n")

        source_text = get_web_content(url)
        if not source_text:
            if log:
                log("  → Could not retrieve content, skipping.\n")
            continue

        source_words = source_text.split()
        chunk_size = 200
        step = 100
        chunks = [
            ' '.join(source_words[j:j+chunk_size])
            for j in range(0, max(1, len(source_words) - chunk_size + 1), step)
        ]

        best_score = 0.0
        for chunk in chunks:
            try:
                docs = [input_text, chunk]
                vectorizer = TfidfVectorizer().fit_transform(docs)
                vectors = vectorizer.toarray()
                score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
                if score > best_score:
                    best_score = score
            except:
                continue

        if log:
            log(f"  → Similarity: {best_score * 100:.1f}%\n")

        findings.append({
            "url": url,
            "score": best_score,
            "flagged": best_score >= threshold
        })

    return findings


# ─── AI detection ────────────────────────────────────────────────────────────

def check_ai_content(text, log=None):
    if log:
        log("Running local AI detector...\n")

    try:
        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model="Hello-SimpleAI/chatgpt-detector-roberta"
        )

        truncated = ' '.join(text.split()[:400])
        result = classifier(truncated)[0]

        if log:
            log(f"Result: {result}\n")

        if result["label"] == "LABEL_1":
            score = result["score"]
        else:
            score = 1 - result["score"]

        # Amplify score for better sensitivity
        score = min(score * 2.5, 1.0)

        if log:
            log(f"AI score: {score * 100:.1f}%\n")

        return score

    except Exception as e:
        if log:
            log(f"AI detection failed: {e}\n")
        return None


# ─── GUI ─────────────────────────────────────────────────────────────────────

class PlagiarismApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plagiarism & AI Content Checker")
        self.root.geometry("750x720")
        self.root.resizable(True, True)
        self.root.configure(bg="#f5f5f0")
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 16, "pady": 8}

        # ── Header ──
        header = tk.Frame(self.root, bg="#2c2c2a")
        header.pack(fill="x")
        tk.Label(header, text="Plagiarism & AI Content Checker",
                 font=("Helvetica", 16, "bold"),
                 bg="#2c2c2a", fg="white", pady=14).pack(side="left", padx=16)

        # ── Input area ──
        input_frame = tk.LabelFrame(self.root, text="Input Text",
                                    font=("Helvetica", 11),
                                    bg="#f5f5f0", fg="#444", padx=12, pady=8)
        input_frame.pack(fill="both", expand=True, **pad)

        self.text_input = tk.Text(input_frame, height=10, font=("Helvetica", 12),
                                  wrap="word", relief="flat", bd=1,
                                  bg="white", fg="#2c2c2a",
                                  insertbackground="#2c2c2a")
        self.text_input.pack(fill="both", expand=True)
        self.text_input.insert("1.0", "Paste your text here...")
        self.text_input.bind("<FocusIn>", self._clear_placeholder)
        self.text_input.bind("<FocusOut>", self._restore_placeholder)

        # ── Buttons row ──
        btn_frame = tk.Frame(self.root, bg="#f5f5f0")
        btn_frame.pack(fill="x", padx=16, pady=4)

        self.upload_btn = tk.Button(btn_frame, text="Upload PDF",
                                    font=("Helvetica", 11), command=self._upload_pdf,
                                    bg="white", fg="#2c2c2a", relief="solid", bd=1,
                                    padx=14, pady=6, cursor="hand2",
                                    activebackground="#e8e8e0")
        self.upload_btn.pack(side="left", padx=(0, 8))

        self.clear_btn = tk.Button(btn_frame, text="Clear",
                                   font=("Helvetica", 11), command=self._clear_all,
                                   bg="white", fg="#888", relief="solid", bd=1,
                                   padx=14, pady=6, cursor="hand2",
                                   activebackground="#e8e8e0")
        self.clear_btn.pack(side="left")

        toggle_frame = tk.Frame(btn_frame, bg="#f5f5f0")
        toggle_frame.pack(side="right")

        self.check_plagiarism_var = tk.BooleanVar(value=True)
        self.check_ai_var = tk.BooleanVar(value=True)

        tk.Checkbutton(toggle_frame, text="Plagiarism",
                       variable=self.check_plagiarism_var,
                       bg="#f5f5f0", font=("Helvetica", 11)).pack(side="left", padx=4)

        tk.Checkbutton(toggle_frame, text="AI Content",
                       variable=self.check_ai_var,
                       bg="#f5f5f0", font=("Helvetica", 11)).pack(side="left", padx=4)

        self.check_btn = tk.Button(toggle_frame, text="Check",
                                   font=("Helvetica", 11, "bold"), command=self._run_check,
                                   bg="#2c2c2a", fg="white", relief="flat",
                                   padx=18, pady=6, cursor="hand2",
                                   activebackground="#444441")
        self.check_btn.pack(side="left", padx=(8, 0))

        # ── Progress bar ──
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=16, pady=(0, 4))

        # ── AI score banner ──
        self.ai_banner = tk.Frame(self.root, bg="#f5f5f0")
        self.ai_banner.pack(fill="x", padx=16, pady=(0, 4))
        self.ai_label = tk.Label(self.ai_banner, text="",
                                 font=("Helvetica", 12, "bold"),
                                 bg="#f5f5f0", pady=6)
        self.ai_label.pack(fill="x")

        # ── Results area ──
        result_frame = tk.LabelFrame(self.root, text="Results",
                                     font=("Helvetica", 11),
                                     bg="#f5f5f0", fg="#444", padx=12, pady=8)
        result_frame.pack(fill="both", expand=True, **pad)

        self.result_text = tk.Text(result_frame, height=12, font=("Courier", 11),
                                   wrap="word", relief="flat", bd=1,
                                   bg="white", fg="#2c2c2a", state="disabled")
        scrollbar = tk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.result_text.pack(fill="both", expand=True)

        self.result_text.tag_config("match", foreground="#a32d2d",
                                    font=("Courier", 11, "bold"))
        self.result_text.tag_config("ok", foreground="#0f6e56")
        self.result_text.tag_config("muted", foreground="#888")

    def _clear_placeholder(self, event):
        if self.text_input.get("1.0", "end-1c") == "Paste your text here...":
            self.text_input.delete("1.0", "end")
            self.text_input.config(fg="#2c2c2a")

    def _restore_placeholder(self, event):
        if not self.text_input.get("1.0", "end-1c").strip():
            self.text_input.insert("1.0", "Paste your text here...")
            self.text_input.config(fg="#aaa")

    def _upload_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        try:
            doc = fitz.open(path)
            text = " ".join(page.get_text() for page in doc)
            self.text_input.delete("1.0", "end")
            self.text_input.config(fg="#2c2c2a")
            self.text_input.insert("1.0", text.strip())
        except Exception as e:
            messagebox.showerror("Error", f"Could not read PDF:\n{e}")

    def _clear_all(self):
        self.text_input.delete("1.0", "end")
        self._restore_placeholder(None)
        self._set_result("")
        self.ai_label.config(text="", bg="#f5f5f0")
        self.ai_banner.config(bg="#f5f5f0")

    def _run_check(self):
        text = self.text_input.get("1.0", "end-1c").strip()
        if not text or text == "Paste your text here...":
            messagebox.showwarning("No text", "Please paste or upload some text first.")
            return

        if not self.check_plagiarism_var.get() and not self.check_ai_var.get():
            messagebox.showwarning("Nothing selected", "Please select at least one check.")
            return

        self.check_btn.config(state="disabled", text="Checking...")
        self.upload_btn.config(state="disabled")
        self.progress.start(10)
        self.ai_label.config(text="", bg="#f5f5f0")
        self.ai_banner.config(bg="#f5f5f0")
        self._set_result("")

        run_plagiarism = self.check_plagiarism_var.get()
        run_ai = self.check_ai_var.get()

        def worker():
            ai_score = None
            findings = []

            if run_ai:
                ai_score = check_ai_content(text, log=self._log)

            if run_plagiarism:
                findings = check_plagiarism(text, log=self._log)

            self.root.after(0, lambda: self._show_results(findings, ai_score, run_plagiarism))

        threading.Thread(target=worker, daemon=True).start()

    def _log(self, msg):
        self.root.after(0, lambda m=msg: self._append_result(m, tag="muted"))

    def _show_results(self, findings, ai_score, ran_plagiarism):
        self.progress.stop()
        self.check_btn.config(state="normal", text="Check")
        self.upload_btn.config(state="normal")

        # ── AI banner ──
        if ai_score is not None:
            pct = ai_score * 100
            if pct >= 50:
                msg = f"AI Content: {pct:.1f}% — Likely AI written"
                bg, fg = "#fcebeb", "#a32d2d"
            elif pct >= 20:
                msg = f"AI Content: {pct:.1f}% — Possibly AI written"
                bg, fg = "#faeeda", "#854f0b"
            else:
                msg = f"AI Content: {pct:.1f}% — Likely human written"
                bg, fg = "#eaf3de", "#3b6d11"
            self.ai_label.config(text=msg, bg=bg, fg=fg)
            self.ai_banner.config(bg=bg)

        # ── Plagiarism results ──
        if ran_plagiarism:
            self._append_result(f"{'='*55}\n", tag="muted")
            if findings:
                flagged = [f for f in findings if f["flagged"]]
                if flagged:
                    self._append_result(
                        f"PLAGIARISM DETECTED — {len(flagged)} source(s) matched\n\n",
                        tag="match")
                else:
                    self._append_result(
                        "No significant plagiarism detected.\n\n", tag="ok")

                for f in findings:
                    score_str = f"{f['score']*100:.1f}%"
                    tag = "match" if f["flagged"] else "ok"
                    self._append_result(f"[{score_str}] ", tag=tag)
                    self._append_result(f"{f['url']}\n", tag="muted")
            else:
                self._append_result("No URLs found to check.\n", tag="muted")
            self._append_result(f"{'='*55}\n", tag="muted")

    def _set_result(self, text, tag=None):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        if text:
            self.result_text.insert("end", text, tag or "")
        self.result_text.config(state="disabled")

    def _append_result(self, text, tag=None):
        self.result_text.config(state="normal")
        self.result_text.insert("end", text, tag or "")
        self.result_text.see("end")
        self.result_text.config(state="disabled")


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = PlagiarismApp(root)
    root.mainloop()