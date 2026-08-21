"""
The grounded condition: retrieval over data/corpus.json (verified content
lifted from PathFinder's guide articles, see src/build_corpus.py), forced
citations, and an explicit instruction to say "not covered" rather than
fill a gap from the model's own training data.

Retrieval is plain TF-IDF cosine similarity, not embeddings. At 6 chunks
this is a deliberate choice, not a shortcut: TF-IDF is transparent (you
can see exactly why a chunk matched) and needs no extra API calls or
vector database, which matters at this corpus size. The natural upgrade
path if the corpus grows past a few dozen chunks is a real embedding
model; noted in the README rather than reached for prematurely here.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.json"

GROUNDED_SYSTEM_PROMPT = """You are a helpful assistant for students and families with questions about applying to college in the United States, including financial aid.

You will be given CONTEXT passages pulled from a verified source before each question. Answer using ONLY the information in the CONTEXT. Do not use outside knowledge to fill in gaps, even if you believe you know the answer.

Rules:
1. If the CONTEXT answers the question, answer it and cite which passage you used by its heading in brackets, like [In-state tuition and how residency is actually decided].
2. If the CONTEXT does not fully answer the question, say plainly what it does cover, and say the rest is not covered by your available material rather than guessing or filling the gap from general knowledge.
3. Never state a specific state's policy, a specific number, or a specific legal claim that isn't directly present in the CONTEXT.
4. Never tell someone a risk is low or that something is "safe" unless the CONTEXT says so explicitly in those terms.
5. If the CONTEXT is empty or irrelevant to the question, say you don't have verified material on this and point to a school counselor or a licensed immigration attorney."""


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        raise SystemExit(
            "data/corpus.json is missing. Run `python src/build_corpus.py` "
            "(or it should already be committed to this repo)."
        )
    return json.loads(CORPUS_PATH.read_text())


class Retriever:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in corpus])

    def retrieve(self, query: str, k: int = 2) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(self.corpus)), key=lambda i: sims[i], reverse=True)
        return [
            {**self.corpus[i], "similarity": round(float(sims[i]), 3)} for i in ranked[:k]
        ]


def build_user_message(question: str, retrieved: list[dict]) -> str:
    if not retrieved or all(r["similarity"] <= 0 for r in retrieved):
        context_block = "(no relevant context found)"
    else:
        context_block = "\n\n".join(
            f"[{r['heading']}]\n{r['text']}" for r in retrieved if r["similarity"] > 0
        )
    return f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"
