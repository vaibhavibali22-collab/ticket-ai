"""Retrieval step: TF-IDF + cosine similarity (scikit-learn).

This is NOT a trained neural network. TfidfVectorizer turns each ticket into a
vector of weighted word features (words that are frequent in one ticket but
rare across all tickets get high weight). cosine_similarity then measures how
closely the new ticket's vector points in the same direction as each
historical ticket's vector (0 = no shared terms, 1 = identical term profile).
"""
import re
from typing import Dict, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SUFFIXES = ("ing", "ed", "es", "s")

# Words that appear in many tickets but say nothing about the technical issue.
# Without these, "my salary slip is WRONG" matched "WRONG password attempts".
_DOMAIN_STOP_WORDS = {
    "need", "needs", "wrong", "issue", "issues", "problem", "problems", "help",
    "please", "urgent", "asap", "today", "yesterday", "month", "week", "hi",
    "hello", "thanks", "thank", "pls", "kindly",
}
STOP_WORDS = ENGLISH_STOP_WORDS.union(_DOMAIN_STOP_WORDS)


def _stem(word: str) -> str:
    """Tiny suffix stripper so 'disconnects', 'disconnecting', 'disconnected'
    all become 'disconnect'. Only strips if >= 4 letters remain."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> List[str]:
    """lowercase -> split into words -> drop stop words -> stem."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOP_WORDS and len(w) > 1]


def ticket_text(title: str, description: str) -> str:
    # Title is repeated so it carries roughly twice the weight of the description.
    return f"{title} {title} {description}"


def _dedupe(tickets: List[Dict]) -> List[Dict]:
    """Collapse exact duplicate tickets (same normalised title, description and
    resolution), keeping the lowest id, so one issue can't fill the top-5."""
    seen, unique = set(), []
    for t in sorted(tickets, key=lambda t: t["id"]):
        key = tuple(" ".join(str(t.get(f) or "").lower().split())
                    for f in ("title", "description", "resolution"))
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def find_similar_tickets(query: str, historical: List[Dict], *, top_k: int = 5,
                         threshold: float = 0.15, min_shared_terms: int = 2) -> List[Dict]:
    """Return up to top_k historical tickets whose cosine similarity to `query`
    is >= threshold, best first. Each result is the ticket dict plus 'similarity'.

    Relevance guard: a match must also share at least `min_shared_terms`
    distinct words with the query (or all of them, for a one-word query), so a
    single coincidental word can't produce a confident-looking match.

    Raises ValueError for empty input. Returns [] for an empty history or when
    nothing is similar enough.
    """
    if not query or not query.strip():
        raise ValueError("Query text must not be empty.")
    tickets = _dedupe(historical)
    if not tickets:
        return []

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, lowercase=False, token_pattern=None,  # we tokenise ourselves
        ngram_range=(1, 2),   # single words + two-word phrases ("not connecting")
        sublinear_tf=True,    # 1 + log(tf): repeating a word has diminishing effect
    )
    try:
        # FIT on history only: learns the vocabulary and IDF weights.
        matrix = vectorizer.fit_transform(
            [ticket_text(t["title"], t["description"]) for t in tickets])
    except ValueError:  # "empty vocabulary": history contains only stop words
        return []

    # TRANSFORM the new ticket with the SAME fitted vocabulary (never refit on it).
    query_vec = vectorizer.transform([query])
    if query_vec.nnz == 0:  # no word in common with any historical ticket
        return []

    scores = cosine_similarity(query_vec, matrix).ravel()

    # Sort by score (high to low); ties broken by ticket id (low to high) so the
    # order is deterministic. Scores are rounded first so float noise can't
    # reorder genuinely equal scores.
    ranked = sorted(range(len(tickets)),
                    key=lambda i: (-round(float(scores[i]), 6), tickets[i]["id"]))

    query_terms = set(tokenize(query))
    needed_overlap = min(min_shared_terms, len(query_terms))

    results = []
    for i in ranked:
        score = round(float(scores[i]), 4)
        if score < threshold:
            break  # list is sorted, so everything after is lower too
        ticket_terms = set(tokenize(ticket_text(tickets[i]["title"], tickets[i]["description"])))
        if len(query_terms & ticket_terms) < needed_overlap:
            continue
        results.append({**tickets[i], "similarity": score})
        if len(results) == top_k:
            break
    return results
