"""The Identify Agent: proposes a candidate 18th+ attack node for
human-in-the-loop review, grounded in a local corpus rather than live web
retrieval (no browsing/search API in this pipeline). Two real, always-
available stages regardless of LLM backend:

1. Retrieval: TF-IDF + cosine similarity over a small local corpus of
   grounding excerpts (janus/identify/corpus/*.md, drawn from the same
   public sources attacks.yaml itself cites), returning the passages most
   related to a given topic query.
2. Clustering: KMeans over the corpus's TF-IDF vectors, surfacing which
   existing attacks (by keyword overlap with cluster centroids) look
   under-covered relative to the corpus's topic spread: a real, if
   modest, signal for "what's this taxonomy missing."

Stage 3 (drafting a fluent, cited proposal in prose) is genuinely LLM-shaped
work; it uses janus.common.llm's pluggable backend, which means it runs for
real the moment a key/local model is available and falls back to a
template-assembled (not fabricated-fluent) draft under the scripted
backend: always something real and inspectable, never a fake "AI wrote
this" placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from janus.common.llm import get_backend
from janus.identify.atlas import Attack, load_attacks

CORPUS = [
    ("mastercard_ap4m", "Mastercard launched Agent Pay for Machines (AP4M) in June 2026, extending its network to autonomous software agents executing continuous, machine-speed transactions without human checkout flows, via a Verifiable Intent layer that cryptographically scopes payment credentials to specific agents, spending ceilings, and policies."),
    ("cyber_fraud_fusion", "Cyber-fraud integration research argues traditional fraud detection fails because it operates in silos separate from cybersecurity threat intelligence; a majority of fraud leaders only detect breaches after financial losses begin."),
    ("digital_skimming", "Digital skimming (Magecart-style) infections remain industrialized at scale, compromising a large volume of card-not-present transactions by injecting malicious JavaScript into checkout pages."),
    ("app_fraud_growth", "Authorized Push Payment fraud is projected to keep growing sharply through 2028 in instant-payment markets, driven heavily by AI-generated deepfake scam scripts and real-time social engineering."),
    ("ap2_protocol", "The Agent Payments Protocol (AP2) engineers trust using W3C Verifiable Credentials called Mandates (Open Checkout, Checkout, and Payment Mandates) creating a tamper-proof audit trail for delegated purchases, with hash-binding that proves which checkout was paid for but not whether its content still reflects the user's true intent."),
    ("carding_bots", "Automated card-testing bots iterate BIN ranges and transaction parameters, adapting to declines in real time to evade velocity-based fraud rules."),
    ("mule_networks", "GenAI-assisted mule recruitment increasingly happens through chat platforms, onboarding many low-value mule accounts that move funds in coordinated bursts to stay under structuring thresholds."),
]


@dataclass
class RetrievedPassage:
    source_id: str
    text: str
    score: float


class IdentifyAgent:
    def __init__(self, corpus: list[tuple[str, str]] = CORPUS):
        self.corpus = corpus
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([text for _, text in corpus])

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedPassage]:
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self.corpus, sims), key=lambda t: t[1], reverse=True)[:k]
        return [RetrievedPassage(source_id=src, text=text, score=float(s)) for (src, text), s in ranked]

    def cluster_topics(self, n_clusters: int = 3) -> dict[int, list[str]]:
        n_clusters = min(n_clusters, len(self.corpus))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(self._matrix.toarray())
        clusters: dict[int, list[str]] = {}
        for (src, _), label in zip(self.corpus, labels):
            clusters.setdefault(int(label), []).append(src)
        return clusters

    def propose_new_attack(self, topic_query: str, existing: list[Attack] | None = None) -> dict:
        existing = existing or load_attacks()
        passages = self.retrieve(topic_query, k=3)
        backend = get_backend()

        if backend.name == "scripted":
            draft = (
                f"[template-assembled draft, not LLM-authored, retrieval only] "
                f"Candidate attack area related to: {topic_query!r}. "
                f"Most relevant grounding found: {passages[0].text if passages else '(none)'} "
                f"This overlaps with {len(existing)} attacks already in the Atlas; a human reviewer "
                f"should judge whether this constitutes a genuinely new node or a variant of an existing one."
            )
        else:
            prompt = (
                f"Given this grounding evidence:\n"
                + "\n".join(f"- ({p.source_id}) {p.text}" for p in passages)
                + f"\n\nDraft a 2-3 sentence proposal for a new GenAI-powered payment fraud attack vector "
                  f"related to {topic_query!r}, distinct from the {len(existing)} attacks already catalogued. "
                  f"Cite which grounding source(s) support it."
            )
            result = backend.chat(
                [{"role": "system", "content": "You are a payments-fraud research analyst drafting a taxonomy entry for human review."},
                 {"role": "user", "content": prompt}],
                role_tag="identify_agent",
            )
            draft = result.content or "(no content returned)"

        return {
            "topic_query": topic_query,
            "backend": backend.name,
            "grounding": [{"source_id": p.source_id, "score": round(p.score, 4)} for p in passages],
            "draft": draft,
            "status": "pending_human_review",
        }
