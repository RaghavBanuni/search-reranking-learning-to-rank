"""Queries, documents, graded relevance -- and click logs that lie in a specific, correctable way.

A ranking dataset is a set of **queries**, each with a candidate list of documents, each document carrying
features and a graded relevance label (0 = irrelevant to 4 = perfect). Two things make it different from
ordinary supervised data, and both change what a loss function may look like:

* **Labels are only meaningful within a query.** A relevance of 2 for a navigational query and a 2 for a broad
  informational one are not comparable, so nothing may be pooled across queries without normalisation. This is
  why NDCG is normalised per query and why pointwise regression on raw labels is a weaker model than it looks.
* **Only the ordering matters.** Any monotone transform of the scores gives the same ranking and the same
  metric, so the loss has enormous freedom the metric does not care about -- and a model can improve its loss
  substantially while its NDCG does not move at all.

The clicks are the interesting part. Real training data is not graded labels from assessors; it is a click log,
and clicks are **position biased**: users click what they see, and they see the top of the list. Under the
standard position-based propensity model,

    P(click on d at rank k) = P(examine rank k) * P(relevant | d)
                            = p_k * r_d              with p_k falling steeply in k

so a document's click-through rate confounds its relevance with the rank the *previous* ranker gave it. Train
on raw clicks and the model learns to reproduce the incumbent ranker -- including its mistakes, which never get
clicks and therefore never get corrected. That feedback loop is the central problem in production search, and
``simulate_clicks`` reproduces it exactly so the correction in ``models.py`` can be measured.

Everything here is synthetic, with the true relevance retained alongside the biased clicks so a debiased model
can be scored against the truth rather than against another model's opinion.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

FEATURE_NAMES = (
    "bm25",          # lexical match, the strongest single signal in most systems
    "title_match",   # match in the title specifically
    "freshness",     # recency, which matters for some intents and not others
    "popularity",    # click-independent popularity prior
    "quality",       # editorial or spam score
    "length_ratio",  # a deliberately weak feature, to test that models can ignore one
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    features: "tuple[float, ...]"
    relevance: int  # 0..4, the assessor's grade -- the truth a click log only hints at


@dataclass(frozen=True)
class Query:
    query_id: str
    documents: "tuple[Document, ...]"
    intent: str = "informational"

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def labels(self) -> "list[int]":
        return [document.relevance for document in self.documents]

    def ideal_order(self) -> "list[Document]":
        return sorted(self.documents, key=lambda document: -document.relevance)

    @property
    def has_relevant(self) -> bool:
        """Queries with no relevant document must be excluded from NDCG, not scored as zero.

        Their ideal DCG is zero, so NDCG is 0/0. Scoring them as 0.0 drags the mean down by an amount that
        depends on how many such queries the sample happens to contain, which makes the metric a function of
        the candidate generator rather than of the ranker.
        """
        return any(document.relevance > 0 for document in self.documents)


@dataclass(frozen=True)
class Dataset:
    queries: "tuple[Query, ...]"
    feature_names: "tuple[str, ...]" = FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def split(self, fraction: float = 0.7, seed: int = 0) -> "tuple[Dataset, Dataset]":
        """Split **by query**, never by document.

        Splitting by document puts documents from the same query on both sides, so the model sees part of a
        list it will be tested on. The leak is subtle, flattering, and produces a model that performs worse in
        production than any offline number suggested.
        """
        rng = random.Random(seed)
        shuffled = list(self.queries)
        rng.shuffle(shuffled)
        cut = int(fraction * len(shuffled))
        return (
            Dataset(tuple(shuffled[:cut]), self.feature_names),
            Dataset(tuple(shuffled[cut:]), self.feature_names),
        )

    def scorable(self) -> "list[Query]":
        return [query for query in self.queries if query.has_relevant]

    def label_distribution(self) -> "dict[int, int]":
        counts: dict[int, int] = {}
        for query in self.queries:
            for document in query.documents:
                counts[document.relevance] = counts.get(document.relevance, 0) + 1
        return dict(sorted(counts.items()))


# the true scoring function the generator uses; models have to recover its ordering, not its scale
TRUE_WEIGHTS = {
    "informational": (1.5, 0.8, 0.2, 0.5, 0.6, 0.0),
    "navigational": (0.9, 2.2, 0.0, 0.9, 0.4, 0.0),
    "fresh": (1.0, 0.5, 2.0, 0.3, 0.5, 0.0),
}


def make_dataset(
    queries: int = 400,
    documents_per_query: int = 20,
    seed: int = 0,
    hard_negatives: bool = True,
) -> Dataset:
    """Queries whose relevance follows an intent-dependent linear rule plus noise.

    Two deliberate properties:

    * **The weights differ by intent.** Freshness matters enormously for one intent and not at all for another,
      so a single global linear model cannot be optimal -- which is realistic, and it is why the linear models
      here plateau below the ceiling instead of solving the problem outright.
    * **Hard negatives.** Documents with high BM25 and low relevance: keyword-stuffed pages that look perfect
      to a lexical scorer. Without them a ranking dataset is trivially easy and every model looks excellent,
      which is how a ranker that fails on the queries people actually complain about passes an offline eval.

    ``length_ratio`` carries no signal at all. A model that puts weight on it is fitting noise, and the tests
    check that the trained weight stays small.
    """
    rng = random.Random(seed)
    intents = tuple(TRUE_WEIGHTS)
    built: list[Query] = []

    for query_index in range(queries):
        intent = intents[query_index % len(intents)]
        weights = TRUE_WEIGHTS[intent]
        documents: list[Document] = []
        for document_index in range(documents_per_query):
            hard = hard_negatives and document_index % 7 == 0
            features = (
                rng.betavariate(5.0, 2.0) if hard else rng.betavariate(2.0, 3.0),  # bm25
                rng.betavariate(1.5, 4.0),
                rng.random(),
                rng.betavariate(2.0, 5.0),
                rng.betavariate(2.0, 2.0),
                rng.random(),
            )
            utility = sum(weight * value for weight, value in zip(weights, features))
            if hard:
                utility -= 1.4  # looks great lexically, is not relevant
            utility += rng.gauss(0.0, 0.25)
            # map utility onto a 0..4 grade with thresholds chosen so grade 4 stays rare
            grade = 0
            for threshold in (0.55, 0.85, 1.15, 1.55):
                if utility >= threshold:
                    grade += 1
            documents.append(
                Document(f"q{query_index}_d{document_index}", features, grade)
            )
        built.append(Query(f"q{query_index}", tuple(documents), intent))

    return Dataset(tuple(built))


# ---------------------------------------------------------------------------------------------
# click simulation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClickLog:
    """One impression: the ranking that was shown, and which positions were clicked."""

    query_id: str
    shown: "tuple[str, ...]"
    clicked: "tuple[bool, ...]"

    def clicked_ranks(self) -> "list[int]":
        return [rank for rank, was_clicked in enumerate(self.clicked, start=1) if was_clicked]


def examination_probability(rank: int, severity: float = 1.0) -> float:
    """``p_k = (1/k)^severity`` -- the position-based propensity model.

    ``severity = 0`` means no bias (every position examined equally); 1 reproduces the roughly ``1/k`` decay
    measured in practice; higher values model a heavier drop-off on mobile. The exponent is what
    ``estimate_propensities`` has to recover, and getting it wrong is worse than assuming no bias in one
    direction: **over-estimating** the bias inflates the weights on low-ranked clicks and the variance with
    them.
    """
    if rank < 1:
        raise ValueError("ranks are 1-based")
    if severity < 0:
        raise ValueError("severity cannot be negative")
    return (1.0 / rank) ** severity


def relevance_probability(grade: int, noise: float = 0.1) -> float:
    """P(click | examined) as a function of the grade: the standard ``(2^g - 1) / (2^gmax - 1)`` curve.

    Plus a floor: an irrelevant document still gets accidental clicks, which is what makes click data noisy as
    well as biased -- two different problems, only one of which propensity weighting fixes.
    """
    base = (2**grade - 1) / (2**4 - 1)
    return min(max(noise + (1.0 - noise) * base, 0.0), 1.0)


def simulate_clicks(
    dataset: Dataset,
    ranker,
    impressions_per_query: int = 20,
    severity: float = 1.0,
    top_k: int = 10,
    seed: int = 0,
) -> "list[ClickLog]":
    """Show each query's list under ``ranker`` and sample clicks from the position-based model.

    ``ranker`` is a callable ``(Query) -> list[Document]``, standing in for whatever is in production today.
    Its ordering enters every click, which is the mechanism behind the feedback loop: a good document the
    incumbent ranks tenth is examined a tenth as often as one it ranks first, gets a tenth of the clicks, and
    looks ten times worse to any model trained on those clicks.
    """
    rng = random.Random(seed)
    logs: list[ClickLog] = []
    for query in dataset.queries:
        ordered = ranker(query)[:top_k]
        for _ in range(impressions_per_query):
            clicks = []
            for rank, document in enumerate(ordered, start=1):
                examined = rng.random() < examination_probability(rank, severity)
                clicked = examined and rng.random() < relevance_probability(document.relevance)
                clicks.append(clicked)
            logs.append(
                ClickLog(
                    query.query_id,
                    tuple(document.doc_id for document in ordered),
                    tuple(clicks),
                )
            )
    return logs


def click_through_rates(logs: "list[ClickLog]") -> "dict[int, float]":
    """Observed CTR by rank -- the shape that makes position bias undeniable in one table."""
    shown: dict[int, int] = {}
    clicked: dict[int, int] = {}
    for log in logs:
        for rank, was_clicked in enumerate(log.clicked, start=1):
            shown[rank] = shown.get(rank, 0) + 1
            clicked[rank] = clicked.get(rank, 0) + int(was_clicked)
    return {rank: clicked[rank] / shown[rank] for rank in sorted(shown)}


def estimate_propensities(logs: "list[ClickLog]", anchor: int = 1) -> "dict[int, float]":
    """Estimate ``p_k`` from observed CTR ratios, normalised so ``p_anchor = 1``.

    This is the cheap estimator, and its assumption is severe: it treats the *average relevance* of documents
    at each rank as constant, which is false whenever the incumbent ranker is any good -- better documents sit
    higher, so the CTR decay it measures mixes position bias with genuine relevance and the propensities come
    out too steep.

    Doing it properly needs interventions: **result randomisation** (swap ranks occasionally and compare CTR
    for the same document at different positions) or an intervention-harvesting scheme over natural ranker
    changes. The estimator here is included with its bias stated because it is what most teams actually deploy,
    and because ``models.py`` demonstrates that even a biased propensity estimate beats ignoring the problem.
    """
    rates = click_through_rates(logs)
    if anchor not in rates or rates[anchor] <= 0.0:
        raise ValueError("the anchor rank has no clicks; cannot normalise")
    return {rank: max(rate / rates[anchor], 1e-3) for rank, rate in rates.items()}


def feature_ranker(weights: "tuple[float, ...]"):
    """Build a ranker from fixed weights, to stand in for an incumbent production system."""

    def rank(query: Query) -> "list[Document]":
        return sorted(
            query.documents,
            key=lambda document: -sum(w * v for w, v in zip(weights, document.features)),
        )

    return rank


def bm25_ranker(query: Query) -> "list[Document]":
    """The honest baseline: sort by lexical match alone. Whatever is being built must beat this."""
    return sorted(query.documents, key=lambda document: -document.features[0])


def random_ranker(seed: int = 0):
    """Uniformly random ordering -- the logging policy that makes propensity estimation unbiased.

    Also the one nobody is allowed to deploy, which is the entire tension in unbiased learning to rank: the
    data that identifies position bias cleanly is the data that costs the most revenue to collect.
    """
    rng = random.Random(seed)

    def rank(query: Query) -> "list[Document]":
        shuffled = list(query.documents)
        rng.shuffle(shuffled)
        return shuffled

    return rank
