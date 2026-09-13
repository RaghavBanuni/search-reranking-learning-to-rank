"""Ranking metrics, written out -- including the parts that make published numbers incomparable.

    DCG@k  = sum over ranks i<=k of (2^rel_i - 1) / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k

The exponential gain ``2^rel - 1`` is a *choice*, not a definition: a grade-4 document is worth 15 while a
grade-1 is worth 1, so NDCG cares far more about the top grades than a linear gain would. Some papers use
linear gain, which produces different numbers for identical rankings. Two NDCG values from different sources
are therefore not comparable unless both the gain and the truncation are stated -- and they usually are not.

Three more decisions that change the number without changing the ranking:

* **Queries with no relevant document.** IDCG is zero, NDCG is 0/0. Scoring them as 0.0 makes the metric depend
  on the candidate generator; excluding them is the standard, and it is what ``mean_ndcg`` does.
* **Lists shorter than k.** Truncating the ideal list to the same length is what makes NDCG@10 attainable at
  1.0 for a query with five documents.
* **Ties in the score.** A model that gives every document the same score gets whatever order the sort
  happened to produce. Deterministic tie-breaking hides the degeneracy; here ties are broken by document id, and
  the diagnostic ``score_spread`` exists to expose the case directly.

The other metrics answer different questions and disagree with NDCG on purpose. MRR only cares where the first
relevant result landed -- right for a navigational query and misleading for a research one. MAP treats relevance
as binary, so it discards the grades entirely. ERR models a user who stops when satisfied, which is the only
metric here whose value depends on what the user does *after* clicking.
"""

from __future__ import annotations

import math

from .data import Document, Query


def dcg(relevances: "list[int]", k: int | None = None, exponential: bool = True) -> float:
    """Discounted cumulative gain with exponential (default) or linear gain."""
    cutoff = len(relevances) if k is None else min(k, len(relevances))
    total = 0.0
    for index in range(cutoff):
        gain = (2 ** relevances[index] - 1) if exponential else float(relevances[index])
        total += gain / math.log2(index + 2)  # rank i is 1-based, so log2(i+1) = log2(index+2)
    return total


def ndcg(
    ranked: "list[Document]", k: int = 10, exponential: bool = True
) -> float:
    """NDCG@k of a ranking, against the ideal ordering of the same documents.

    Returns ``nan`` when no document is relevant, rather than 0.0 -- the value is genuinely undefined, and
    propagating a nan forces the caller to decide what to do instead of quietly biasing an average.
    """
    labels = [document.relevance for document in ranked]
    ideal = sorted(labels, reverse=True)
    ideal_dcg = dcg(ideal, k, exponential)
    if ideal_dcg == 0.0:
        return float("nan")
    return dcg(labels, k, exponential) / ideal_dcg


def mean_ndcg(
    queries: "list[Query]", scorer, k: int = 10, exponential: bool = True
) -> float:
    """Mean NDCG@k over queries that have at least one relevant document.

    ``scorer`` is a callable ``(Document) -> float``. Documents are sorted by descending score with ties broken
    by document id, so a degenerate all-equal model gets a reproducible -- and visibly poor -- result rather
    than an accidentally good one.
    """
    values = []
    for query in queries:
        if not query.has_relevant:
            continue
        ranked = sorted(query.documents, key=lambda document: (-scorer(document), document.doc_id))
        value = ndcg(ranked, k, exponential)
        if not math.isnan(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")


def mrr(ranked: "list[Document]", threshold: int = 1) -> float:
    """Reciprocal rank of the first document at or above ``threshold`` relevance.

    Right for navigational intent, where there is one correct answer and everything below it is irrelevant.
    Wrong for a research question, where the user wants five good documents and MRR is indifferent to whether
    ranks 2 through 10 are excellent or worthless.
    """
    for rank, document in enumerate(ranked, start=1):
        if document.relevance >= threshold:
            return 1.0 / rank
    return 0.0


def average_precision(ranked: "list[Document]", threshold: int = 1) -> float:
    """Mean of precision@i over the ranks holding a relevant document. Binary relevance only.

    MAP discards the grades: a perfect document and a marginally relevant one are the same event. That makes it
    robust when grades are unreliable, and blind when they are not.
    """
    hits = 0
    total = 0.0
    relevant = sum(1 for document in ranked if document.relevance >= threshold)
    if relevant == 0:
        return float("nan")
    for rank, document in enumerate(ranked, start=1):
        if document.relevance >= threshold:
            hits += 1
            total += hits / rank
    return total / relevant


def expected_reciprocal_rank(ranked: "list[Document]", k: int = 10, max_grade: int = 4) -> float:
    """ERR (Chapelle et al., 2009): a cascade user who stops once satisfied.

        R_i = (2^rel_i - 1) / 2^max_grade
        ERR = sum_i (1/i) * R_i * prod_{j<i} (1 - R_j)

    The product term is what the others lack: a relevant document at rank 5 contributes almost nothing if ranks
    1 to 4 already satisfied the user. That makes ERR the closest of these to a user-facing quantity, and the
    most sensitive to whether the cascade assumption fits the interface at all -- on a grid of images, it does
    not.
    """
    cutoff = min(k, len(ranked))
    unsatisfied = 1.0
    total = 0.0
    for index in range(cutoff):
        satisfaction = (2 ** ranked[index].relevance - 1) / (2**max_grade)
        total += unsatisfied * satisfaction / (index + 1)
        unsatisfied *= 1.0 - satisfaction
    return total


def precision_at_k(ranked: "list[Document]", k: int = 10, threshold: int = 1) -> float:
    cutoff = min(k, len(ranked))
    if cutoff == 0:
        return float("nan")
    return sum(1 for document in ranked[:cutoff] if document.relevance >= threshold) / cutoff


def evaluate(queries: "list[Query]", scorer, k: int = 10) -> "dict[str, float]":
    """Every metric at once, because they disagree and the disagreement is informative.

    A change that lifts NDCG while dropping MRR has moved good documents up the middle of the list and pushed
    the single best one down. Whether that is an improvement depends on the product, not on the metric -- and
    reporting one number hides the question.
    """
    rows: dict[str, list[float]] = {
        "ndcg@k": [],
        "mrr": [],
        "map": [],
        "err": [],
        "precision@k": [],
    }
    for query in queries:
        if not query.has_relevant:
            continue
        ranked = sorted(query.documents, key=lambda document: (-scorer(document), document.doc_id))
        value = ndcg(ranked, k)
        if not math.isnan(value):
            rows["ndcg@k"].append(value)
        rows["mrr"].append(mrr(ranked))
        precision = average_precision(ranked)
        if not math.isnan(precision):
            rows["map"].append(precision)
        rows["err"].append(expected_reciprocal_rank(ranked, k))
        rows["precision@k"].append(precision_at_k(ranked, k))
    return {
        name: (sum(values) / len(values) if values else float("nan"))
        for name, values in rows.items()
    }


def delta_ndcg(
    labels: "list[int]", first: int, second: int, ideal_dcg: float, k: int | None = None
) -> float:
    """|change in NDCG| from swapping two positions -- the weight LambdaRank puts on a pair.

    This is the bridge between a smooth loss and a discontinuous metric: the gradient of a pairwise loss,
    multiplied by how much the metric would move if that pair were swapped. Swapping ranks 1 and 2 matters
    enormously; swapping 19 and 20 does not, and a pairwise loss without this factor treats them identically --
    which is exactly why plain RankNet optimises a quantity nobody reports.
    """
    if ideal_dcg <= 0.0:
        return 0.0
    if k is not None and (first >= k and second >= k):
        return 0.0  # both outside the truncation: swapping them cannot move NDCG@k

    def discount(position: int) -> float:
        return 1.0 / math.log2(position + 2)

    gain_first = 2 ** labels[first] - 1
    gain_second = 2 ** labels[second] - 1
    change = (gain_first - gain_second) * (discount(second) - discount(first))
    return abs(change) / ideal_dcg


def score_spread(queries: "list[Query]", scorer) -> float:
    """Mean within-query spread of the scores: a degeneracy check.

    A model whose scores are nearly identical within a query produces a ranking determined by the tie-break
    rule. Its loss may look fine and its NDCG will be near random, and this number is what makes that visible
    rather than mysterious.
    """
    spreads = []
    for query in queries:
        scores = [scorer(document) for document in query.documents]
        if scores:
            spreads.append(max(scores) - min(scores))
    return sum(spreads) / len(spreads) if spreads else 0.0
