"""Three ways to learn a ranking, and the correction that makes click training work.

All three share a **linear scorer** so the comparison isolates the loss rather than the model class. Gradient
boosting would beat all of them; it would also hide which part of the improvement came from the objective.

**Pointwise** regression on the grade. Simple, and structurally mismatched: it spends capacity making the
predicted grade of an irrelevant document accurate, which no ranking metric rewards. It also pools labels across
queries, which is invalid when a grade means different things for different intents.

**Pairwise (RankNet, Burges et al. 2005).** For a pair with ``rel_i > rel_j``, minimise

    L = log(1 + exp(-sigma * (s_i - s_j)))
    dL/ds_i = -sigma / (1 + exp(sigma * (s_i - s_j))) = -lambda_ij

Correct in the ordering it prefers, and blind to *where* in the list a mistake happens: an inversion between
ranks 1 and 2 costs exactly the same as one between 19 and 20. So it optimises a quantity nobody reports.

**Listwise (LambdaRank, Burges et al. 2006).** Take the RankNet gradient and scale it by ``|delta NDCG|`` for
that pair -- how much the metric would actually move if the two documents swapped:

    lambda_ij <- lambda_ij * |delta NDCG_ij|

That is the whole idea, and it is remarkable how cheap it is: one multiplication turns a loss that ignores the
metric into one whose updates are proportional to it. It is a *heuristic* gradient -- LambdaRank has no known
loss function whose gradient it is, though Wang et al. (2018) later showed it optimises a bound on a
metric-based objective. Worth knowing before describing it as "optimising NDCG directly".

**The click correction.** Training on clicks means learning from ``p_k * r_d`` rather than ``r_d``. Weighting
each observed click by ``1 / p_k`` (Joachims et al., 2017) makes the pairwise loss unbiased with respect to
position -- so a good document that the incumbent buried is no longer penalised for having been buried. The
weights are large for low ranks and the variance grows with them, which is why clipping the propensities matters
and why ``propensity_floor`` is an explicit parameter rather than a hidden constant.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .data import ClickLog, Dataset, Document, Query
from .metrics import delta_ndcg, dcg


@dataclass
class LinearScorer:
    """``s(d) = w . features(d)``. No bias: it cancels within a query and affects nothing."""

    weights: "list[float]"

    @classmethod
    def zeros(cls, dimension: int) -> "LinearScorer":
        return cls([0.0] * dimension)

    def score(self, document: Document) -> float:
        return sum(weight * value for weight, value in zip(self.weights, document.features))

    def __call__(self, document: Document) -> float:
        return self.score(document)

    def rank(self, query: Query) -> "list[Document]":
        return sorted(query.documents, key=lambda document: (-self.score(document), document.doc_id))

    def normalised(self) -> "list[float]":
        """Weights scaled to unit max magnitude -- the only comparable form, since scale is arbitrary."""
        largest = max((abs(weight) for weight in self.weights), default=0.0)
        return [weight / largest for weight in self.weights] if largest else list(self.weights)


@dataclass
class TrainingLog:
    losses: "list[float]" = field(default_factory=list)
    ndcgs: "list[float]" = field(default_factory=list)
    pairs_used: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.losses)} epochs, {self.pairs_used} pair updates, "
            f"loss {self.losses[0]:.4f} -> {self.losses[-1]:.4f}"
            if self.losses
            else "no training"
        )


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


# ---------------------------------------------------------------------------------------------
# pointwise
# ---------------------------------------------------------------------------------------------


def train_pointwise(
    dataset: Dataset, learning_rate: float = 0.05, epochs: int = 30, seed: int = 0
) -> "tuple[LinearScorer, TrainingLog]":
    """Least-squares regression of the grade on the features.

    Included as the baseline that looks reasonable and underperforms for a structural reason: squared error on
    the grade is not a ranking objective, and the model happily trades an inversion at rank 1 for a better fit
    on an irrelevant document at rank 40.
    """
    rng = random.Random(seed)
    model = LinearScorer.zeros(dataset.dimension)
    log = TrainingLog()
    documents = [document for query in dataset.queries for document in query.documents]

    for _ in range(epochs):
        rng.shuffle(documents)
        total = 0.0
        for document in documents:
            error = model.score(document) - document.relevance
            total += error**2
            for index, value in enumerate(document.features):
                model.weights[index] -= learning_rate * error * value / len(documents) * 10.0
        log.losses.append(total / len(documents))
    return model, log


# ---------------------------------------------------------------------------------------------
# pairwise and listwise
# ---------------------------------------------------------------------------------------------


def train_pairwise(
    dataset: Dataset,
    learning_rate: float = 0.1,
    epochs: int = 30,
    sigma: float = 1.0,
    lambda_weighting: bool = False,
    truncation: int | None = 10,
    seed: int = 0,
) -> "tuple[LinearScorer, TrainingLog]":
    """RankNet, and -- with ``lambda_weighting=True`` -- LambdaRank.

    The two differ by a single factor, which is the point of implementing them in one function: any performance
    gap is attributable to ``|delta NDCG|`` and to nothing else.

    Pairs are formed only between documents with *different* grades, since a pair with equal labels carries no
    ordering information and contributes gradient noise in both directions.
    """
    rng = random.Random(seed)
    model = LinearScorer.zeros(dataset.dimension)
    log = TrainingLog()
    queries = list(dataset.queries)

    for _ in range(epochs):
        rng.shuffle(queries)
        total_loss = 0.0
        pair_count = 0

        for query in queries:
            documents = list(query.documents)
            # ranking by current score is what makes delta NDCG meaningful: the positions are the model's own
            ordered = sorted(documents, key=lambda document: (-model.score(document), document.doc_id))
            labels = [document.relevance for document in ordered]
            ideal = dcg(sorted(labels, reverse=True), truncation)
            positions = {document.doc_id: index for index, document in enumerate(ordered)}

            for first in range(len(ordered)):
                for second in range(first + 1, len(ordered)):
                    high, low = ordered[first], ordered[second]
                    if high.relevance == low.relevance:
                        continue
                    if high.relevance < low.relevance:
                        high, low = low, high  # ensure `high` is the more relevant document

                    difference = model.score(high) - model.score(low)
                    total_loss += math.log1p(math.exp(-sigma * min(max(difference, -30.0), 30.0)))
                    gradient = -sigma * (1.0 - sigmoid(sigma * difference))

                    if lambda_weighting:
                        gradient *= delta_ndcg(
                            labels,
                            positions[high.doc_id],
                            positions[low.doc_id],
                            ideal,
                            truncation,
                        )

                    pair_count += 1
                    for index in range(len(model.weights)):
                        step = gradient * (high.features[index] - low.features[index])
                        model.weights[index] -= learning_rate * step

        log.losses.append(total_loss / max(pair_count, 1))
        log.pairs_used += pair_count
    return model, log


# ---------------------------------------------------------------------------------------------
# learning from clicks
# ---------------------------------------------------------------------------------------------


def train_from_clicks(
    logs: "list[ClickLog]",
    dataset: Dataset,
    propensities: "dict[int, float] | None" = None,
    learning_rate: float = 0.1,
    epochs: int = 20,
    sigma: float = 1.0,
    propensity_floor: float = 0.05,
    seed: int = 0,
) -> "tuple[LinearScorer, TrainingLog]":
    """Pairwise learning from clicks, optionally corrected by inverse propensity weighting.

    The pair construction is the standard click-based one: a **clicked** document is treated as preferred over
    an **unclicked document shown above it**. Documents below a click are not usable as negatives, because the
    user may simply never have looked at them -- and treating them as negatives is the single most common way a
    click-trained ranker learns to freeze the incumbent ordering in place.

    With ``propensities=None`` the clicks are used raw, which reproduces position bias faithfully. With
    estimated propensities each pair is weighted by ``1 / p_k`` of the clicked rank, floored at
    ``propensity_floor`` -- the floor caps the variance at the cost of some bias, and a propensity of 0.01
    would otherwise let one click at rank 10 outweigh a hundred at rank 1.
    """
    rng = random.Random(seed)
    model = LinearScorer.zeros(dataset.dimension)
    log = TrainingLog()
    by_id = {
        document.doc_id: document
        for query in dataset.queries
        for document in query.documents
    }
    impressions = list(logs)

    for _ in range(epochs):
        rng.shuffle(impressions)
        total_loss = 0.0
        pair_count = 0

        for impression in impressions:
            for rank, was_clicked in enumerate(impression.clicked, start=1):
                if not was_clicked:
                    continue
                clicked = by_id[impression.shown[rank - 1]]
                weight = 1.0
                if propensities is not None:
                    weight = 1.0 / max(propensities.get(rank, 1.0), propensity_floor)

                for higher_rank in range(1, rank):
                    if impression.clicked[higher_rank - 1]:
                        continue  # also clicked: no preference is implied between them
                    skipped = by_id[impression.shown[higher_rank - 1]]
                    difference = model.score(clicked) - model.score(skipped)
                    total_loss += weight * math.log1p(
                        math.exp(-sigma * min(max(difference, -30.0), 30.0))
                    )
                    gradient = -sigma * (1.0 - sigmoid(sigma * difference)) * weight
                    pair_count += 1
                    for index in range(len(model.weights)):
                        step = gradient * (clicked.features[index] - skipped.features[index])
                        model.weights[index] -= learning_rate * step

        log.losses.append(total_loss / max(pair_count, 1))
        log.pairs_used += pair_count
    return model, log


def oracle_scorer(intent_weights: "dict[str, tuple[float, ...]]"):
    """A scorer that knows each query's intent and its true weights: the achievable ceiling.

    Not a model -- an upper bound. A single linear model cannot match it, because the true weights differ by
    intent, and showing the gap is more honest than presenting the best linear model as if it were the limit.
    """

    def score_for(query: Query):
        weights = intent_weights[query.intent]

        def score(document: Document) -> float:
            return sum(w * v for w, v in zip(weights, document.features))

        return score

    return score_for
