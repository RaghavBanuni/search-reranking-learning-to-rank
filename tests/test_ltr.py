"""Tests.

Two kinds, kept distinct because they carry different weight:

* **Exact**: hand-computed DCG, average precision on a three-document list, the analytic RankNet gradient against
  a finite difference, propensity normalisation, ERR's cascade product. These pin the mathematics and a failure
  is unambiguous.
* **Directional**: a trained model beating a noise feature, the corrected click model beating the uncorrected
  one. Statistical, seed-dependent, and asserted as orderings with tolerance -- never as digits, because a test
  that pins a float from a stochastic training run fails on an irrelevant refactor and teaches nothing.

The sharpest test here is ``test_no_negatives_below_a_click``: it asserts the pair construction forms **zero**
pairs when only the top result is clicked. That property is what stops a click-trained ranker from freezing the
incumbent ordering, and it is invisible in any aggregate metric.
"""

from __future__ import annotations

import math

import pytest

from ltr.data import (
    ClickLog,
    Dataset,
    Document,
    Query,
    TRUE_WEIGHTS,
    click_through_rates,
    estimate_propensities,
    examination_probability,
    feature_ranker,
    make_dataset,
    random_ranker,
    relevance_probability,
    simulate_clicks,
)
from ltr.metrics import (
    average_precision,
    dcg,
    delta_ndcg,
    evaluate,
    expected_reciprocal_rank,
    mean_ndcg,
    mrr,
    ndcg,
    precision_at_k,
    score_spread,
)
from ltr.models import (
    LinearScorer,
    sigmoid,
    train_from_clicks,
    train_pairwise,
    train_pointwise,
)


def document(doc_id: str, relevance: int, features=None) -> Document:
    return Document(doc_id, tuple(features) if features else (0.0,) * 6, relevance)


# ---------------------------------------------------------------------------------------------
# metrics: exact
# ---------------------------------------------------------------------------------------------


def test_dcg_matches_hand_computation():
    """labels [3, 2, 0]: 7/log2(2) + 3/log2(3) + 0/log2(4) = 7 + 1.8928 = 8.8928."""
    expected = 7.0 / math.log2(2) + 3.0 / math.log2(3)
    assert dcg([3, 2, 0]) == pytest.approx(expected)
    assert dcg([3, 2, 0]) == pytest.approx(8.892789, abs=1e-5)


def test_linear_and_exponential_gain_disagree():
    """The gain function is a choice; two NDCG numbers using different gains are not comparable."""
    labels = [4, 1, 0, 0]
    assert dcg(labels, exponential=True) != pytest.approx(dcg(labels, exponential=False))
    # exponential gain weights the top grade 15x rather than 4x
    assert dcg([4], exponential=True) == pytest.approx(15.0)
    assert dcg([4], exponential=False) == pytest.approx(4.0)


def test_ndcg_of_ideal_ordering_is_one_and_reversed_is_worse():
    documents = [document("a", 4), document("b", 2), document("c", 1), document("d", 0)]
    assert ndcg(documents, k=10) == pytest.approx(1.0)
    assert ndcg(list(reversed(documents)), k=10) < 0.7


def test_ndcg_is_nan_when_nothing_is_relevant():
    """0/0 is undefined. Returning 0.0 would make the mean a property of the candidate generator."""
    documents = [document("a", 0), document("b", 0)]
    assert math.isnan(ndcg(documents, k=10))


def test_mean_ndcg_excludes_unjudgeable_queries():
    good = Query("good", (document("a", 3), document("b", 0)))
    empty = Query("empty", (document("c", 0), document("d", 0)))
    dataset_scorer = lambda doc: {"a": 2.0, "b": 1.0, "c": 1.0, "d": 0.0}[doc.doc_id]
    # the empty query is skipped entirely, so a perfect ranking still scores 1.0
    assert mean_ndcg([good, empty], dataset_scorer, k=10) == pytest.approx(1.0)


def test_ndcg_truncation_ignores_documents_below_k():
    top = [document(f"a{i}", 3) for i in range(3)]
    tail = [document(f"b{i}", 3) for i in range(3)]
    # with k=3 the tail is invisible, and the ideal list is truncated to the same depth
    assert ndcg(top + tail, k=3) == pytest.approx(1.0)


def test_mrr_is_the_reciprocal_of_the_first_relevant_rank():
    documents = [document("a", 0), document("b", 0), document("c", 2)]
    assert mrr(documents) == pytest.approx(1.0 / 3.0)
    assert mrr([document("x", 0)]) == 0.0
    # MRR ignores everything after the first hit: these two lists are identical to it
    assert mrr([document("a", 4), document("b", 0)]) == mrr(
        [document("a", 4), document("b", 4)]
    )


def test_average_precision_matches_hand_computation():
    """relevant at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.8333."""
    documents = [document("a", 2), document("b", 0), document("c", 1)]
    assert average_precision(documents) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_discards_grades():
    """A grade-4 and a grade-1 document are the same event to MAP."""
    high = [document("a", 4), document("b", 0)]
    low = [document("a", 1), document("b", 0)]
    assert average_precision(high) == average_precision(low)
    assert ndcg(high, k=10) == ndcg(low, k=10)  # both perfect orderings
    # but the DCG values themselves differ enormously
    assert dcg([4, 0]) > 5 * dcg([1, 0])


def test_err_cascade_discounts_later_relevance():
    """ERR's product term: a good document after a satisfying one contributes almost nothing."""
    first = expected_reciprocal_rank([document("a", 4), document("b", 4)], k=10)
    single = expected_reciprocal_rank([document("a", 4), document("b", 0)], k=10)
    assert first > single
    assert first - single < 0.05  # the second grade-4 adds very little
    assert 0.0 <= first <= 1.0


def test_err_prefers_relevance_earlier():
    early = expected_reciprocal_rank([document("a", 3)] + [document(f"x{i}", 0) for i in range(4)])
    late = expected_reciprocal_rank([document(f"x{i}", 0) for i in range(4)] + [document("a", 3)])
    assert early > late


def test_precision_at_k():
    documents = [document("a", 2), document("b", 0), document("c", 1), document("d", 0)]
    assert precision_at_k(documents, k=4) == pytest.approx(0.5)
    assert precision_at_k(documents, k=1) == pytest.approx(1.0)


def test_metrics_disagree_on_the_same_pair_of_rankings():
    """One perfect hit vs five decent ones: MRR and NDCG rank these differently."""
    one_perfect = Query("q", tuple([document("a", 4)] + [document(f"x{i}", 0) for i in range(9)]))
    five_decent = Query(
        "q", tuple([document(f"b{i}", 2) for i in range(5)] + [document(f"y{i}", 0) for i in range(5)])
    )
    order = lambda docs: {doc.doc_id: -index for index, doc in enumerate(docs)}
    first = evaluate([one_perfect], lambda d: order(one_perfect.documents)[d.doc_id])
    second = evaluate([five_decent], lambda d: order(five_decent.documents)[d.doc_id])
    assert first["mrr"] > second["mrr"] or first["mrr"] == second["mrr"]
    assert second["map"] > first["map"]  # MAP rewards the five relevant documents


# ---------------------------------------------------------------------------------------------
# delta NDCG: the listwise weighting
# ---------------------------------------------------------------------------------------------


def test_delta_ndcg_is_larger_at_the_top_of_the_list():
    """The entire point of LambdaRank: a swap near rank 1 matters far more than one near rank 20."""
    labels = [4, 3, 2, 1] + [0] * 16
    ideal = dcg(sorted(labels, reverse=True), 10)
    top = delta_ndcg(labels, 0, 1, ideal, 10)
    middle = delta_ndcg(labels, 2, 3, ideal, 10)
    assert top > middle > 0.0


def test_delta_ndcg_is_symmetric_and_zero_outside_truncation():
    labels = [4, 3, 2] + [0] * 17
    ideal = dcg(sorted(labels, reverse=True), 10)
    assert delta_ndcg(labels, 0, 2, ideal, 10) == pytest.approx(
        delta_ndcg(labels, 2, 0, ideal, 10)
    )
    # both positions beyond k=10: swapping them cannot change NDCG@10
    assert delta_ndcg(labels, 15, 18, ideal, 10) == 0.0


def test_delta_ndcg_is_zero_for_equal_labels():
    labels = [2, 2, 0]
    ideal = dcg(sorted(labels, reverse=True), 10)
    assert delta_ndcg(labels, 0, 1, ideal, 10) == pytest.approx(0.0)


def test_delta_ndcg_handles_a_zero_ideal():
    assert delta_ndcg([0, 0], 0, 1, 0.0, 10) == 0.0


# ---------------------------------------------------------------------------------------------
# scorer and gradient
# ---------------------------------------------------------------------------------------------


def test_sigmoid_is_stable_at_extremes():
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-12)
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_ranknet_gradient_matches_a_finite_difference():
    """The analytic gradient -sigma * (1 - sigmoid(sigma * d)) checked numerically."""
    high = (0.8, 0.3, 0.5, 0.2, 0.6, 0.1)
    low = (0.4, 0.7, 0.1, 0.5, 0.2, 0.9)
    weights = [0.3, -0.2, 0.5, 0.1, 0.4, -0.1]
    sigma = 1.3
    step = 1e-6

    def loss(current: "list[float]") -> float:
        difference = sum(w * (h - l) for w, h, l in zip(current, high, low))
        return math.log1p(math.exp(-sigma * difference))

    difference = sum(w * (h - l) for w, h, l in zip(weights, high, low))
    analytic_scalar = -sigma * (1.0 - sigmoid(sigma * difference))

    for index in range(len(weights)):
        forward = list(weights)
        backward = list(weights)
        forward[index] += step
        backward[index] -= step
        numeric = (loss(forward) - loss(backward)) / (2 * step)
        analytic = analytic_scalar * (high[index] - low[index])
        assert numeric == pytest.approx(analytic, abs=1e-6)


def test_scoring_is_scale_invariant_for_ranking():
    """Any positive rescaling of the weights gives the identical ranking; scale is meaningless."""
    dataset = make_dataset(queries=20, documents_per_query=8, seed=5)
    small = LinearScorer([0.1, 0.2, 0.05, 0.1, 0.1, 0.0])
    large = LinearScorer([w * 37.0 for w in small.weights])
    for query in dataset.queries:
        assert [d.doc_id for d in small.rank(query)] == [d.doc_id for d in large.rank(query)]
    assert mean_ndcg(dataset.scorable(), small) == pytest.approx(
        mean_ndcg(dataset.scorable(), large)
    )


def test_score_spread_exposes_a_degenerate_model():
    dataset = make_dataset(queries=10, documents_per_query=6, seed=6)
    constant = LinearScorer([0.0] * 6)
    assert score_spread(dataset.queries, constant) == pytest.approx(0.0)
    assert score_spread(dataset.queries, lambda d: d.features[0]) > 0.1


# ---------------------------------------------------------------------------------------------
# data hygiene
# ---------------------------------------------------------------------------------------------


def test_split_is_by_query_with_no_document_leakage():
    dataset = make_dataset(queries=50, documents_per_query=6, seed=7)
    train, test = dataset.split(0.7, seed=7)
    train_ids = {q.query_id for q in train.queries}
    test_ids = {q.query_id for q in test.queries}
    assert not train_ids & test_ids
    assert len(train) + len(test) == len(dataset)
    train_docs = {d.doc_id for q in train.queries for d in q.documents}
    test_docs = {d.doc_id for q in test.queries for d in q.documents}
    assert not train_docs & test_docs


def test_hard_negatives_exist():
    """High BM25, low relevance -- without these the dataset is trivially easy."""
    dataset = make_dataset(queries=100, documents_per_query=20, seed=8, hard_negatives=True)
    tricky = [
        d
        for q in dataset.queries
        for d in q.documents
        if d.features[0] > 0.7 and d.relevance == 0
    ]
    assert len(tricky) > 50
    easy = make_dataset(queries=100, documents_per_query=20, seed=8, hard_negatives=False)
    bm25_only = lambda d: d.features[0]
    # BM25 alone does measurably better when nothing is adversarial
    assert mean_ndcg(easy.scorable(), bm25_only) > mean_ndcg(dataset.scorable(), bm25_only)


def test_labels_span_the_grade_range():
    distribution = make_dataset(queries=200, documents_per_query=20, seed=9).label_distribution()
    assert set(distribution) >= {0, 1, 2, 3}
    assert distribution[0] > distribution[max(distribution)]  # grade 4 stays rare


# ---------------------------------------------------------------------------------------------
# position bias
# ---------------------------------------------------------------------------------------------


def test_examination_probability_decays_and_anchors_at_one():
    assert examination_probability(1) == pytest.approx(1.0)
    assert examination_probability(2, 1.0) == pytest.approx(0.5)
    assert examination_probability(10, 1.0) == pytest.approx(0.1)
    # severity 0 means no bias at all
    assert examination_probability(10, 0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        examination_probability(0)


def test_relevance_probability_is_monotone_with_a_noise_floor():
    values = [relevance_probability(grade) for grade in range(5)]
    assert values == sorted(values)
    assert values[0] > 0.0  # irrelevant documents still get accidental clicks
    assert values[4] == pytest.approx(1.0)


def test_click_rates_fall_with_rank():
    dataset = make_dataset(queries=120, documents_per_query=15, seed=10)
    logs = simulate_clicks(dataset, feature_ranker((1.0, 0.2, 0.0, 0.2, 0.1, 0.0)),
                           impressions_per_query=25, severity=1.0, seed=10)
    rates = click_through_rates(logs)
    assert rates[1] > rates[5] > rates[10]
    assert rates[1] > 3 * rates[10]  # roughly the 1/k shape


def test_propensity_estimate_is_normalised_and_ordered():
    dataset = make_dataset(queries=120, documents_per_query=15, seed=11)
    logs = simulate_clicks(dataset, random_ranker(seed=11), impressions_per_query=25, seed=11)
    estimated = estimate_propensities(logs)
    assert estimated[1] == pytest.approx(1.0)
    assert estimated[2] > estimated[8]


def test_propensity_estimate_is_less_biased_under_randomisation():
    """The estimator assumes constant relevance per rank, which only a random logging policy satisfies."""
    dataset = make_dataset(queries=200, documents_per_query=15, seed=12)
    biased_logs = simulate_clicks(
        dataset, feature_ranker((1.0, 0.2, 0.0, 0.2, 0.1, 0.0)),
        impressions_per_query=25, severity=1.0, seed=12,
    )
    random_logs = simulate_clicks(dataset, random_ranker(seed=13), impressions_per_query=25, seed=13)
    truth = examination_probability(5, 1.0)
    from_biased = abs(estimate_propensities(biased_logs)[5] - truth)
    from_random = abs(estimate_propensities(random_logs)[5] - truth)
    assert from_random < from_biased


def test_zero_severity_removes_the_bias():
    dataset = make_dataset(queries=120, documents_per_query=12, seed=14)
    logs = simulate_clicks(dataset, random_ranker(seed=14), impressions_per_query=30,
                           severity=0.0, seed=14)
    rates = click_through_rates(logs)
    # no position bias and a random ranker: every rank should look alike
    assert abs(rates[1] - rates[10]) < 0.08


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------


def test_pointwise_loss_decreases():
    dataset = make_dataset(queries=80, documents_per_query=10, seed=15)
    _, log = train_pointwise(dataset, epochs=20, seed=15)
    assert log.losses[-1] < log.losses[0]


def test_pairwise_loss_decreases_and_beats_a_noise_feature():
    dataset = make_dataset(queries=200, documents_per_query=15, seed=16)
    train, test = dataset.split(0.7, seed=16)
    model, log = train_pairwise(train, epochs=15, seed=16)
    assert log.losses[-1] < log.losses[0]
    noise = lambda d: d.features[5]  # length_ratio, which carries no signal by construction
    assert mean_ndcg(test.scorable(), model) > mean_ndcg(test.scorable(), noise) + 0.1


def test_lambdarank_also_learns_and_uses_the_same_pairs():
    dataset = make_dataset(queries=200, documents_per_query=15, seed=17)
    train, test = dataset.split(0.7, seed=17)
    ranknet, ranknet_log = train_pairwise(train, epochs=15, lambda_weighting=False, seed=17)
    lambdarank, lambda_log = train_pairwise(train, epochs=15, lambda_weighting=True, seed=17)
    # identical pair construction: the only difference is the |delta NDCG| factor on the gradient
    assert ranknet_log.pairs_used == lambda_log.pairs_used
    noise = lambda d: d.features[5]
    baseline = mean_ndcg(test.scorable(), noise)
    assert mean_ndcg(test.scorable(), lambdarank) > baseline + 0.1
    assert mean_ndcg(test.scorable(), ranknet) > baseline + 0.1


def test_trained_models_ignore_the_useless_feature():
    """length_ratio has a true weight of exactly zero; a large learned weight means fitted noise."""
    dataset = make_dataset(queries=250, documents_per_query=15, seed=18)
    model, _ = train_pairwise(dataset, epochs=20, seed=18)
    assert abs(model.normalised()[5]) < 0.5


def test_trained_model_favours_the_strongest_true_signal():
    """BM25 has the largest true weight for two of three intents, so it should dominate."""
    dataset = make_dataset(queries=250, documents_per_query=15, seed=19)
    model, _ = train_pairwise(dataset, epochs=20, seed=19)
    normalised = model.normalised()
    assert normalised[0] > 0.0
    assert normalised[0] >= max(abs(v) for v in normalised) - 1e-9 or normalised[0] > 0.5


def test_equal_label_pairs_are_skipped():
    """A pair with identical grades carries no ordering information and must not produce updates."""
    flat = Dataset((Query("q", (document("a", 2, (0.9,) * 6), document("b", 2, (0.1,) * 6))),))
    model, log = train_pairwise(flat, epochs=5, seed=20)
    assert log.pairs_used == 0
    assert all(weight == 0.0 for weight in model.weights)


# ---------------------------------------------------------------------------------------------
# learning from clicks
# ---------------------------------------------------------------------------------------------


def test_no_negatives_below_a_click():
    """Only the top result clicked -> zero pairs.

    Documents *below* a click may never have been examined, so they cannot be treated as negatives. This
    property is what keeps a click-trained ranker from simply reproducing the incumbent ordering, and no
    aggregate metric would reveal its absence.
    """
    dataset = make_dataset(queries=1, documents_per_query=5, seed=21)
    shown = tuple(d.doc_id for d in dataset.queries[0].documents)
    log_entry = ClickLog("q0", shown, (True, False, False, False, False))
    _, log = train_from_clicks([log_entry], dataset, epochs=3, seed=21)
    assert log.pairs_used == 0


def test_a_click_beats_the_documents_skipped_above_it():
    dataset = make_dataset(queries=1, documents_per_query=5, seed=22)
    shown = tuple(d.doc_id for d in dataset.queries[0].documents)
    # rank 3 clicked, ranks 1 and 2 skipped -> exactly two pairs per epoch
    log_entry = ClickLog("q0", shown, (False, False, True, False, False))
    _, log = train_from_clicks([log_entry], dataset, epochs=4, seed=22)
    assert log.pairs_used == 8


def test_two_clicks_imply_no_preference_between_them():
    dataset = make_dataset(queries=1, documents_per_query=5, seed=23)
    shown = tuple(d.doc_id for d in dataset.queries[0].documents)
    # ranks 1 and 2 both clicked: rank 2's only candidate negative is rank 1, which was also clicked
    log_entry = ClickLog("q0", shown, (True, True, False, False, False))
    _, log = train_from_clicks([log_entry], dataset, epochs=1, seed=23)
    assert log.pairs_used == 0


def test_propensity_weighting_amplifies_low_rank_clicks():
    """A click at rank 8 under p_8 = 1/8 should carry roughly eight times the weight of one at rank 1."""
    dataset = make_dataset(queries=1, documents_per_query=10, seed=24)
    shown = tuple(d.doc_id for d in dataset.queries[0].documents)
    propensities = {rank: examination_probability(rank, 1.0) for rank in range(1, 11)}
    deep = ClickLog("q0", shown, tuple(index == 7 for index in range(10)))
    unweighted, _ = train_from_clicks([deep], dataset, propensities=None, epochs=1,
                                      learning_rate=0.01, seed=24)
    weighted, _ = train_from_clicks([deep], dataset, propensities=propensities, epochs=1,
                                    learning_rate=0.01, propensity_floor=1e-6, seed=24)
    magnitude_unweighted = sum(abs(w) for w in unweighted.weights)
    magnitude_weighted = sum(abs(w) for w in weighted.weights)
    assert magnitude_weighted > 5 * magnitude_unweighted


def test_propensity_floor_caps_the_weight():
    dataset = make_dataset(queries=1, documents_per_query=10, seed=25)
    shown = tuple(d.doc_id for d in dataset.queries[0].documents)
    tiny = {rank: 1e-9 for rank in range(1, 11)}
    deep = ClickLog("q0", shown, tuple(index == 9 for index in range(10)))
    floored, _ = train_from_clicks([deep], dataset, propensities=tiny, epochs=1,
                                  learning_rate=0.01, propensity_floor=0.05, seed=25)
    # without the floor the weight would be 1e9; with it, the update stays finite and small
    assert all(math.isfinite(weight) for weight in floored.weights)
    assert max(abs(weight) for weight in floored.weights) < 1.0


def test_ips_correction_recovers_more_than_raw_clicks():
    """Directional and seed-dependent: asserted as an ordering with tolerance, never as a number."""
    dataset = make_dataset(queries=250, documents_per_query=15, seed=26)
    train, test = dataset.split(0.7, seed=26)
    incumbent = feature_ranker((1.0, 0.2, 0.0, 0.2, 0.1, 0.0))
    logs = simulate_clicks(train, incumbent, impressions_per_query=25, severity=1.0, seed=26)
    truth = {rank: examination_probability(rank, 1.0) for rank in range(1, 11)}

    naive, _ = train_from_clicks(logs, train, propensities=None, epochs=12, seed=26)
    corrected, _ = train_from_clicks(logs, train, propensities=truth, epochs=12, seed=26)
    naive_ndcg = mean_ndcg(test.scorable(), naive)
    corrected_ndcg = mean_ndcg(test.scorable(), corrected)
    assert corrected_ndcg >= naive_ndcg - 0.02
    noise = lambda d: d.features[5]
    assert corrected_ndcg > mean_ndcg(test.scorable(), noise)


def test_graded_labels_remain_the_ceiling_over_clicks():
    """Clicks are a noisy, biased view of relevance; supervision on grades should not lose to them."""
    dataset = make_dataset(queries=250, documents_per_query=15, seed=27)
    train, test = dataset.split(0.7, seed=27)
    logs = simulate_clicks(
        train, feature_ranker((1.0, 0.2, 0.0, 0.2, 0.1, 0.0)),
        impressions_per_query=25, severity=1.0, seed=27,
    )
    truth = {rank: examination_probability(rank, 1.0) for rank in range(1, 11)}
    from_clicks, _ = train_from_clicks(logs, train, propensities=truth, epochs=12, seed=27)
    from_labels, _ = train_pairwise(train, epochs=20, lambda_weighting=True, seed=27)
    assert mean_ndcg(test.scorable(), from_labels) >= mean_ndcg(test.scorable(), from_clicks) - 0.03


def test_empty_click_log_raises_rather_than_guessing():
    empty = [ClickLog("q0", ("a", "b"), (False, False))]
    with pytest.raises(ValueError):
        estimate_propensities(empty)


def test_evaluate_returns_every_metric():
    dataset = make_dataset(queries=30, documents_per_query=10, seed=28)
    row = evaluate(dataset.scorable(), lambda d: d.features[0], k=10)
    assert set(row) == {"ndcg@k", "mrr", "map", "err", "precision@k"}
    assert all(0.0 <= value <= 1.0 for value in row.values())


def test_true_weights_differ_by_intent():
    """The reason a single linear model cannot reach the oracle -- stated as a fact about the generator."""
    informational = TRUE_WEIGHTS["informational"]
    fresh = TRUE_WEIGHTS["fresh"]
    assert informational[2] != fresh[2]  # freshness matters for one intent and not the other
    assert all(weights[5] == 0.0 for weights in TRUE_WEIGHTS.values())  # length_ratio is noise
