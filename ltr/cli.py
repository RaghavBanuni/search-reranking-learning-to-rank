"""Demos.

    python -m ltr.cli metrics     the metrics disagree, on rankings built to make them disagree
    python -m ltr.cli losses       pointwise vs RankNet vs LambdaRank, same model class
    python -m ltr.cli bias         position bias in the click log, and what it teaches a model
    python -m ltr.cli debias       inverse propensity weighting against raw clicks
    python -m ltr.cli all
"""

from __future__ import annotations

import sys

from .data import (
    TRUE_WEIGHTS,
    Document,
    Query,
    bm25_ranker,
    click_through_rates,
    estimate_propensities,
    examination_probability,
    feature_ranker,
    make_dataset,
    random_ranker,
    simulate_clicks,
)
from .metrics import evaluate, mean_ndcg, score_spread
from .models import (
    LinearScorer,
    oracle_scorer,
    train_from_clicks,
    train_pairwise,
    train_pointwise,
)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fake(doc_id: str, relevance: int) -> Document:
    return Document(doc_id, (0.0,) * 6, relevance)


def demo_metrics() -> None:
    rule("THE METRICS DISAGREE -- and the disagreement is the information")
    lists = {
        "one perfect result at rank 1": [fake("a", 4)] + [fake(f"x{i}", 0) for i in range(9)],
        "five decent results, none perfect": [fake(f"b{i}", 2) for i in range(5)]
        + [fake(f"y{i}", 0) for i in range(5)],
        "perfect result at rank 5": [fake(f"z{i}", 0) for i in range(4)]
        + [fake("c", 4)]
        + [fake(f"w{i}", 0) for i in range(5)],
    }
    print(f"{'ranking':<34} {'NDCG@10':>8} {'MRR':>7} {'MAP':>7} {'ERR':>7}")
    for label, ranked in lists.items():
        query = Query("q", tuple(ranked))
        scores = {document.doc_id: -index for index, document in enumerate(ranked)}
        row = evaluate([query], lambda document: scores[document.doc_id], k=10)
        print(
            f"{label:<34} {row['ndcg@k']:>8.3f} {row['mrr']:>7.3f} {row['map']:>7.3f} {row['err']:>7.3f}"
        )
    print(
        "\nMRR loves the single perfect hit and is indifferent to everything below it. MAP throws the\n"
        "grades away, so five grade-2 documents beat one grade-4. Reporting one number picks a user\n"
        "model without saying so."
    )


def demo_losses() -> None:
    rule("POINTWISE VS PAIRWISE VS LISTWISE -- identical model class, three objectives")
    dataset = make_dataset(queries=300, documents_per_query=20, seed=0)
    train, test = dataset.split(0.7, seed=0)
    print(f"{len(train)} training queries, {len(test)} test queries")
    print(f"label distribution: {dataset.label_distribution()}\n")

    rows = []
    rows.append(("BM25 only (baseline)", lambda document: document.features[0], None))

    pointwise, _ = train_pointwise(train, epochs=30, seed=0)
    rows.append(("pointwise regression", pointwise, pointwise))

    ranknet, ranknet_log = train_pairwise(train, epochs=25, lambda_weighting=False, seed=0)
    rows.append(("RankNet (pairwise)", ranknet, ranknet))

    lambdarank, lambda_log = train_pairwise(train, epochs=25, lambda_weighting=True, seed=0)
    rows.append(("LambdaRank (listwise)", lambdarank, lambdarank))

    print(f"{'model':<24} {'NDCG@10':>8} {'MRR':>7} {'ERR':>7} {'spread':>8}")
    for label, scorer, model in rows:
        metrics = evaluate(test.scorable(), scorer, k=10)
        spread = score_spread(test.scorable(), scorer)
        print(
            f"{label:<24} {metrics['ndcg@k']:>8.3f} {metrics['mrr']:>7.3f} "
            f"{metrics['err']:>7.3f} {spread:>8.3f}"
        )

    # the ceiling: a scorer that knows each query's intent
    per_intent = oracle_scorer(TRUE_WEIGHTS)
    values = []
    for query in test.scorable():
        values.append(mean_ndcg([query], per_intent(query), k=10))
    print(f"{'oracle (knows intent)':<24} {sum(values) / len(values):>8.3f}")

    print("\nlearned weights (normalised to unit max):")
    for label, model in (("pointwise", pointwise), ("RankNet", ranknet), ("LambdaRank", lambdarank)):
        weights = ", ".join(
            f"{name}={value:+.2f}"
            for name, value in zip(dataset.feature_names, model.normalised())
        )
        print(f"  {label:<12} {weights}")
    print(
        "\nlength_ratio carries no signal; a large weight on it is fitting noise.\n"
        "The oracle gap is structural: the true weights differ by intent, and one linear model\n"
        "cannot represent that. Presenting the best linear model as the ceiling would be dishonest."
    )
    print(f"\nRankNet pair updates: {ranknet_log.pairs_used}, LambdaRank: {lambda_log.pairs_used}")


def demo_bias() -> None:
    rule("POSITION BIAS -- clicks measure attention as much as relevance")
    dataset = make_dataset(queries=250, documents_per_query=20, seed=1)
    incumbent = feature_ranker((1.0, 0.2, 0.0, 0.2, 0.1, 0.0))  # a mediocre production ranker
    logs = simulate_clicks(dataset, incumbent, impressions_per_query=20, severity=1.0, seed=1)

    print("rank   examination p_k   observed CTR   estimated p_k (from CTR ratios)")
    estimated = estimate_propensities(logs)
    for rank, ctr in sorted(click_through_rates(logs).items()):
        print(
            f"{rank:>4}   {examination_probability(rank, 1.0):>15.3f}   {ctr:>12.4f}   "
            f"{estimated[rank]:>28.3f}"
        )
    print(
        "\nThe true p_k is 1/k. The estimate is close but systematically steep, because better\n"
        "documents sit higher under any competent ranker, so the CTR decay it measures mixes\n"
        "position bias with genuine relevance. Result randomisation is what separates them."
    )

    randomised = simulate_clicks(dataset, random_ranker(seed=2), impressions_per_query=20, seed=2)
    print("\nunder a uniformly random ranker, where relevance no longer correlates with rank:")
    random_estimate = estimate_propensities(randomised)
    for rank in (1, 3, 5, 10):
        if rank in random_estimate:
            print(
                f"  rank {rank:>2}: estimated {random_estimate[rank]:.3f}  "
                f"(true {examination_probability(rank, 1.0):.3f})"
            )
    print("Much closer -- and nobody is allowed to ship a random ranker. That is the whole tension.")


def demo_debias() -> None:
    rule("INVERSE PROPENSITY WEIGHTING -- undoing the incumbent's influence")
    dataset = make_dataset(queries=300, documents_per_query=20, seed=3)
    train, test = dataset.split(0.7, seed=3)
    incumbent = feature_ranker((1.0, 0.2, 0.0, 0.2, 0.1, 0.0))
    logs = simulate_clicks(train, incumbent, impressions_per_query=25, severity=1.0, seed=3)
    propensities = estimate_propensities(logs)

    naive, _ = train_from_clicks(logs, train, propensities=None, epochs=15, seed=3)
    corrected, _ = train_from_clicks(logs, train, propensities=propensities, epochs=15, seed=3)
    true_propensities = {rank: examination_probability(rank, 1.0) for rank in range(1, 11)}
    oracle_weighted, _ = train_from_clicks(
        logs, train, propensities=true_propensities, epochs=15, seed=3
    )
    supervised, _ = train_pairwise(train, epochs=25, lambda_weighting=True, seed=3)

    print(f"{'trained on':<38} {'NDCG@10':>8} {'MRR':>7}")
    for label, scorer in (
        ("the incumbent ranker itself", LinearScorer([1.0, 0.2, 0.0, 0.2, 0.1, 0.0])),
        ("raw clicks (no correction)", naive),
        ("clicks + estimated propensities", corrected),
        ("clicks + true propensities", oracle_weighted),
        ("graded labels (LambdaRank)", supervised),
    ):
        metrics = evaluate(test.scorable(), scorer, k=10)
        print(f"{label:<38} {metrics['ndcg@k']:>8.3f} {metrics['mrr']:>7.3f}")

    print("\nlearned weights (normalised):")
    for label, model in (("raw clicks", naive), ("IPS corrected", corrected)):
        weights = ", ".join(
            f"{name}={value:+.2f}" for name, value in zip(dataset.feature_names, model.normalised())
        )
        print(f"  {label:<14} {weights}")
    print(
        "\nRaw clicks pull the model towards whatever the incumbent already ranked highly -- here, BM25,\n"
        "because that is what the incumbent used. Weighting by 1/p_k removes most of that pull even with\n"
        "a biased propensity estimate, which is the practical finding: an imperfect correction beats\n"
        "ignoring the problem, and the graded-label model remains the ceiling."
    )


DEMOS = {
    "metrics": demo_metrics,
    "losses": demo_losses,
    "bias": demo_bias,
    "debias": demo_debias,
}


def main(argv: "list[str] | None" = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    choice = arguments[0] if arguments else "all"
    if choice == "all":
        for demo in DEMOS.values():
            demo()
        return 0
    if choice not in DEMOS:
        print(f"unknown demo {choice!r}\navailable: {', '.join(DEMOS)}, all")
        return 2
    DEMOS[choice]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
