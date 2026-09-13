# Search Reranking: Learning to Rank

Ranking metrics that disagree with each other on purpose, three loss families on one model class, and
**unbiased learning from position-biased click logs**. Pure Python, standard library only — no LightGBM, no
XGBoost, no NumPy. The `lambdarank` objective is four lines once you see it, and importing it hides them.

```bash
python -m ltr.cli metrics    # NDCG, MRR, MAP and ERR ranking the same lists differently
python -m ltr.cli losses     # pointwise vs RankNet vs LambdaRank, identical model class
python -m ltr.cli bias       # position bias in a click log, and estimating it
python -m ltr.cli debias     # inverse propensity weighting vs raw clicks
```

## The metric you report picks a user model, whether or not you say so

```
ranking                             NDCG@10     MRR     MAP     ERR
one perfect result at rank 1          1.000   1.000   1.000   0.938
five decent results, none perfect     1.000   1.000   1.000   0.795
perfect result at rank 5              0.387   0.200   0.200   0.204
```

Two things this table shows immediately. NDCG is normalised **per query**, so it cannot compare rankings across
different candidate sets — the first two rows both score 1.0 because each is the best ordering of its own
documents. And within one candidate set the metrics genuinely conflict:

```
same two documents, two orderings           MRR     NDCG@10     ERR
grade-4 first, then grade-1                1.000       1.000   0.939
grade-1 first, then grade-4                1.000       0.669   0.502
```

MRR scores the inversion **perfectly**, because a grade-1 document still clears its relevance threshold at
rank 1. MAP does the same, for a different reason: it never looks at the grades at all. Both facts are asserted
as tests, and both are why a navigational metric must not be used to judge a graded ranking.

Three further choices that change an NDCG number without changing the ranking:

- **Gain function.** `2^rel - 1` makes a grade-4 worth 15 and a grade-1 worth 1. Linear gain makes them 4 and 1.
  Both are published as "NDCG". Two NDCG values from different sources are not comparable unless both the gain
  and the truncation are stated — and they usually are not.
- **Queries with no relevant document.** IDCG is 0, so NDCG is 0/0. Here `ndcg` returns `nan` and `mean_ndcg`
  excludes them; scoring them as 0.0 turns the metric into a property of the candidate generator.
- **Ties.** A model that scores every document identically gets whatever the sort produced. Ties break by
  document id here, and `score_spread` exists to expose that degeneracy rather than let it hide behind a
  plausible-looking loss curve.

## Three objectives, one model class

All three use the same linear scorer, so the comparison isolates the loss. Gradient boosting would beat all of
them and would also obscure which part of the gain came from the objective.

**Pointwise** — regress the grade. Structurally mismatched: it spends capacity getting the predicted grade of an
irrelevant document right, which no ranking metric rewards, and it pools labels across queries where a grade
means different things.

**Pairwise (RankNet)** — for `rel_i > rel_j`:

```
L        = log(1 + exp(-sigma * (s_i - s_j)))
dL/ds_i  = -sigma / (1 + exp(sigma * (s_i - s_j)))
```

Correct about which ordering it prefers, and indifferent to **where** the mistake is: an inversion between ranks
1 and 2 costs exactly what one between 19 and 20 costs. So it optimises something nobody reports.

**Listwise (LambdaRank)** — multiply that gradient by how much NDCG would move if the pair swapped:

```
lambda_ij <- lambda_ij * |delta NDCG_ij|
```

One multiplication, and the updates become proportional to the metric. Worth being precise about what it is: a
**heuristic** gradient, with no known loss function whose gradient it is. Wang et al. (2018) later showed it
optimises a bound on a metric-based objective — which is a weaker and more accurate claim than "optimises NDCG
directly". `delta_ndcg` also returns exactly zero when both positions lie outside the truncation, since swapping
ranks 19 and 20 cannot move NDCG@10.

```
model                     NDCG@10     MRR     ERR   spread
BM25 only (baseline)        0.681   0.812   0.611    0.418
pointwise regression        0.704   0.826   0.628    1.914
RankNet (pairwise)          0.769   0.869   0.681    2.402
LambdaRank (listwise)       0.784   0.881   0.694    2.556
oracle (knows intent)       0.871
```

> Illustrative and seed-dependent. The tests assert orderings and tolerances, never these digits.

The oracle gap is **structural, not a tuning failure**: the generator's true weights differ by query intent —
freshness is decisive for one intent and worthless for another — and no single linear model can represent that.
Presenting the best linear model as the ceiling would misrepresent the problem.

## Clicks are not labels

Real training data is a click log, and clicks measure attention as much as relevance:

```
P(click on d at rank k) = p_k * r_d          p_k = examination probability, falling steeply in k
```

```
rank   true p_k   observed CTR   estimated p_k
   1      1.000         0.3218           1.000
   2      0.500         0.1487           0.462
   5      0.200         0.0498           0.155
  10      0.100         0.0221           0.069
```

So a document's CTR confounds its relevance with **the rank the previous ranker gave it**. Train on raw clicks
and the model learns to reproduce the incumbent — including its mistakes, which get no clicks and therefore
never get corrected. That feedback loop is the central problem in production search.

The estimator above is the cheap one, and its assumption is severe: it treats average relevance per rank as
constant, which is false whenever the incumbent is any good, so the decay it measures is **systematically too
steep** (0.069 against a true 0.100 at rank 10). Under a uniformly random ranker the same estimator lands much
closer — and nobody may ship a random ranker. The data that identifies position bias cleanly is exactly the data
that costs the most revenue to collect.

## The correction

Weight each click by `1 / p_k` (Joachims et al., 2017), so a good document the incumbent buried is no longer
penalised for having been buried:

```
trained on                                NDCG@10     MRR
the incumbent ranker itself                 0.681   0.812
raw clicks (no correction)                  0.699   0.822
clicks + estimated propensities             0.742   0.851
clicks + true propensities                  0.751   0.858
graded labels (LambdaRank)                  0.784   0.881
```

Raw click training lands close to the incumbent, because that is what the clicks encode. An **imperfect**
propensity estimate recovers most of the gap — the practical finding here — and graded labels remain the ceiling.

Two implementation details do more work than the weighting itself:

1. **A clicked document is preferred only over unclicked documents shown *above* it.** Documents below a click
   may never have been examined. Treating them as negatives is the single most common way a click-trained ranker
   learns to freeze the incumbent ordering in place. `test_no_negatives_below_a_click` asserts that a log with
   only the top result clicked produces **zero** training pairs.
2. **Propensities must be floored.** `1/p_k` has unbounded variance; without a floor one click at rank 10 can
   outweigh a hundred at rank 1. `propensity_floor` is an explicit parameter, not a hidden constant, because it
   trades variance for bias and the caller should own that.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Exact: DCG by hand, average precision on a three-document list, ERR's cascade product, the analytic RankNet
gradient against a central finite difference, `delta_ndcg` symmetry and its zero outside the truncation, scale
invariance of the ranking under reweighting, and the exact pair counts for a click at rank 3 (two negatives) and
two adjacent clicks (none).

Directional, asserted with tolerance: trained models beating a pure-noise feature, the propensity estimator
measurably less biased under a random logging policy, IPS beating raw clicks, graded labels beating clicks, and
the zero-true-weight feature staying small in the learned model.

## Limits

- **Linear scorer.** Real systems use gradient-boosted trees or a cross-encoder; the losses are identical, the
  absolute numbers are not, and the intent-conditional structure a tree captures for free is exactly what the
  linear model's oracle gap measures.
- **Synthetic data with known ground truth**, which is what makes the debiasing claim measurable at all — and
  also means the position-bias model is *exactly* the one being corrected for. On real logs the examination
  model is wrong in unknown ways, and trust/attractiveness bias, presentation bias and selection bias all sit
  on top of position bias, uncorrected here.
- **A naive propensity estimator** whose bias is demonstrated rather than fixed. Production needs result
  randomisation or intervention harvesting.
- **No selection bias correction.** Documents the incumbent never showed at all get no clicks and no
  propensity — IPW cannot reweight a zero. This is a harder problem than position bias and is not addressed.
- **No diversity, no de-duplication, no fairness constraint.** Every metric here treats documents as
  independent, which is false: two near-identical results at ranks 1 and 2 waste a slot, and no metric in this
  repository notices.
- **Single-stage reranking.** Candidate generation, which decides what can be ranked at all, is out of scope.

## References

- Burges et al. (2005), *Learning to rank using gradient descent* — RankNet.
- Burges, Ragno & Le (2006), *Learning to rank with nonsmooth cost functions* — LambdaRank.
- Burges (2010), *From RankNet to LambdaRank to LambdaMART: an overview*.
- Wang et al. (2018), *The LambdaLoss framework for ranking metric optimization*.
- Järvelin & Kekäläinen (2002), *Cumulated gain-based evaluation of IR techniques* — DCG and NDCG.
- Chapelle et al. (2009), *Expected reciprocal rank for graded relevance*.
- Joachims, Swaminathan & Schnabel (2017), *Unbiased learning-to-rank with biased feedback*.
- Wang et al. (2018), *Position bias estimation for unbiased learning to rank in personal search*.
- Agarwal et al. (2019), *Estimating position bias without intrusive interventions*.
- Craswell et al. (2008), *An experimental comparison of click position-bias models*.
- Oosterhuis & de Rijke (2020), *Policy-aware unbiased learning to rank for top-k rankings*.

MIT licensed.
