import math
from importlib.resources import files

import pandas as pd
import pytest

from bolingual.experiment import metrics_from_ranks, rank_of_word
from bolingual.models import CandidateIndex
from bolingual.phonetics import weighted_similarity


def test_packaged_benchmark_supports_retrieval() -> None:
    benchmark = pd.read_csv(files("bolingual.data").joinpath("benchmark.csv"))
    assert len(benchmark) > 2000
    assert not benchmark["hindi"].duplicated().any()
    assert set(benchmark["split"]) == {"dev", "test"}
    index = CandidateIndex.from_words(benchmark["gold"].unique())
    rankings = index.hybrid_ranking("वॉटसन", top_k=20)
    assert len(rankings["hybrid"]) == 20
    assert "watson" in [word for _, word in rankings["hybrid"][:5]]
    assert all(0 <= score <= 1 for score, _ in rankings["hybrid"])


def test_metrics_count_missing_candidates_as_misses() -> None:
    metrics = metrics_from_ranks(pd.Series([1.0, 2.0, 10.0, math.inf]))
    assert metrics == pytest.approx({"top1": 0.25, "top5": 0.5, "top10": 0.75, "mrr": 0.4})
    assert rank_of_word([(0.9, "cat"), (0.8, "dog")], "dog") == 2
    assert math.isinf(rank_of_word([(0.9, "cat")], "absent"))


@pytest.mark.parametrize("tokens", [(), ("v", "a", "t"), ("k",)])
def test_identical_pronunciations_have_perfect_similarity(tokens: tuple[str, ...]) -> None:
    assert weighted_similarity(tokens, tokens) == 1.0


def test_index_deduplicates_and_handles_empty_vocabulary() -> None:
    assert CandidateIndex.from_words(["cat", "dog", "cat"]).vocabulary == ["cat", "dog"]
    assert CandidateIndex.from_words([]).hybrid_ranking("वॉटसन") == {
        "orthographic": [],
        "phonetic": [],
        "hybrid": [],
    }
