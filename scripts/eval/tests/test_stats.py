import math

import pytest

import stats

try:
    from scipy.stats import binomtest
except Exception:  # scipy ausente: so valores conhecidos
    binomtest = None


def test_mcnemar_known_values():
    # b=10, w=0 -> 2 * 0.5**10
    assert stats.mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert stats.mcnemar_p(5, 5) == 1.0
    assert stats.mcnemar_p(0, 0) == 1.0
    # b=3, w=9: n=12, 2*P(X<=3)=2*(1+12+66+220)/4096
    assert stats.mcnemar_p(3, 9) == pytest.approx(2 * 299 / 4096)
    assert stats.mcnemar_p(9, 3) == stats.mcnemar_p(3, 9)


@pytest.mark.skipif(binomtest is None, reason="scipy ausente")
def test_mcnemar_vs_scipy():
    for b, w in [(0, 7), (4, 11), (30, 22), (120, 80)]:
        assert stats.mcnemar_p(b, w) == pytest.approx(binomtest(min(b, w), b + w, 0.5).pvalue, rel=1e-9)


def test_mcnemar_counts():
    a = [1, 1, 0, 0, 1, 0, 0, 1]
    b = [1, 0, 1, 1, 1, 1, 0, 1]
    r = stats.mcnemar_exact(a, b)
    assert (r["better"], r["worse"], r["equal"], r["n"]) == (3, 1, 4, 8)
    assert r["p"] == pytest.approx(stats.mcnemar_p(3, 1))
    with pytest.raises(ValueError):
        stats.mcnemar_exact([1], [1, 0])


def test_bootstrap_deterministic_and_ci():
    a = [0.0] * 200
    b = [1.0] * 100 + [0.0] * 100
    r1 = stats.paired_bootstrap(a, b, n_resamples=2000, seed=7)
    r2 = stats.paired_bootstrap(a, b, n_resamples=2000, seed=7)
    assert r1 == r2
    assert r1["delta"] == pytest.approx(0.5)
    assert r1["ci_low"] < 0.5 < r1["ci_high"]
    assert 0.4 < r1["ci_low"] and r1["ci_high"] < 0.6
    assert stats.DEFAULT_RESAMPLES == 10_000


def test_bootstrap_zero_delta_ci_contains_zero():
    x = [float(i % 5) for i in range(100)]
    r = stats.paired_bootstrap(x, x, n_resamples=500, seed=1)
    assert r["delta"] == 0 and r["ci_low"] == 0 == r["ci_high"]


def test_p95_nearest_rank_and_bootstrap():
    vals = list(range(1, 101))
    assert stats.nearest_rank(vals, 95) == 95.0
    r = stats.p95_bootstrap(vals, n_resamples=500, seed=3)
    assert r["p95"] == 95.0 and r["ci_low"] <= 95 <= r["ci_high"]


def test_holm():
    adj = stats.holm_bonferroni([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm_bonferroni([0.5]) == [0.5]
    assert all(0 <= x <= 1 for x in stats.holm_bonferroni([0.9, 0.8, 0.7]))


def test_superior_noninferior():
    assert stats.is_superior(p=0.03)
    assert not stats.is_superior(p=0.05)  # estrito
    assert stats.is_superior(ci_low=0.01)
    assert not stats.is_superior(p=0.2, ci_low=-0.1)
    assert stats.is_noninferior(-0.04, 0.05)
    assert not stats.is_noninferior(-0.05, 0.05)  # estrito
    assert not stats.is_noninferior(None, 0.05)
    assert not stats.is_noninferior(0.1, None)
