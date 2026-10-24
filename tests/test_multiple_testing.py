import pytest

from niftygex_eval.multiple_testing import adjust_family, holm


def test_holm_adjustment_known_values() -> None:
    with_missing = holm([0.01, 0.04, 0.03, None])
    assert with_missing[:3] == pytest.approx([0.03, 0.06, 0.06]) and with_missing[3] is None
    adjusted = holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    family = adjust_family("F", "x", [{"test": "a", "p": 0.01}, {"test": "b", "p": 0.04}, {"test": "c", "p": 0.03}], 0.05)
    assert [m["reject_holm"] for m in family["members"]] == [True, False, False]
