"""回归门禁的契约测试：三种退出码各是什么条件下返回。

★ 为什么门禁本身要有测试：
  门禁最常见的失效方式是「永远返回 0」——那样它看起来一直很绿，
  实际上什么都没拦。所以这里既测「该拦的拦住了」，也测「不该拦的没拦」。
"""

from __future__ import annotations

import json

import pytest

from eval.compare import EXIT_ERROR, EXIT_OK, EXIT_REGRESSED, main


def write(path, *, mrr=0.7292, hit=0.875, k=3, with_llm=False, chunks=139, extra=None):
    data = {
        "generated_at": "2026-11-19T00:00:00+08:00",
        "k": k,
        "with_llm": with_llm,
        "corpus": {"chunks": chunks},
        "metrics": {
            "hit_rate_at_recall": 1.0,
            "hit_rate_at_k": hit,
            "mrr_at_k": mrr,
            "refusal_accuracy": 1.0,
            "false_refusal_rate": 0.0,
            "citation_precision": None,
            "answers_without_citation": None,
            "invalid_citations": None,
        },
    }
    if extra:
        data["metrics"].update(extra)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture
def pair(tmp_path):
    base = write(tmp_path / "baseline.json")
    cur = write(tmp_path / "current.json")
    return base, cur


def test_identical_results_pass(pair, capsys) -> None:
    base, cur = pair
    assert main(["--baseline", base, "--current", cur]) == EXIT_OK
    assert "[门禁] 通过" in capsys.readouterr().out


def test_rate_drop_is_regression(tmp_path, capsys) -> None:
    base = write(tmp_path / "b.json")
    cur = write(tmp_path / "c.json", mrr=0.60)

    assert main(["--baseline", base, "--current", cur]) == EXIT_REGRESSED
    out = capsys.readouterr().out
    assert "REGRESSED" in out
    assert "-12.92pp" in out


def test_drop_within_tolerance_passes(tmp_path) -> None:
    base = write(tmp_path / "b.json")
    cur = write(tmp_path / "c.json", mrr=0.70)      # 掉 2.92pp，阈值 5pp

    assert main(["--baseline", base, "--current", cur]) == EXIT_OK


def test_exact_tolerance_boundary_is_not_regression(tmp_path) -> None:
    base = write(tmp_path / "b.json")
    cur = write(tmp_path / "c.json", mrr=0.6792)    # 恰好 5.00pp

    assert main(["--baseline", base, "--current", cur, "--tolerance-pp", "5"]) == EXIT_OK


def test_count_metric_increase_is_regression(tmp_path) -> None:
    base = write(tmp_path / "b.json", extra={"invalid_citations": 0})
    cur = write(tmp_path / "c.json", extra={"invalid_citations": 2})

    assert main(["--baseline", base, "--current", cur]) == EXIT_REGRESSED


def test_count_metric_within_budget_passes(tmp_path) -> None:
    base = write(tmp_path / "b.json", extra={"invalid_citations": 0})
    cur = write(tmp_path / "c.json", extra={"invalid_citations": 2})

    assert main(["--baseline", base, "--current", cur, "--max-extra", "2"]) == EXIT_OK


def test_mismatched_k_is_error_not_regression(tmp_path, capsys) -> None:
    base = write(tmp_path / "b.json")
    cur = write(tmp_path / "c.json", k=5)

    assert main(["--baseline", base, "--current", cur]) == EXIT_ERROR
    assert "口径不一致" in capsys.readouterr().err


def test_mismatched_llm_mode_is_error(tmp_path) -> None:
    base = write(tmp_path / "b.json", with_llm=False)
    cur = write(tmp_path / "c.json", with_llm=True)

    assert main(["--baseline", base, "--current", cur]) == EXIT_ERROR


def test_corpus_size_change_is_error(tmp_path) -> None:
    base = write(tmp_path / "b.json", chunks=139)
    cur = write(tmp_path / "c.json", chunks=200)

    assert main(["--baseline", base, "--current", cur]) == EXIT_ERROR


def test_missing_file_is_error(tmp_path) -> None:
    cur = write(tmp_path / "c.json")
    assert main(["--baseline", str(tmp_path / "nope.json"), "--current", cur]) == EXIT_ERROR


def test_json_output_is_parsable(pair, capsys) -> None:
    base, cur = pair
    assert main(["--baseline", base, "--current", cur, "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["regressed"] is False
    assert any(row["name"] == "mrr_at_k" for row in payload["rows"])
