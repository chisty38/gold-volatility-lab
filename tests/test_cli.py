import json

import pytest

from gold_volatility import __version__
from gold_volatility.cli import main
from gold_volatility.data import MergeStats


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])

    assert capsys.readouterr().out.strip() == __version__


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])

    assert "required" in capsys.readouterr().err


def test_fetch_reports_what_changed(monkeypatch, capsys, tmp_path, bars):
    # No network: replace the download with the simulated bars.
    monkeypatch.setattr(
        "gold_volatility.cli.update_cache",
        lambda path: (bars, MergeStats(added=30, revised=0, total=len(bars))),
    )

    main(["--cache", str(tmp_path / "cache.csv"), "fetch"])

    out = capsys.readouterr().out
    assert "+30 new bars, 0 revised" in out
    assert "Latest close: $" in out


def test_evaluate_writes_every_report(capsys, tmp_path, long_bars):
    cache = tmp_path / "cache.csv"
    long_bars.to_csv(cache)
    reports = tmp_path / "reports"

    main(["--cache", str(cache), "evaluate", "--reports", str(reports)])

    written = {p.name for p in reports.iterdir()}
    assert written == {
        "real_chart.png",
        "real_folds.csv",
        "real_metrics.csv",
        "real_predictions.csv",
        "real_run.json",
    }

    record = json.loads((reports / "real_run.json").read_text(encoding="utf-8"))
    assert len(record["data"]["cache_sha256"]) == 64
    assert record["data"]["out_of_sample_forecasts"] > 0
    assert "OUT OF SAMPLE" in capsys.readouterr().out
