import json
from pathlib import Path

import pandas as pd
import pytest

import bias_analysis.dataset as ds
from bias_analysis.dataset import (
    _resolve_paths,
    _validate_paths,
    load_engagement_jsonl,
    load_jsonl_dataset,
    load_predictit_data,
    load_user_demographics_jsonl,
    load_user_score_jsonl,
)


def _write_jsonl(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# load_jsonl_dataset
# ---------------------------------------------------------------------------


class TestLoadJsonlDataset:
    def test_loads_all_records(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"id": 1, "text": "foo"}, {"id": 2, "text": "bar"}])
        df = load_jsonl_dataset(p)
        assert len(df) == 2
        assert set(df.columns) == {"id", "text"}

    def test_max_rows_limits_output(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"id": i} for i in range(10)])
        df = load_jsonl_dataset(p, max_rows=3)
        assert len(df) == 3

    def test_skips_malformed_lines_and_keeps_valid(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"id": 1}\nnot_json\n{"id": 2}\n', encoding="utf-8")
        df = load_jsonl_dataset(p)
        assert len(df) == 2
        assert list(df["id"]) == [1, 2]

    def test_empty_file_returns_empty_dataframe(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        df = load_jsonl_dataset(p)
        assert df.empty

    def test_preserves_all_fields(self, tmp_path):
        p = tmp_path / "data.jsonl"
        record = {"tweet_id": "123", "text": "hello", "votes": 42}
        _write_jsonl(p, [record])
        df = load_jsonl_dataset(p)
        assert df.iloc[0]["tweet_id"] == "123"
        assert df.iloc[0]["votes"] == 42


# ---------------------------------------------------------------------------
# load_user_score_jsonl
# ---------------------------------------------------------------------------


class TestLoadUserScoreJsonl:
    def test_returns_user_id_and_score_columns(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        _write_jsonl(p, [{"user_1": 0.8}, {"user_2": -0.3}])
        df = load_user_score_jsonl(p)
        assert set(df.columns) == {"user_id", "score"}

    def test_correct_number_of_rows(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        _write_jsonl(p, [{"u1": 0.5}, {"u2": -0.5}, {"u3": 0.0}])
        df = load_user_score_jsonl(p)
        assert len(df) == 3

    def test_preserves_score_value(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        _write_jsonl(p, [{"u123": 0.75}])
        df = load_user_score_jsonl(p)
        assert df.iloc[0]["user_id"] == "u123"
        assert df.iloc[0]["score"] == pytest.approx(0.75)

    def test_max_rows_limits_output(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        _write_jsonl(p, [{"u" + str(i): float(i)} for i in range(10)])
        df = load_user_score_jsonl(p)
        assert len(df) == 10
        df2 = load_user_score_jsonl(p, max_rows=4)
        assert len(df2) == 4

    def test_empty_file_returns_empty_df_with_columns(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        p.write_text("", encoding="utf-8")
        df = load_user_score_jsonl(p)
        assert df.empty
        assert "user_id" in df.columns
        assert "score" in df.columns


# ---------------------------------------------------------------------------
# load_user_demographics_jsonl
# ---------------------------------------------------------------------------


_DEMO_RECORD = {
    "gender": {"male": 0.6, "female": 0.4},
    "age": {"<=18": 0.05, "19-29": 0.15, "30-39": 0.35, ">=40": 0.45},
    "org": {"non-org": 0.9, "is-org": 0.1},
}


class TestLoadUserDemographicsJsonl:
    def _write_demo(self, path: Path, uid: str = "u1") -> None:
        _write_jsonl(path, [{uid: _DEMO_RECORD}])

    def test_extracts_all_expected_columns(self, tmp_path):
        p = tmp_path / "demo.jsonl"
        self._write_demo(p)
        df = load_user_demographics_jsonl(p)
        expected = {
            "user_id",
            "gender_male_prob",
            "gender_female_prob",
            "age_under_29_prob",
            "age_30_39_prob",
            "age_40_over_prob",
            "org_non_org_prob",
            "org_is_org_prob",
        }
        assert expected.issubset(set(df.columns))

    def test_age_under_29_is_sum_of_two_brackets(self, tmp_path):
        p = tmp_path / "demo.jsonl"
        self._write_demo(p)
        df = load_user_demographics_jsonl(p)
        assert df.iloc[0]["age_under_29_prob"] == pytest.approx(0.05 + 0.15)

    def test_gender_probabilities_are_correct(self, tmp_path):
        p = tmp_path / "demo.jsonl"
        self._write_demo(p)
        df = load_user_demographics_jsonl(p)
        assert df.iloc[0]["gender_male_prob"] == pytest.approx(0.6)
        assert df.iloc[0]["gender_female_prob"] == pytest.approx(0.4)

    def test_multiple_users(self, tmp_path):
        p = tmp_path / "demo.jsonl"
        _write_jsonl(p, [{"u1": _DEMO_RECORD}, {"u2": _DEMO_RECORD}])
        df = load_user_demographics_jsonl(p)
        assert len(df) == 2

    def test_empty_file_returns_empty_dataframe(self, tmp_path):
        p = tmp_path / "demo.jsonl"
        p.write_text("", encoding="utf-8")
        df = load_user_demographics_jsonl(p)
        assert df.empty


# ---------------------------------------------------------------------------
# load_engagement_jsonl
# ---------------------------------------------------------------------------


class TestLoadEngagementJsonl:
    def test_returns_tweet_id_and_users_list_columns(self, tmp_path):
        p = tmp_path / "engage.jsonl"
        _write_jsonl(p, [{"tweet_1": [{"id": "u1"}, {"id": "u2"}]}])
        df = load_engagement_jsonl(p)
        assert set(df.columns) == {"tweet_id", "users_list"}

    def test_correct_tweet_id(self, tmp_path):
        p = tmp_path / "engage.jsonl"
        _write_jsonl(p, [{"tweet_abc": [{"id": "u1"}]}])
        df = load_engagement_jsonl(p)
        assert df.iloc[0]["tweet_id"] == "tweet_abc"

    def test_users_list_is_preserved(self, tmp_path):
        p = tmp_path / "engage.jsonl"
        users = [{"id": "u1"}, {"id": "u2"}]
        _write_jsonl(p, [{"t1": users}])
        df = load_engagement_jsonl(p)
        assert df.iloc[0]["users_list"] == users

    def test_empty_file_returns_empty_df_with_columns(self, tmp_path):
        p = tmp_path / "engage.jsonl"
        p.write_text("", encoding="utf-8")
        df = load_engagement_jsonl(p)
        assert df.empty
        assert "tweet_id" in df.columns
        assert "users_list" in df.columns


# ---------------------------------------------------------------------------
# _resolve_paths / _validate_paths
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_resolve_paths_builds_absolute_paths(self, tmp_path):
        result = _resolve_paths(tmp_path, ["a/b.jsonl", "c/d.jsonl"])
        assert result == [tmp_path / "a/b.jsonl", tmp_path / "c/d.jsonl"]

    def test_resolve_paths_empty_list(self, tmp_path):
        assert _resolve_paths(tmp_path, []) == []

    def test_validate_paths_raises_for_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.jsonl"
        with pytest.raises(FileNotFoundError, match="nonexistent.jsonl"):
            _validate_paths([missing], "test")

    def test_validate_paths_passes_for_existing_file(self, tmp_path):
        existing = tmp_path / "file.jsonl"
        existing.touch()
        _validate_paths([existing], "test")  # must not raise

    def test_validate_paths_raises_on_first_missing_in_list(self, tmp_path):
        good = tmp_path / "good.jsonl"
        good.touch()
        bad = tmp_path / "bad.jsonl"
        with pytest.raises(FileNotFoundError):
            _validate_paths([good, bad], "test")


# ---------------------------------------------------------------------------
# load_predictit_data
# ---------------------------------------------------------------------------


class TestLoadPredictitData:
    def test_missing_file_returns_empty_dataframe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ds, "RAW_DATA_DIR", tmp_path)
        df = load_predictit_data("us20")
        assert df.empty

    def test_loads_csv_and_parses_date(self, tmp_path, monkeypatch):
        predictit_dir = tmp_path / "predictit"
        predictit_dir.mkdir()
        (predictit_dir / "us20.csv").write_text(
            "Date (ET),Open Share Price,Close Share Price,Low Share Price,High Share Price\n"
            "2020-11-01,0.40,0.44,0.38,0.50\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(ds, "RAW_DATA_DIR", tmp_path)
        df = load_predictit_data("us20")
        assert not df.empty
        assert "date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_comma_decimal_is_converted(self, tmp_path, monkeypatch):
        predictit_dir = tmp_path / "predictit"
        predictit_dir.mkdir()
        (predictit_dir / "us20.csv").write_text(
            "Date (ET),Open Share Price,Close Share Price,Low Share Price,High Share Price\n"
            '2020-11-01,"0,40","0,44","0,38","0,50"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(ds, "RAW_DATA_DIR", tmp_path)
        df = load_predictit_data("us20")
        assert not df.empty
        assert df.iloc[0]["Open Share Price"] == pytest.approx(0.40)
        assert df.iloc[0]["High Share Price"] == pytest.approx(0.50)
