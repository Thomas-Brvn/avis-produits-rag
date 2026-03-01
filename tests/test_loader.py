import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import ReviewLoader


SAMPLE_PATH = Path(__file__).parent.parent / "data"


def test_load_json_returns_dataframe():
    loader = ReviewLoader(data_path=str(SAMPLE_PATH))
    df = loader.load_json("sample_reviews.json")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_json_has_required_columns():
    loader = ReviewLoader(data_path=str(SAMPLE_PATH))
    df = loader.load_json("sample_reviews.json")
    assert "reviewText" in df.columns
    assert "rating" in df.columns
    assert "asin" in df.columns


def test_load_dispatches_by_extension():
    loader = ReviewLoader(data_path=str(SAMPLE_PATH))
    df = loader.load("sample_reviews.json")
    assert isinstance(df, pd.DataFrame)


def test_load_unsupported_format_raises():
    loader = ReviewLoader(data_path=str(SAMPLE_PATH))
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load("file.txt")


def test_list_files_returns_list():
    loader = ReviewLoader(data_path=str(SAMPLE_PATH))
    files = loader.list_files()
    assert isinstance(files, list)
    assert "sample_reviews.json" in files
