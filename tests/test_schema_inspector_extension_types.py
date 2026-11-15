from __future__ import annotations

import pandas as pd

from src.data.schema_inspector import SchemaInspector


def test_schema_inspector_handles_pandas_extension_and_mixed_dtypes():
    frame = pd.DataFrame({
        "label": pd.Series(["pair", pd.NA, "mirror"], dtype="string"),
        "count": pd.Series([1, pd.NA, 3], dtype="Int64"),
        "enabled": pd.Series([True, pd.NA, False], dtype="boolean"),
        "category": pd.Series(["a", "b", None], dtype="category"),
        "created_at": pd.to_datetime(["2026-01-01", None, "2026-01-03"]),
        "mixed": pd.Series(["x", 2, None], dtype="object"),
        "empty": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
    })
    report = SchemaInspector(frame).run()
    columns = report["columns"]
    assert columns["label"]["role"] == "categorical"
    assert columns["count"]["role"] == "ordinal_int"
    assert columns["enabled"]["role"] == "boolean"
    assert columns["category"]["role"] == "categorical"
    assert columns["created_at"]["role"] == "datetime"
    assert columns["mixed"]["role"] == "categorical"
    assert columns["empty"]["missing"] == 3
    assert report["missing_values"]["total_missing"] == 9
    assert report["missing_values"]["total_missing_pct"] > 0


def test_schema_inspector_handles_empty_dataframe_and_zero_row_columns():
    zero_rows = SchemaInspector(pd.DataFrame({"name": pd.Series(dtype="string")})).run()
    assert zero_rows["shape"] == {"rows": 0, "columns": 1}
    assert zero_rows["columns"]["name"]["missing_pct"] == 0.0
    assert zero_rows["missing_values"]["total_missing_pct"] == 0.0
    assert zero_rows["duplicates"]["duplicate_pct"] == 0.0

    no_columns = SchemaInspector(pd.DataFrame()).run()
    assert no_columns["shape"] == {"rows": 0, "columns": 0}
    assert no_columns["columns"] == {}
    assert no_columns["data_quality_score"] == 100.0
