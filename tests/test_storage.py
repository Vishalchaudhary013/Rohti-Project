from src import storage


def test_register_and_list_uploads(tmp_path):
    db = tmp_path / "t.db"
    assert storage.list_uploads(db_path=db).empty
    first = storage.register_upload("a.csv", 10, 9, "abc", "accepted", "mat", db_path=db)
    second = storage.register_upload("b.csv", 5, 0, "def", "rejected", db_path=db)
    assert second > first
    hist = storage.list_uploads(db_path=db)
    assert list(hist["filename"]) == ["b.csv", "a.csv"]
    assert hist.loc[1, "rows_kept"] == 9
    assert hist.loc[0, "subject"] is None
    assert len(storage.list_uploads(limit=1, db_path=db)) == 1


def test_prediction_runs(tmp_path):
    db = tmp_path / "t.db"
    assert storage.list_prediction_runs(db_path=db).empty
    storage.log_prediction_run(
        "m.joblib",
        "Random Forest",
        "B",
        0.5,
        100,
        30,
        {"High": 10, "Medium": 20, "Low": 70},
        db_path=db,
    )
    storage.log_prediction_run("m.joblib", "Random Forest", "B", 0.4, 100, 45, {}, db_path=db)
    runs = storage.list_prediction_runs(db_path=db)
    assert list(runs["threshold"]) == [0.4, 0.5]
    assert runs.loc[1, "n_high"] == 10 and runs.loc[0, "n_low"] == 0
