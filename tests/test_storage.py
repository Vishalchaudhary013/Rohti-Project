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
