from src import config


def test_schema_has_33_attributes():
    assert len(config.SCHEMA) == 33
    assert set(config.NUMERIC_COLUMNS) | set(config.CATEGORICAL_COLUMNS) == set(config.SCHEMA)


def test_absence_bands_consistent():
    assert len(config.ABSENCE_BAND_EDGES) == len(config.ABSENCE_BAND_LABELS) + 1


def test_paths_are_inside_project():
    for d in config.ALL_DIRS:
        assert config.PROJECT_ROOT in d.parents


def test_risk_thresholds_ordered():
    assert 0 < config.RISK_MEDIUM < config.RISK_HIGH < 1
