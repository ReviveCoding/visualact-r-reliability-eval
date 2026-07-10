from __future__ import annotations

import pytest

from visualact_r.config import load_config
from visualact_r.data.build import build_task_records
from visualact_r.data.fixtures import create_synthetic_fixture
from visualact_r.data.splits import build_splits


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def fixture_source(tmp_path):
    return create_synthetic_fixture(tmp_path / "fixture")


@pytest.fixture
def records(tmp_path, fixture_source, config):
    raw = build_task_records(
        fixture_source["annotations"], fixture_source["images_dir"], config, tmp_path / "data"
    )
    return build_splits(raw, config, str(tmp_path / "data"))
