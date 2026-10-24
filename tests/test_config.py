from pathlib import Path

from niftygex.config import REPO_ROOT, config_hash, load_config


def test_local_profile_resolves_repo_data_root() -> None:
    config = load_config("local")
    assert config["paths"]["data_root"] == REPO_ROOT / "data"
    assert config["data"]["snapshot_grid"]["t2_daily"] == "15:15"
    assert len(config_hash(config)) == 64


def test_data_root_environment_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NIFTYGEX_DATA_ROOT", str(tmp_path))
    config = load_config("local")
    assert config["paths"]["data_root"] == tmp_path
    assert config["paths"]["raw_root"] == tmp_path / "raw"


def test_git_provenance_reports_commit() -> None:
    from niftygex.io.provenance import git_provenance

    info = git_provenance()
    assert set(info) == {"git_commit", "git_dirty"}
    assert info["git_commit"] is None or len(info["git_commit"]) == 40
