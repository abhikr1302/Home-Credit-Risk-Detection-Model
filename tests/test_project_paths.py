from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_required_project_paths_exist():
    required_paths = [
        REPO_ROOT / "src" / "config.py",
        REPO_ROOT / "src" / "data_ingestion.py",
        REPO_ROOT / "src" / "feature_engineering.py",
        REPO_ROOT / "src" / "feature_selection.py",
        REPO_ROOT / "src" / "train_model.py",
        REPO_ROOT / "src" / "api.py",
        REPO_ROOT / "app" / "main.py",
        REPO_ROOT / "app" / "schemas.py",
        REPO_ROOT / "ui" / "streamlit_app.py",
    ]

    for path in required_paths:
        assert path.exists(), f"Required project path is missing: {path}"