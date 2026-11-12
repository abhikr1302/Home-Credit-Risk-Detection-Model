"""Launch the canonical UI with: streamlit run app.py."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "ui" / "streamlit_app.py"), run_name="__main__")
