import sys
from pathlib import Path

# Ensure 'src' is in sys.path even when executed directly with python main.py
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langchain_app.main import run_cli

if __name__ == "__main__":
    run_cli()
