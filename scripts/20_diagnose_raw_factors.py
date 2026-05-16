"""CLI entry: diagnose raw price-volume single factors."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.diagnostics.raw_factor_diagnostics import main


if __name__ == "__main__":
    main()
