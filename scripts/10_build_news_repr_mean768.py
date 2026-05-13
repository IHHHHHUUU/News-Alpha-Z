"""CLI entry: build the raw 768-d stock-day mean-pooled news embedding."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.features.build_news_repr_mean768 import main


if __name__ == "__main__":
    main()
