"""CLI entry: build B3 stock-day chunk embedding memmap cache."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.training.build_chunk_memmap_cache import main


if __name__ == "__main__":
    main()
