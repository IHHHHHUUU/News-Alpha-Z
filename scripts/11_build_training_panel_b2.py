"""CLI entry: build the B2 mean-pooling training panel."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.features.build_training_panel_b2 import main


if __name__ == "__main__":
    main()
