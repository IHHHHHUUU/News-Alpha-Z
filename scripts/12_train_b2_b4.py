"""CLI entry: train the B2+B4 (mean pooling + conventional mixture) baseline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.training.train_b2_b4 import main


if __name__ == "__main__":
    main()
