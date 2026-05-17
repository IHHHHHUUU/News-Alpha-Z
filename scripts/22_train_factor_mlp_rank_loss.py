"""CLI entry: train factor-only MLP with daily RankIC loss."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.training.factor_mlp_rank_loss import main


if __name__ == "__main__":
    main()
