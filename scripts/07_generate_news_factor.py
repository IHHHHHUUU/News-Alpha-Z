from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fulltext_news_alpha.factors.build_news_factor import main


if __name__ == "__main__":
    main()
