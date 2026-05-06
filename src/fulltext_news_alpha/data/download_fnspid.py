"""FNSPID dataset interface.

This module intentionally does not download data by default. The first project
version keeps reproducible interfaces while avoiding accidental large transfers.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FNSPID_SOURCES = {
    "paper": "https://arxiv.org/abs/2402.06698",
    "github": "https://github.com/Zdong104/FNSPID_Financial_News_Dataset",
    "huggingface": "https://huggingface.co/datasets/Zihan1004/FNSPID",
}


def validate_fnspid_root(raw_dir: str | Path) -> dict[str, Path | None]:
    """Return expected local paths if a user has already placed FNSPID files."""

    root = Path(raw_dir)
    candidates = {
        "news": next(root.glob("*news*.parquet"), None) if root.exists() else None,
        "prices": next(root.glob("*price*.parquet"), None) if root.exists() else None,
    }
    return candidates


def print_download_instructions(raw_dir: str | Path) -> None:
    root = Path(raw_dir)
    print("FNSPID download is intentionally not executed by this interface.")
    print(f"Place FNSPID news and price files under: {root}")
    print("References:")
    for name, url in FNSPID_SOURCES.items():
        print(f"- {name}: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or explain FNSPID data placement.")
    parser.add_argument("--raw-dir", default="data/raw/fnspid")
    args = parser.parse_args()
    found = validate_fnspid_root(args.raw_dir)
    print_download_instructions(args.raw_dir)
    print(f"Detected local files: {found}")


if __name__ == "__main__":
    main()
