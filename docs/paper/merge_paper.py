#!/usr/bin/env python3
"""
Merge all paper sections into a single Markdown document.

If the target output file already exists, creates a new uniquely named version
(e.g., final_paper_1.md, final_paper_2.md, etc.) in the same directory.
"""

from pathlib import Path
import sys

# Directory containing the paper sections
PAPER_DIR = Path(__file__).resolve().parent

# Canonical section order
SECTION_FILES = [
    "abstract/abstract.md",
    "1.introduction/introduction.md",
    "2.Related_Work/related_work.md",
    "3.method/method.md",
    "4.experimental_setup/experimental_setup.md",
    "5.Results/results.md",
    "6.discussion_limitations/discussion.md",
    "7.conclusion/conclusion.md",
]

BASE_OUTPUT_NAME = "final_paper.md"


def get_unique_output_path(base_dir: Path, filename: str) -> Path:
    """Return a non-colliding file path.
    If 'final_paper.md' exists, generates 'final_paper_1.md', 'final_paper_2.md', etc.
    """
    target = base_dir / filename
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = base_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def merge_sections(paper_dir: Path, output_file: Path) -> None:
    merged_parts = []
    missing_files = []

    for rel_path in SECTION_FILES:
        full_path = paper_dir / rel_path
        if not full_path.is_file():
            missing_files.append(str(rel_path))
            continue

        content = full_path.read_text(encoding="utf-8").strip()
        merged_parts.append(content)

    if missing_files:
        print(f"Error: The following required section files were not found:")
        for mf in missing_files:
            print(f"  - {mf}")
        sys.exit(1)

    # Join with standard spacing and clean divider
    full_document = "\n\n---\n\n".join(merged_parts) + "\n"

    output_file.write_text(full_document, encoding="utf-8")

    # Statistics
    lines = full_document.splitlines()
    words = full_document.split()
    print(f"Successfully merged {len(SECTION_FILES)} sections.")
    print(f"Output saved to: {output_file.resolve()}")
    print(f"Total lines: {len(lines)}")
    print(f"Total words: {len(words)}")


def main():
    target_path = get_unique_output_path(PAPER_DIR, BASE_OUTPUT_NAME)
    merge_sections(PAPER_DIR, target_path)


if __name__ == "__main__":
    main()
