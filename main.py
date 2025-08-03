from clean_text_pipeline import clean_pipeline

def main():
    with open("raw_corpus.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned, stats = clean_pipeline(lines)

    with open("clean_corpus.txt", "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line.strip() + "\n")

    total_tokens = sum(len(l.split()) for l in cleaned)
    removed = stats["total_lines"] - stats["retained_lines"]

    with open("stats.md", "w") as f:
        f.write(f"# Cleaning Statistics\n")
        f.write(f"- Total input lines: {stats['total_lines']}\n")
        f.write(f"- Retained lines: {stats['retained_lines']}\n")
        f.write(f"- Removed (non-English): {stats['removed_lang']}\n")
        f.write(f"- Removed (duplicates): {stats['removed_dupes']}\n")
        f.write(f"- Total tokens: {total_tokens}\n")
        f.write(f"- Removal percentage: {removed / stats['total_lines'] * 100:.2f}%\n")

if __name__ == "__main__":
    main()
