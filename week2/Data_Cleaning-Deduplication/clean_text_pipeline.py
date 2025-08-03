import re
from langdetect import detect
from datasketch import MinHash, MinHashLSH
from bs4 import BeautifulSoup
from tqdm import tqdm

# 参数配置
MINHASH_THRESHOLD = 0.7
NGRAM_WINDOW = 3
PII_PATTERNS = [
    r"\b[\w.-]+?@\w+?\.\w+?\b",  # email
    r"\b(?:\d[ -]*?){13,16}\b",  # credit card
    r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b"  # phone
]

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_pii(text):
    for pattern in PII_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text

def remove_repetitive_ngrams(text, n=NGRAM_WINDOW):
    tokens = text.split()
    seen = set()
    filtered = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        if ngram in seen:
            continue
        seen.add(ngram)
        filtered.extend(ngram)
    return ' '.join(filtered)

def get_minhash(text):
    m = MinHash(num_perm=128)
    for word in set(text.split()):
        m.update(word.encode('utf8'))
    return m

def clean_pipeline(lines):
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=128)
    cleaned = []
    stats = {
        "total_lines": 0,
        "retained_lines": 0,
        "removed_lang": 0,
        "removed_dupes": 0,
        "removed_pii": 0
    }

    for idx, line in enumerate(tqdm(lines)):
        stats["total_lines"] += 1
        line = line.strip()
        if not line or not is_english(line):
            stats["removed_lang"] += 1
            continue

        line = remove_html(line)
        line = remove_pii(line)
        line = remove_repetitive_ngrams(line)

        mh = get_minhash(line)
        if any(lsh.query(mh)):
            stats["removed_dupes"] += 1
            continue

        lsh.insert(f"line_{idx}", mh)
        cleaned.append(line)
        stats["retained_lines"] += 1

    return cleaned, stats
