import unicodedata
import re


def normalize_name(s: str) -> str:
    if s is None or type(s) is float:
        return ""
    # 1) Unicode normalize & strip accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # 2) Lowercase
    s = s.lower()
    # 3) Replace punctuation you want to ignore (apostrophes, hyphens, periods) with nothing
    s = re.sub(r"[\'\.\-]", "", s)
    # 4) Collapse any non-letters to a single space (keeps letters & digits)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # 5) Trim & collapse whitespace
    s = " ".join(s.split())
    return s
