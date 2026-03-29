import pandas as pd
import re
from collections import Counter

try:
    import pdfplumber
except:
    pdfplumber = None


def extract_text(file):
    name = file.name.lower()

    if name.endswith(".txt"):
        return file.read().decode("utf-8")

    if name.endswith(".pdf") and pdfplumber:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    raise ValueError("Unsupported unstructured file")


def text_to_table(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    rows = []

    for line in lines:
        parts = re.split(r"[,\t|;]", line)
        if len(parts) > 1:
            rows.append(parts)

    if not rows:
        words = re.findall(r"\w+", text)
        freq = Counter(words)
        return pd.DataFrame(freq.items(), columns=["Token", "Frequency"])

    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]

    df = pd.DataFrame(rows)
    df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df


def unstructured_to_dataframe(file):
    text = extract_text(file)
    df = text_to_table(text)
    return df
